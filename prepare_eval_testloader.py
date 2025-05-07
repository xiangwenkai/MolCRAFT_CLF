import os
import pickle
from rdkit import Chem
import torch
from core.datasets.pl_pair_dataset import PocketLigandPairDataset
from core.datasets.pl_data import ProteinLigandData
import core.utils.transforms as trans
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
import lmdb
from prepare_data_mask import Mask
from core.datasets.pl_data import FOLLOW_BATCH
from torch_geometric.loader import DataLoader
import json
import shutil

import pandas as pd
# from sklearn.metrics import roc_auc_score
import argparse
import random

from core.config.config import Config, parse_config
from core.models.sbdd_train_loop import SBDDTrainLoop
from core.callbacks.basic import NormalizerCallback
from core.callbacks.validation_callback_for_sample import (
    DockingTestCallback,
    OUT_DIR
)

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from core.evaluation.utils import scoring_func
from core.evaluation.docking_vina import VinaDockingTask
from posecheck import PoseCheck
import numpy as np

random.seed(42)


features_to_save = [
                'protein_pos', 'protein_atom_feature', 'protein_element',
                'ligand_pos', 'ligand_atom_feature_full', 'ligand_element',
                'protein_filename', 'ligand_filename', 'lig_emb', 'mask_indexes'
            ]
ligand_atom_mode = 'add_aromatic'
protein_featurizer = trans.FeaturizeProteinAtom()
ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
transform_list = [
    protein_featurizer,
    ligand_featurizer,
    # trans.FeaturizeLigandBond(),
]
transform = Compose(transform_list)

def _transform_subset(ids):
    data_list = []

    for idx in tqdm(ids):
        data = raw_dataset[idx]
        data = transform(data)
        tr_data = {}
        for k in features_to_save:
            tr_data[k] = getattr(data, k)
        tr_data['id'] = idx
        tr_data = ProteinLigandData(**tr_data)
        data_list.append(tr_data)
    return data_list


def call(test_loader, ckpt_path='./checkpoints/last.ckpt',
         num_samples=10, sample_steps=100, sample_num_atoms='ref',
         beta1=1.5, sigma1_coord=0.03, sampling_strategy='end_back', seed=1234):
    cfg = Config('./checkpoints/config.yaml')
    seed_everything(cfg.seed)

    # cfg.evaluation.emb = emb_info
    # cfg.evaluation.ckpt_path = ckpt_path
    cfg.test_only = True
    cfg.no_wandb = True
    cfg.evaluation.num_samples = num_samples
    cfg.evaluation.sample_steps = sample_steps
    cfg.evaluation.sample_num_atoms = sample_num_atoms  # or 'prior'
    cfg.dynamics.beta1 = beta1
    cfg.dynamics.sigma1_coord = sigma1_coord
    cfg.dynamics.sampling_strategy = sampling_strategy
    cfg.seed = seed
    cfg.train.max_grad_norm = 'Q'

    # print(f"The config of this process is:\n{cfg}")
    # wandb_logger.log_hyperparams(cfg.todict())

    model = SBDDTrainLoop(config=cfg)  # !!!!!!!!!!!!!!!!!! prop
    # model.to('cuda')
    # outputs = []
    # for batch_idx, batch in enumerate(test_loader):
    #     batch.to('cuda')
    #     out = model.test_step(batch, batch_idx)
    #     outputs.extend(out)
    # on_test_epoch_end(outputs)

    trainer = pl.Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        devices=1,
        # logger=wandb_logger,
        num_sanity_val_steps=0,
        callbacks=[
            NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
            DockingTestCallback(
                dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=True,
                docking_config=cfg.evaluation.docking_config,
            ),
        ],
    )

    trainer.test(model, dataloaders=test_loader, ckpt_path=cfg.evaluation.ckpt_path)


class Metrics:
    def __init__(self, protein_fn, ref_ligand_fn, ligand_fn):
        self.protein_fn = protein_fn
        self.ref_ligand_fn = ref_ligand_fn
        self.ligand_fn = ligand_fn
        self.exhaustiveness = 16

    def vina_dock(self, mol):
        chem_results = {}

        try:
            # qed, logp, sa, lipinski, ring size, etc
            chem_results.update(scoring_func.get_chem(mol))
            chem_results['atom_num'] = mol.GetNumAtoms()

            # docking
            vina_task = VinaDockingTask.from_generated_mol(mol, ligand_filename=self.ref_ligand_fn, protein_root='./')
            score_only_results = vina_task.run(mode='score_only', exhaustiveness=self.exhaustiveness)
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=self.exhaustiveness)
            docking_results = vina_task.run(mode='dock', exhaustiveness=self.exhaustiveness)

            chem_results['vina_score'] = score_only_results[0]['affinity']
            chem_results['vina_minimize'] = minimize_results[0]['affinity']
            chem_results['vina_dock'] = docking_results[0]['affinity']
            chem_results['vina_dock_pose'] = docking_results[0]['pose']
            return chem_results
        except Exception as e:
            print("vina score failed")
            print(e)

        return chem_results

    def pose_check(self, mol):
        pc = PoseCheck()

        pose_check_results = {}

        protein_ready = False
        try:
            pc.load_protein_from_pdb(self.protein_fn)
            protein_ready = True
        except ValueError as e:
            return pose_check_results

        ligand_ready = False
        try:
            pc.load_ligands_from_mols([mol])
            ligand_ready = True
        except ValueError as e:
            return pose_check_results

        if ligand_ready:
            try:
                strain = pc.calculate_strain_energy()[0]
                pose_check_results['strain'] = strain
            except Exception as e:
                pass

        if protein_ready and ligand_ready:
            try:
                clash = pc.calculate_clashes()[0]
                pose_check_results['clash'] = clash
            except Exception as e:
                pass

            try:
                df = pc.calculate_interactions()
                columns = np.array([column[2] for column in df.columns])
                flags = np.array([df[column][0] for column in df.columns])

                def count_inter(inter_type):
                    if len(columns) == 0:
                        return 0
                    count = sum((columns == inter_type) & flags)
                    return count

                # ['Hydrophobic', 'HBDonor', 'VdWContact', 'HBAcceptor']
                hb_donor = count_inter('HBDonor')
                hb_acceptor = count_inter('HBAcceptor')
                vdw = count_inter('VdWContact')
                hydrophobic = count_inter('Hydrophobic')

                pose_check_results['hb_donor'] = hb_donor
                pose_check_results['hb_acceptor'] = hb_acceptor
                pose_check_results['vdw'] = vdw
                pose_check_results['hydrophobic'] = hydrophobic
            except Exception as e:
                pass

        for k, v in pose_check_results.items():
            mol.SetProp(k, str(v))

        return pose_check_results

    def evaluate(self):
        mol = Chem.SDMolSupplier(self.ligand_fn, removeHs=False)[0]

        chem_results = self.vina_dock(mol)
        try:
            pose_check_results = self.pose_check(mol)
            chem_results.update(pose_check_results)
        except:
            print(f"pose check fail !!!!!!!")

        return chem_results


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)



split_path = f'/data4/wenkai/MolCRAFT_CLF/data/crossdocked_pocket10_pose_split.pt'
split = torch.load(split_path)
train_ids, test_ids = split['train'], split['test']


crossdock_path = '/data4/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10'

db = lmdb.open('/data4/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final_origin.lmdb', map_size=10 * (1024 * 1024 * 1024),
                   create=False,subdir=False,readonly=True,lock=False,readahead=False,meminit=False)
with db.begin() as txn:
    keys = list(txn.cursor().iternext(values=False))
for scenario in ['frag', 'link', 'scaffold', 'denovo']:
    db_new = lmdb.open(f'/data4/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10_processed_test{scenario}.lmdb',map_size=1 * (1024 * 1024 * 1024),  #
                           create=False,subdir=False,readonly=False,lock=False,readahead=False,meminit=False)
    txn_new = db_new.begin(write=True, buffers=True)
    for key in keys:
        if int(key) not in set(test_ids):
            continue
        data = pickle.loads(db.begin().get(key))
        protein_filename = data['protein_filename']
        ligand_filename = data['ligand_filename']
        protein_path = os.path.join(crossdock_path, protein_filename)
        ligand_path = os.path.join(crossdock_path, ligand_filename)

        # mol = Chem.MolFromMolFile(ligand_path, sanitize=False)
        # try:
        #     mol = Chem.RemoveHs(mol)
        # except:
        #     pass
        # if data['ligand_element'].size()[0] != mol.GetNumAtoms():
        #     fail_idxes.append(idx)

        try:
            mol = Chem.MolFromMolFile(ligand_path, sanitize=False)
            try:
                mol = Chem.RemoveHs(mol)
            except:
                pass
            # Chem.SanitizeMol(mol)
            # mol.UpdatePropertyCache(strict=False)
            mask = Mask(mol)
            if scenario == 'frag':
                mask_idxes = mask.get_frag_mask()
            elif scenario == 'link':
                mask_idxes = mask.get_link_mask() + mask.get_single_link_mask()
            elif scenario == 'scaffold':
                mask_idxes = mask.get_scaffold_side_chain_mask()
            elif scenario == 'denovo':
                mask_idxes = []
        except:
            print(f"{key} failed")
            mask_idxes = []
        data['mask_indexes'] = mask_idxes
        txn_new.put(
            key=key,
            value=pickle.dumps(data)
        )
    txn_new.commit()


for scenario in ['frag', 'link', 'scaffold', 'denovo']:
    raw_dataset = PocketLigandPairDataset(f"/data4/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10", None, f'test{scenario}')
    test_data = _transform_subset([i for i in range(len(raw_dataset))])
    torch.save({
        'test': test_data,
        'protein_atom_feature_dim': 27,
        'ligand_atom_feature_dim': 13,
    }, f'/data4/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10_transformed_test{scenario}.pt')


for scenario in ['frag', 'link', 'scaffold', 'denovo']:
    test_data = torch.load(f'/data4/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10_transformed_test{scenario}.pt')
    test_loader = DataLoader(
        test_data,
        batch_size=100,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=["ligand_nbh_list"]
    )
    call(test_loader)

