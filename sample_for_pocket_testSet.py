# import argparse
import os
import sys
import json
import shutil

import pandas as pd
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import torch

# from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from prepare_data_mask import Mask
import argparse

import torch
import random
# import datetime, pytz

from core.config.config import Config, parse_config
from core.models.sbdd_train_loop import SBDDTrainLoop
from core.callbacks.basic import NormalizerCallback
from core.callbacks.validation_callback_for_sample import (
    DockingTestCallback,
    OUT_DIR
)

import core.utils.transforms as trans
from core.datasets.utils import PDBProtein, parse_sdf_file
from core.datasets.pl_data import ProteinLigandData, torchify_dict
from core.datasets.pl_data import FOLLOW_BATCH
from core.callbacks.validation_callback import reconstruct_mol_and_filter_invalid

import pytorch_lightning as pl

from pytorch_lightning import seed_everything

# from absl import logging
# import glob

from core.evaluation.utils import scoring_func
from core.evaluation.docking_vina import VinaDockingTask
from posecheck import PoseCheck
import numpy as np
from rdkit import Chem
random.seed(42)


def get_dataloader_from_pdb(cfg, emb_info, guide_index):
    assert cfg.evaluation.protein_path is not None and cfg.evaluation.ligand_path is not None
    protein_fn, ligand_fn = cfg.evaluation.protein_path, cfg.evaluation.ligand_path

    # load protein and ligand
    protein = PDBProtein(protein_fn)
    ligand_dict = parse_sdf_file(ligand_fn)
    lig_pos = ligand_dict["pos"]

    print('[DEBUG] get_dataloader')
    print(lig_pos.shape, lig_pos.mean(axis=0))

    pdb_block_pocket = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand_dict, cfg.dynamics.net_config.r_max)
    )
    pocket = PDBProtein(pdb_block_pocket)
    pocket_dict = pocket.to_dict_atom()

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = protein_fn
    data.ligand_filename = ligand_fn

    data.lig_emb = emb_info
    data.mask_indexes = guide_index

    # transform
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
    ]
    transform = Compose(transform_list)
    cfg.dynamics.protein_atom_feature_dim = protein_featurizer.feature_dim
    cfg.dynamics.ligand_atom_feature_dim = ligand_featurizer.feature_dim
    print(f"protein feature dim: {cfg.dynamics.protein_atom_feature_dim}, " +
            f"ligand feature dim: {cfg.dynamics.ligand_atom_feature_dim}")

    # dataloader
    collate_exclude_keys = ["ligand_nbh_list"]
    test_set = [transform(data)] * cfg.evaluation.num_samples
    cfg.evaluation.num_samples = 1
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    )

    cfg.evaluation.docking_config.protein_root = os.path.dirname(os.path.abspath(protein_fn))
    print(f"protein root: {cfg.evaluation.docking_config.protein_root}")

    return test_loader

def on_test_epoch_end(outputs):
        results, recon_dict = reconstruct_mol_and_filter_invalid(outputs)

        OUT_DIR = './output'
        if os.path.exists(OUT_DIR):
            shutil.rmtree(OUT_DIR)
        os.makedirs(OUT_DIR, exist_ok=True)

        for idx, res in enumerate(tqdm(results, desc="Chem eval")):
            try:
                mol = res['mol']
                ligand_filename = res['ligand_filename']
                mol.SetProp('_Name', ligand_filename)

                Chem.SanitizeMol(mol)
                smiles = Chem.MolToSmiles(mol)
                validity = smiles is not None
                complete = '.' not in smiles
            except:
                print('sanitize failed')
                continue

            if not validity or not complete:
                print('validity', validity, 'complete', complete)
                continue

            # ligand_filename = graph.ligand_filename
            # ligand_dir = os.path.dirname(ligand_filename)
            # ligand_fn = os.path.basename(ligand_filename)
            # protein_fn = os.path.join(ligand_dir, ligand_fn[:10] + '.pdb')
            # print(json.dumps(chem_results, indent=4, cls=NpEncoder))
            out_fn = os.path.join(OUT_DIR, f'{idx}.sdf')
            with Chem.SDWriter(out_fn) as w:
                w.write(mol)

def call(protein_fn, ligand_fn, emb_info, guide_index, ckpt_path='./checkpoints/last.ckpt',
         num_samples=10, sample_steps=100, sample_num_atoms='ref',
         beta1=1.5, sigma1_coord=0.03, sampling_strategy='end_back', seed=1234):
    
    cfg = Config('./checkpoints/config.yaml')
    seed_everything(cfg.seed)
    
    cfg.evaluation.protein_path = protein_fn
    cfg.evaluation.ligand_path = ligand_fn
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

    print(protein_fn, ligand_fn)
    test_loader = get_dataloader_from_pdb(cfg, emb_info, guide_index)
    # wandb_logger.log_hyperparams(cfg.todict())

    model = SBDDTrainLoop(config=cfg)  # !!!!!!!!!!!!!!!!!! prop
    # model.to('cuda')
    # outputs = []
    # for batch_idx, batch in enumerate(test_loader):
    #     batch.to('cuda')
    #     out = model.test_step(batch, batch_idx)
    #     outputs.extend(out)
    # on_test_epoch_end(outputs)

    '''
    因为pytorch_lighting包，下列位置:
    /home/xiang_wenkai/anaconda3/envs/molcraft/lib/python3.9/site-packages/pytorch_lightning/loops/evaluation_loop.py line 108 
    中限定了no grad，因此无法在生成样本时加入预训练模型的梯度引导，所以此处绕开pytorch_lighting包
    '''
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
        pose_check_results = self.pose_check(mol)
        chem_results.update(pose_check_results)

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



if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=4 python sample_for_pocket_testSet.py --scenario frag --output_file res/res_frag_last_v2.csv
    # CUDA_VISIBLE_DEVICES=5 python sample_for_pocket_testSet.py --scenario link --output_file res/res_link_last_v2.csv
    # CUDA_VISIBLE_DEVICES=6 python sample_for_pocket_testSet.py --scenario scaffold --output_file res/res_scaffold_last_v2.csv
    # CUDA_VISIBLE_DEVICES=3 python sample_for_pocket_testSet.py --scenario denovo --output_file res/res_denovo_last_v2.csv
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='denovo', choices=['frag', 'link', 'scaffold', 'denovo'])
    parser.add_argument('--output_file', type=str, default='res.csv')

    args = parser.parse_args()

    if '/data4/wenkai/anaconda3/envs/molcraft/bin:' not in os.environ['PATH']:
        os.environ['PATH'] = '/data4/wenkai/anaconda3/envs/molcraft/bin:' + os.environ['PATH']

    scenario = args.scenario  # frag, link, scaffold, denovo

    path = '/data4/wenkai/MolCRAFT_CLF/data/test_set'
    file_names = os.listdir(path)
    qeds, sas, lipinskis, logps, vina_scores, vina_mins, vina_docks = [], [], [], [], [], [], []
    for file_idx, file_name in enumerate(file_names):
        # file_name = 'GLMU_STRPN_2_459_0'
        names = os.listdir(os.path.join(path, file_name))
        pdb_name = [x for x in names if 'pdb' in x and 'pdbqt' not in x][0]
        sdf_name = [x for x in names if 'sdf' in x and 'pdbqt' not in x][0]
        protein_path = os.path.join(path, file_name, pdb_name)
        ligand_path = os.path.join(path, file_name, sdf_name)
        emb_info = torch.load(os.path.join(path, file_name, 'emb_info.pt'))

        # guide_index = [[0,1,2,3,4,5,6,7,8]]
        try:
            mol = Chem.MolFromMolFile(ligand_path, sanitize=False)
            mol = Chem.RemoveHs(mol)
            num_nodes = mol.GetNumAtoms()
            mask = Mask(mol)
            if scenario == 'frag':
                guide_index = mask.get_frag_mask()
            elif scenario == 'link':
                guide_index = mask.get_link_mask() + mask.get_single_link_mask()
            elif scenario == 'scaffold':
                guide_index = mask.get_scaffold_side_chain_mask()
            elif scenario == 'denovo':
                guide_index = [[k for k in range(num_nodes)]]
        except:
            print(f"process {file_name} fail")
            continue
        if not guide_index:
            print(f"index is none")
            continue
        call(protein_path, ligand_path, emb_info, guide_index)

        files = os.listdir('output')
        n = len(files)
        if n == 0:
            continue
        qed, sa, lipinski, logp, vina_score, vina_min, vina_dock = [], [], [], [], [], [], []
        for file in files:
            out_fn = f'output/{file}'
            metrics = Metrics(protein_path, ligand_path, out_fn).evaluate()
            qed.append(round(metrics['qed'], 3))
            sa.append(round(metrics['sa'], 3))
            lipinski.append(metrics['lipinski'])
            # logp.append(round(metrics['logp'], 3))
            vina_score.append(round(metrics['vina_score'], 3))
            vina_min.append(round(metrics['vina_minimize'], 3))
            vina_dock.append(round(metrics['vina_dock'], 3))
            # clash.append(metrics['clash'])
        print(f"{file_idx}. Avg metrics:\nQED: {sum(qed)/n}\nSA: {sum(sa)/n}\nVina Score: {sum(vina_score)/n}\nVina Min: {sum(vina_min)/n}\nVina Dock: {sum(vina_dock)/n}\nLipinski: {sum(lipinski)/n}\nLogp: {sum(logp)/n}")
        qeds.append(sum(qed)/n)
        sas.append(sum(sa)/n)
        vina_scores.append(sum(vina_score)/n)
        vina_mins.append(sum(vina_min)/n)
        vina_docks.append(sum(vina_dock)/n)
        lipinskis.append(sum(lipinski)/n)
    res = pd.DataFrame({'qed': qeds, 'sa': sas, 'vina_score': vina_scores, 'vina_min':vina_mins, 'vina_dock':vina_docks, 'lipinski':lipinskis})
    # res.to_csv('res/gradient_vanilla_res.csv', index=False)
    # res.to_csv('res/gradient_endback_res.csv', index=False)
    # res.to_csv('res/base_res.csv', index=False)
    # res.to_csv('res/noise_diff3_res.csv', index=False)
    res.to_csv(args.output_file, index=False)
    # print(json.dumps(metrics, indent=4, cls=NpEncoder))

