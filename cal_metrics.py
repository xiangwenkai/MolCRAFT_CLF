import os
import random
import pandas as pd
import torch
from core.evaluation.utils import scoring_func
from core.evaluation.docking_vina import VinaDockingTask
from posecheck import PoseCheck
import numpy as np
from rdkit import Chem
import argparse
random.seed(42)


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='denovo',
                        choices=['frag', 'link', 'scaffold', 'denovo', 'nomask'])
    args = parser.parse_args()
    path = "/data4/wenkai/MolCRAFT_CLF/data/test_set"
    files = os.listdir(path)
    scene = args.scenario
    logs_dir = f"logs/test_{scene}"
    os.makedirs(logs_dir, exist_ok=True)
    qeds, sas, lipinskis, logps, vina_scores, vina_mins, vina_docks = [], [], [], [], [], [], []
    for f in files:
        names = os.listdir(f'{path}/{f}')
        pdb_name = [x for x in names if 'pdb' in x and 'pdbqt' not in x][0]
        sdf_name = [x for x in names if 'sdf' in x and 'pdbqt' not in x][0]
        protein_path = f'{path}/{f}/{pdb_name}'
        ligand_path = f'{path}/{f}/{sdf_name}'
        if os.path.exists(f'/data4/wenkai/MolCRAFT_CLF/test_{scene}/{f}'):
            pred_sdfs = os.listdir(f'/data4/wenkai/MolCRAFT_CLF/test_{scene}/{f}')
            n = len(pred_sdfs)
            qed, sa, lipinski, logp, vina_score, vina_min, vina_dock = [], [], [], [], [], [], []
            n_vina, n_vina_min = n, n
            for i, pred_sdf in enumerate(pred_sdfs):
                out_fn = f'/data4/wenkai/MolCRAFT_CLF/test_{scene}/{f}/{pred_sdf}'
                metrics = Metrics(protein_path, ligand_path, out_fn).evaluate()
                qed.append(round(metrics['qed'], 3))
                sa.append(round(metrics['sa'], 3))
                lipinski.append(metrics['lipinski'])
                # logp.append(round(metrics['logp'], 3))
                if metrics['vina_score'] < 0:
                    vina_score.append(round(metrics['vina_score'], 3))
                else:
                    n_vina -= 1
                if metrics['vina_minimize'] < 0:
                    vina_min.append(round(metrics['vina_minimize'], 3))
                else:
                    n_vina_min -= 1
                vina_dock.append(round(metrics['vina_dock'], 3))
                if metrics['vina_score'] > 0 or metrics['vina_minimize'] > 0:
                    with open(f"{logs_dir}/log.txt", 'a') as log:
                        log.write(f"file: {f}   pred file: {pred_sdf}  vina score: {round(metrics['vina_score'], 3)}    vina min: {round(metrics['vina_minimize'], 3)}\n")
                    os.makedirs(f"logs/test_{scene}/{f}", exist_ok=True)
                    os.system(f"cp /data4/wenkai/MolCRAFT_CLF/test_{scene}/{f}/{pred_sdf} logs/test_{scene}/{f}/{pred_sdf}")
                    os.system(
                        f"cp /data4/wenkai/MolCRAFT_CLF/data/test_set/{f}/*.sdf logs/test_{scene}/{f}/ligand.sdf")
                    os.system(
                        f"cp /data4/wenkai/MolCRAFT_CLF/data/test_set/{f}/*.pdb logs/test_{scene}/{f}/protein.pdb")
            qeds.append(sum(qed) / n)
            sas.append(sum(sa) / n)
            if n_vina > 0:
                vina_scores.append(sum(vina_score) / n_vina)
            else:
                vina_scores.append(0.)
            if n_vina_min > 0:
                vina_mins.append(sum(vina_min) / n_vina_min)
            else:
                vina_mins.append(0.)
            vina_docks.append(sum(vina_dock) / n)
            lipinskis.append(sum(lipinski) / n)
    res = pd.DataFrame(
        {'qed': qeds, 'sa': sas, 'vina_score': vina_scores, 'vina_min': vina_mins, 'vina_dock': vina_docks,
         'lipinski': lipinskis})
    res.to_csv(f'/data4/wenkai/MolCRAFT_CLF/test_{scene}/res_{scene}.csv', index=False)