import lmdb
import pickle
import argparse
import json
import time
import numpy as np
from unimol_tools import UniMolRepr
import pandas as pd
import re
import os
import pickle
import random
import torch
from rdkit import Chem

# from openbabel import pybel
import warnings
warnings.filterwarnings("ignore")
random.seed(42)

# environment: fgdd
db = lmdb.open('/data4/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final_origin.lmdb', map_size=10 * (1024 * 1024 * 1024),
               create=False,subdir=False,readonly=True,lock=False,readahead=False,meminit=False)
with db.begin() as txn:
    keys = list(txn.cursor().iternext(values=False))

db_new = lmdb.open('/data4/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final_emb.lmdb',map_size=50 * (1024 * 1024 * 1024),  # 80GB
                   create=False,subdir=False,readonly=False,lock=False,readahead=False,meminit=False)
txn_new = db_new.begin(write=True, buffers=True)


# idx=1
# key = keys[idx]
# data = pickle.loads(db.begin().get(key))

crossdock_path = '/data4/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10'
clf = UniMolRepr(data_type='molecule',
                 remove_hs=True,
                 model_name='unimolv1', # avaliable: unimolv1, unimolv2
                 model_size='84m', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                 )
m = 0
idxes, all_atoms, all_coordinates, fail_idxes = [], [], [], []
for key in keys:
    data = pickle.loads(db.begin().get(key))
    protein_filename = data['protein_filename']
    ligand_filename = data['ligand_filename']
    protein_path = os.path.join(crossdock_path, protein_filename)
    ligand_path = os.path.join(crossdock_path, ligand_filename)

    try:
        mol = Chem.MolFromMolFile(ligand_path, sanitize=False)
        try:
            mol = Chem.RemoveHs(mol)
        except:
            pass
        # Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache(strict=False)
        atoms = []
        coordinates = []
        postions = mol.GetConformer().GetPositions()
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            pos = postions[atom.GetIdx()].tolist()
            atoms.append(sym)
            coordinates.append(pos)
        # mol = Chem.RemoveHs(mol)
        # smi = Chem.MolToSmiles(mol)
        # name = smi
        # num_nodes = mol.GetNumAtoms()
    except:
        print(f"{key} failed")
        fail_idxes.append(key)
        continue
    # if smi == 'C[C@H](N)[C@@H](CCCCCC(=O)O)NC(=O)O[Al](F)(F)F':
    #     print(f"{idx} failed")
    #     fail_idxes.append(idx)
    #     continue
    idxes.append(key)
    all_atoms.append(atoms)
    all_coordinates.append(coordinates)
    m += 1
print(f"samples:{m}")


l = len(all_atoms)
batch = 1024
k = int(l / batch)
for i in range(0, k+1):
    atoms_list = all_atoms[i*batch: min((i+1)*batch, l)]
    coord_list = all_coordinates[i*batch: min((i+1)*batch, l)]
    input_dict = {'atoms': atoms_list, 'coordinates': coord_list}
    unimol_repr = clf.get_repr(input_dict, return_atomic_reprs=True)

    for j, key in enumerate(idxes[i*batch: min((i+1)*batch, l)]):
        # CLS token repr
        # print(np.array(unimol_repr['cls_repr']).shape)
        # atomic level repr, align with rdkit mol.GetAtoms()
        # print(np.array(unimol_repr['atomic_reprs']).shape)
        data = pickle.loads(db.begin().get(key))
        # if key in [136380, 136616, 136619, 136619, 141229, 141259, 141265, 141305, 141310, 141315, 141334, 141360, 141365, 141366, 141397, 141398, 141404, 141415, 141424, 141441, 141501, 141553, 141578, 156269, 157696, 157893, 158158, 159261]:
        #     data['lig_emb'] = torch.zeros(unimol_repr['atomic_reprs'][j].shape)
        # else:
        data['lig_emb'] = torch.tensor(unimol_repr['atomic_reprs'][j])
        txn_new.put(
            key=key,
            value=pickle.dumps(data)
        )

txn_new.commit()






# 为crossdock test set生成embedding信息，用于molcraft模型推理测试
clf = UniMolRepr(data_type='molecule',
                 remove_hs=True,
                 model_name='unimolv1', # avaliable: unimolv1, unimolv2
                 model_size='84m', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                 )
path = '/data4/wenkai/MolCRAFT_CLF/data/test_set'
file_names = os.listdir(path)
for file_name in file_names:
    names = os.listdir(os.path.join(path, file_name))
    pdb_name = [x for x in names if 'pdb' in x and 'pdbqt' not in x][0]
    sdf_name = [x for x in names if 'sdf' in x and 'pdbqt' not in x][0]
    protein_path = os.path.join(path, file_name, pdb_name)
    ligand_path = os.path.join(path, file_name, sdf_name)

    try:
        mol = Chem.MolFromMolFile(ligand_path, sanitize=False)
        mol = Chem.RemoveHs(mol)
        # Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache(strict=False)
        atoms = []
        coordinates = []
        postions = mol.GetConformer().GetPositions()
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            pos = postions[atom.GetIdx()].tolist()
            atoms.append(sym)
            coordinates.append(pos)
        input_dict = {'atoms': [atoms], 'coordinates': [coordinates]}
        unimol_repr = clf.get_repr(input_dict, return_atomic_reprs=True)

        emb_info = torch.tensor(unimol_repr['atomic_reprs'][0])
        emb_path = os.path.join(path, file_name, 'emb_info.pt')
        torch.save(emb_info, emb_path)
    except:
        print(f"error {file_name}")


