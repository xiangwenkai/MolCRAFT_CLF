import os
import pickle
import random
import torch
from rdkit import Chem

# from openbabel import pybel
import warnings
warnings.filterwarnings("ignore")
random.seed(42)

# crossdock_path = '/data/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10'
#
# df_split = torch.load("/data/wenkai/MolCRAFT_CLF/data/crossdocked_pocket10_pose_split.pt")
# len(df_split['test'])
#
# with open('/data/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10/index.pkl', 'rb') as f:
#     df_index = pickle.load(f)
# df_index[0]


pocket_count = {}
af3_pair_dir = f"/data/wenkai/MolCRAFT_CLF/data/chembl_pocket10"
pairs = os.listdir(af3_pair_dir)
index_vals = []
for pair in pairs:
    if pair == 'index.pkl':
        continue
    protein_name = pair.split('-')[0].split('_')[0]
    if protein_name not in pocket_count:
        pocket_count[protein_name] = 1
    else:
        pocket_count[protein_name] += 1
    sub_files = os.listdir(f"{af3_pair_dir}/{pair}")
    pdb_val, sdf_val = None, None
    for sub_file in sub_files:
        if '.pdb' in sub_file:
            pdb_val = f"{pair}/{sub_file}"
        if '.sdf' in sub_file:
            sdf_val = f"{pair}/{sub_file}"
    if pdb_val is not None and sdf_val is not None:
        index_vals.append((pdb_val, sdf_val, f"{pair}/{protein_name}.pdb"))
with open('/data/wenkai/MolCRAFT_CLF/data/chembl_pocket10/index.pkl', 'wb') as f:
    pickle.dump(index_vals, f)
# os.system("cp /data/wenkai/MolCRAFT_CLF/data/chembl_pocket10/index.pkl /data/wenkai/Delete/data/chembl_pocket10/index.pkl")

test_pocket = []
k = 0
for pocket in pocket_count:
    if pocket_count[pocket] <= 3:
        if k + pocket_count[pocket] <= 100:
            test_pocket.append(pocket)
            k += pocket_count[pocket]
        if k >= 100:
            break

train_index, test_index = [], []
try:
    pairs.remove('index.pkl')
except:
    pass
for i, pair in enumerate(pairs):
    protein_name = pair.split('-')[0].split('_')[0]
    if protein_name in test_pocket:
        test_index.append(i)
    else:
        train_index.append(i)
split = {'train': train_index, 'val': [], 'test': test_index}

torch.save(split, "/data/wenkai/MolCRAFT_CLF/data/chembl_pocket10_pose_split.pt")








