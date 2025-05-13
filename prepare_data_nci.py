import os
if '/data4/wenkai/anaconda3/envs/molcraft/bin:' not in os.environ['PATH']:
    os.environ['PATH'] = '/data4/wenkai/anaconda3/envs/molcraft/bin:' + os.environ['PATH']
if '/data/wenkai/anaconda3/envs/molcraft/bin:' not in os.environ['PATH']:
    os.environ['PATH'] = '/data/wenkai/anaconda3/envs/molcraft/bin:' + os.environ['PATH']
import lmdb
import pickle
import oddt
import argparse
import json
import time
import numpy as np
import pandas as pd
import re
import os
import random
import torch
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.BRICS import FindBRICSBonds
from tqdm import tqdm
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
# from src import const
from functools import partial
# from openbabel import pybel
from oddt.interactions import hbonds, hbond_acceptor_donor, halogenbonds, pi_stacking, salt_bridges, hydrophobic_contacts, pi_cation
from oddt.toolkits.common import canonize_ring_path
import warnings
warnings.filterwarnings("ignore")
random.seed(42)


# #################################################################################### #
# ####################################### BRICS ###################################### #
# #################################################################################### #
REGEX = re.compile('\[\d*\*\]')


def split_into_n_fragments(mol, bonds, num_frags):
    num_bonds = num_frags - 1
    bondidx2minfrag = {}
    bondidx2atoms = {}
    for bond in bonds:
        bond_idx = mol.GetBondBetweenAtoms(bond[0], bond[1]).GetIdx()
        frags = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=False)
        for atom in frags.GetAtoms():
            atom.SetIsAromatic(False)
        frags = Chem.GetMolFrags(frags, asMols=True)
        minfragsize = min([f.GetNumAtoms() for f in frags])
        bondidx2minfrag[bond_idx] = minfragsize
        bondidx2atoms[bond_idx] = bond

    # Selecting only top-N bonds connecting the biggest 6 fragments
    sorted_bonds = sorted(bondidx2minfrag.keys(), key=lambda bidx: -bondidx2minfrag[bidx])
    bonds_to_split = sorted_bonds[:num_bonds]

    # Selecting atoms connected by top-N bonds
    # Note that we add 1 to start numeration from 1 to correctly assign labels (RDKit issue)
    bond_atoms = [
        (bondidx2atoms[bidx][0] + 1, bondidx2atoms[bidx][1] + 1)
        for bidx in bonds_to_split
    ]

    frags = Chem.FragmentOnBonds(mol, bonds_to_split, addDummies=True, dummyLabels=bond_atoms)
    for atom in frags.GetAtoms():
        atom.SetIsAromatic(False)
    frags = Chem.GetMolFrags(frags, asMols=True)
    return frags, bond_atoms


def check_fragments_brics(frags, min_frag_size):
    for frag in frags:
        num_dummy_atoms = len(re.findall(REGEX, Chem.MolToSmiles(frag)))
        if (frag.GetNumAtoms() - num_dummy_atoms) < min_frag_size:
            return False

    return True


def generate_possible_connected_linkers(neighbors):
    candidates = np.where(neighbors.sum(0) > 2)[0]
    possible_linkers = set([
        (candidate,)
        for candidate in candidates
    ])
    return possible_linkers


def generate_possible_2nd_order_linkers(neighbors):
    # Removing edge fragments
    initial_candidates = np.where(neighbors.sum(0) > 1)[0]
    neighbors = neighbors[initial_candidates][:, initial_candidates]

    # Computing 2nd order neighbors and finding all loops
    second_order_neigh = ((neighbors @ neighbors) > 0).astype(int) * (1 - neighbors) - np.eye(neighbors.shape[0])
    candidates = set(np.where(np.diag(second_order_neigh @ second_order_neigh))[0])

    possible_linkers_pairs = set()
    for first_candidate in candidates:
        for second_candidate in set(np.where(second_order_neigh[first_candidate])[0]) & candidates:
            linker_1 = initial_candidates[first_candidate]
            linker_2 = initial_candidates[second_candidate]
            if linker_1 != linker_2:
                possible_linkers_pairs.add(tuple(sorted([linker_1, linker_2])))

    return possible_linkers_pairs


def generate_possible_3nd_order_linkers(neighbors):
    # Removing edge fragments
    initial_candidates = np.where(neighbors.sum(0) > 1)[0]
    neighbors = neighbors[initial_candidates][:, initial_candidates]

    # Computing 3rd order neighbors and finding all loops
    third_order_neigh = ((neighbors @ neighbors @ neighbors) > 0).astype(int)
    third_order_neigh = third_order_neigh * (1 - neighbors) - np.eye(neighbors.shape[0])
    candidates = set(np.where(np.diag(third_order_neigh @ third_order_neigh @ third_order_neigh))[0])

    possible_linkers_triples = set()
    for first_candidate in candidates:
        rest_candidates = candidates.difference({first_candidate})
        rest_candidates = set(np.where(third_order_neigh[first_candidate])[0]) & rest_candidates
        for second_candidate in rest_candidates:
            remainders = rest_candidates.difference({second_candidate})
            for third_candidate in remainders:
                linker_1 = initial_candidates[first_candidate]
                linker_2 = initial_candidates[second_candidate]
                linker_3 = initial_candidates[third_candidate]
                if linker_1 != linker_2 and linker_1 != linker_3 and linker_2 != linker_3:
                    possible_linkers_triples.add(tuple(sorted([linker_1, linker_2, linker_3])))

    return possible_linkers_triples


def fragment_by_brics(smiles, min_frag_size, max_frag_size, num_frags, linker_type=None):
    mol = Chem.MolFromSmiles(smiles)

    # Finding all BRICS bonds
    bonds = [bond[0] for bond in FindBRICSBonds(mol)]
    if len(bonds) == 0:
        return []

    # Splitting molecule into fragments
    frags, bond_atoms = split_into_n_fragments(mol, bonds, num_frags)
    if not check_fragments_brics(frags, min_frag_size):
        return []

    # Building mapping between fragments and connecting atoms
    atom2frag = {}
    for i, frag in enumerate(frags):
        matches = re.findall(REGEX, Chem.MolToSmiles(frag))
        for match in matches:
            atom = int(match[1:-2])
            atom2frag[atom] = i

    # Creating adjacency matrix
    neighbors = np.zeros((len(frags), len(frags)))
    for atom1, atom2 in bond_atoms:
        neighbors[atom2frag[atom1], atom2frag[atom2]] = 1
        neighbors[atom2frag[atom2], atom2frag[atom1]] = 1

    # Generating possible linkers
    if linker_type is None:
        possible_linkers = []
        possible_linkers += generate_possible_connected_linkers(neighbors)
        possible_linkers += generate_possible_2nd_order_linkers(neighbors)
        possible_linkers += generate_possible_3nd_order_linkers(neighbors)
    elif linker_type == 1:
        possible_linkers = generate_possible_connected_linkers(neighbors)
    elif linker_type == 2:
        possible_linkers = generate_possible_2nd_order_linkers(neighbors)
    elif linker_type == 3:
        possible_linkers = generate_possible_3nd_order_linkers(neighbors)
    else:
        raise NotImplementedError

    # Formatting the results
    results = []
    for linkers in possible_linkers:
        linkers_smi = ''
        fragments_smi = ''
        for i in range(len(frags)):
            if i in linkers:
                linkers_smi += Chem.MolToSmiles(frags[i]) + '.'
            else:
                fragments_smi += Chem.MolToSmiles(frags[i]) + '.'

        linkers_smi = linkers_smi[:-1]
        fragments_smi = fragments_smi[:-1]
        results.append([smiles, linkers_smi, fragments_smi, 'brics'])

    return results


# #################################################################################### #
# ####################################### BIND SITE ################################## #
# #################################################################################### #
def get_break_bonds(molecule, atom_indices):
    """
    打断选定原子和其它原子的链接，形成片段
    :param molecule: RDKit 分子对象
    :param atom_indices: 要截断的原子的索引列表
    :return: bonds: 要打断的键列表
    """
    # 记录需要打断的键 (由所选原子涉及的所有键)
    bonds_to_break = []
    for idx in atom_indices:
        atom = molecule.GetAtomWithIdx(idx)
        bonds = atom.GetBonds()
        for bond in bonds:
            bonds_to_break.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return bonds_to_break


# sort bind site atom
def sort_bind_atom(mol, h_bond_donors, h_bond_acceptors, hydrophobic_groups, positive_sites, negative_sites, coordination_sites):
    all_idx = list(set(h_bond_donors + h_bond_acceptors + hydrophobic_groups + positive_sites +
                       negative_sites + coordination_sites))
    score = {i: 0 for i in all_idx}
    # 不同作用的得分不同
    for idx in all_idx:
        if idx in h_bond_donors or idx in h_bond_acceptors:
            score[idx] += 5
        if idx in hydrophobic_groups:
            score[idx] += 3
        if idx in positive_sites or idx in negative_sites:
            score[idx] += 2
        if idx in coordination_sites:
            score[idx] += 1
    # 根据原子的连边数量，赋予得分。一般在外围的原子连边少，得分高
    # 获取该原子连接的原子数目
    for idx in all_idx:
        atom = mol.GetAtomWithIdx(idx)
        degree = atom.GetDegree()
        if degree == 1:
            score[idx] += 5
        elif degree == 2:
            score[idx] += 3
        elif degree == 3:
            score[idx] += 1
    if len(score) <= 10:
        return list(score.keys())
    else:
        score = dict(sorted(score.items(), key=lambda item: item[1], reverse=True))
        return list(score.keys())[:10]


def get_pocket(pdb_path):
    struct = PDBParser().get_structure('', pdb_path)
    # Remove hydrogen atoms from the structure
    for model in struct:
        for chain in model:
            residues_to_remove = []
            for residue in chain:
                atoms_to_remove = [atom for atom in residue if atom.element == "H"]
                for atom in atoms_to_remove:
                    residue.detach_child(atom.get_id())
                if not residue:
                    residues_to_remove.append(residue)
            for residue in residues_to_remove:
                chain.detach_child(residue)

    pocket_coords_full = []
    pocket_types_full = []

    pocket_coords_bb = []
    pocket_types_bb = []

    for residue in struct.get_residues():
        for atom in residue.get_atoms():
            atom_name = atom.get_name()
            atom_type = atom.element.upper()
            # if atom_type == 'H':
            #     continue
            atom_coord = atom.get_coord()

            pocket_coords_full.append(atom_coord.tolist())
            pocket_types_full.append(atom_type)

            if atom_name in {'N', 'CA', 'C', 'O'}:
                pocket_coords_bb.append(atom_coord.tolist())
                pocket_types_bb.append(atom_type)

    return {
        'full_coord': pocket_coords_full,
        'full_types': pocket_types_full,
        'bb_coord': pocket_coords_bb,
        'bb_types': pocket_types_bb,
    }


def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot


def get_interactions(pdb_file, sdf_file):
    '''
        return res_id, res_atom_name, ligand_atom_id
    '''
    nci_maps = {'halogenbonds': 4, 'salt_bridges': 5, 'hydrophobic_contacts': 6, 'pi_stacking': 7,
                'pi_cation': 8}

    interactions = {
        'hbond_acceptor_donor': partial(hbond_acceptor_donor, cutoff=3.0, tolerance=30),
        'halogenbonds': partial(halogenbonds, cutoff=3.5, tolerance=30),
        # 'pi_stacking': partial(pi_stacking, cutoff=5.5, tolerance=90),
        'salt_bridges': partial(salt_bridges, cutoff=5.0),
        # 'hydrophobic_contacts': hydrophobic_contacts,
        # 'pi_cation': partial(pi_cation, cutoff=6.6, tolerance=60),
    }

    result = {}
    protein = next(oddt.toolkit.readfile('pdb', pdb_file))
    protein.protein = True
    protein.removeh()
    ligand = next(oddt.toolkit.readfile('sdf', sdf_file))
    ligand.removeh()
    for k, v in interactions.items():
        inter = []
        inter_res = v(protein, ligand)
        protein_atoms, ligand_atoms = inter_res[:2]
        if k in ['hbond_acceptor_donor', 'halogenbonds', 'pi_cation']:
            strict = inter_res[-1]
        elif k in ['pi_stacking']:
            strict = [(inter_res[-2][t] or inter_res[-1][t]) for t in range(protein_atoms.shape[0])]
        elif k in ['salt_bridges','hydrophobic_contacts']:
            strict = [True for _ in range(protein_atoms.shape[0])]
        else:
            raise ValueError
        # print(k)

        for idx in range(len(protein_atoms)):
            # print(strict[idx])
            if (strict[idx] == True):
                if k == 'hbond_acceptor_donor':
                    tmp = {
                        'res_id': protein_atoms[idx][9],  # seq index
                        'res_num': protein_atoms[idx][10],
                        'res_atom_type': protein_atoms[idx][5].split('.')[0],
                        'res_atom_id': protein_atoms[idx][0],  # 0-index
                        'res_atom_isacceptor': protein_atoms[idx][13],
                        'res_atom_isdonor': protein_atoms[idx][14],
                        'lig_atom_id': ligand_atoms[idx][0],
                        'lig_atom_isacceptor': ligand_atoms[idx][13],
                        'lig_atom_isdonor': ligand_atoms[idx][14],
                    }
                elif k in ['halogenbonds', 'salt_bridges', 'hydrophobic_contacts']:
                    tmp = {
                        'res_id': protein_atoms[idx][9],  # seq index
                        'res_num': protein_atoms[idx][10],
                        'res_atom_type': protein_atoms[idx][5].split('.')[0],
                        'res_atom_id': protein_atoms[idx][0],  # 0-index
                        'lig_atom_id': ligand_atoms[idx][0],
                    }
                elif k in ['pi_stacking']:
                    for ring in protein.sssr:
                        if ring.IsAromatic():
                            path = [x - 1 for x in ring._path]  #
                            atoms = protein.atom_dict[canonize_ring_path(path)]
                            if len(atoms):
                                atom = atoms[0]
                                coords = atoms['coords']
                                centroid = coords.mean(axis=0)
                                if np.linalg.norm(centroid-protein_atoms[idx][0]) < 1e-3:
                                    res_atom_type=[t[5].split('.')[0] for t in atoms]
                                    res_atom_id=[t[0] for t in atoms]
                                    break
                    for ring in ligand.sssr:
                        if ring.IsAromatic():
                            path = [x - 1 for x in ring._path]  #
                            atoms = ligand.atom_dict[canonize_ring_path(path)]
                            if len(atoms):
                                atom = atoms[0]
                                coords = atoms['coords']
                                centroid = coords.mean(axis=0)
                                if np.linalg.norm(centroid-ligand_atoms[idx][0]) < 1e-3:
                                    lig_atom_id=[t[0] for t in atoms]
                                    break
                    tmp = {
                        'res_id': protein_atoms[idx][2],
                        'res_num': protein_atoms[idx][3],
                        'res_atom_type': res_atom_type,
                        'res_atom_id': res_atom_id,
                        'lig_atom_id': lig_atom_id,
                    }
                elif k in ['pi_cation']:
                    for ring in protein.sssr:
                        if ring.IsAromatic():
                            path = [x - 1 for x in ring._path]  #
                            atoms = protein.atom_dict[canonize_ring_path(path)]
                            if len(atoms):
                                atom = atoms[0]
                                coords = atoms['coords']
                                centroid = coords.mean(axis=0)
                                if np.linalg.norm(centroid-protein_atoms[idx][0]) < 1e-3:
                                    res_atom_type=[t[5].split('.')[0] for t in atoms]
                                    res_atom_id=[t[0] for t in atoms]
                                    break
                    tmp = {
                        'res_id': protein_atoms[idx][2],
                        'res_num': protein_atoms[idx][3],
                        'res_atom_type': res_atom_type,
                        'res_atom_id': res_atom_id,
                        'lig_atom_id': ligand_atoms[idx][0],
                    }
                else:
                    raise ValueError
                inter.append(tmp)
        result[k] = inter

    nci_type, protein_atom_id, protein_atom_type, lig_atom_id = [], [], [], []
    for k, v in result.items():
        if k == 'hbond_acceptor_donor':
            for x in v:
                if x['res_atom_isacceptor'] == False and x['res_atom_isdonor'] == True:
                    nci_type.append(1)
                elif x['res_atom_isacceptor'] == True and x['res_atom_isdonor'] == False:
                    nci_type.append(2)
                elif x['res_atom_isacceptor'] == True and x['res_atom_isdonor'] == True:
                    nci_type.append(3)
                protein_atom_type.append(x['res_atom_type'])
                protein_atom_id.append(x['res_atom_id'])
                lig_atom_id.append(x['lig_atom_id'])
        else:
            for x in v:
                if type(x['lig_atom_id']) is not list and x['lig_atom_id'] not in lig_atom_id:
                    nci_type.append(nci_maps[k])
                    protein_atom_type.append(x['res_atom_type'])
                    protein_atom_id.append(x['res_atom_id'])
                    lig_atom_id.append(x['lig_atom_id'])
    return {'nci_type': nci_type, 'protein_atom_type': protein_atom_type,
            'protein_atom_id': protein_atom_id, 'lig_atom_id': lig_atom_id}


db = lmdb.open('/data/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final_mask.lmdb', map_size=20 * (1024 * 1024 * 1024),
               create=False,subdir=False,readonly=True,lock=False,readahead=False,meminit=False)
with db.begin() as txn:
    keys = list(txn.cursor().iternext(values=False))

db_new = lmdb.open('/data/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10_processed_nci.lmdb',map_size=20 * (1024 * 1024 * 1024),
                   create=False,subdir=False,readonly=False,lock=False,readahead=False,meminit=False)
txn_new = db_new.begin(write=True, buffers=True)


# idx=1
# key = keys[idx]
# data = pickle.loads(db.begin().get(key))

crossdock_path = '/data/wenkai/MolCRAFT_CLF/data/crossdocked_pocket10'

m = 0
for key in tqdm(keys):
    data = pickle.loads(db.begin().get(key))
    protein_filename = data['protein_filename']
    ligand_filename = data['ligand_filename']
    protein_path = os.path.join(crossdock_path, protein_filename)
    ligand_path = os.path.join(crossdock_path, ligand_filename)

    try:
        mol = Chem.MolFromMolFile(ligand_path, sanitize=False)
        # Chem.SanitizeMol(mol)
        # mol.UpdatePropertyCache(strict=False)
        mol = Chem.RemoveHs(mol)
        smi = Chem.MolToSmiles(mol)
        name = smi
        num_nodes = mol.GetNumAtoms()
    except:
        data['nci_indexes'] = []
        txn_new.put(
            key=key,
            value=pickle.dumps(data)
        )
        continue
    # nci infromation
    nci_info = get_interactions(protein_path, ligand_path)
    candidate_nci = len(nci_info['nci_type'])

    if candidate_nci <= 0:
        data['nci_indexes'] = []
    else:
        data['nci_indexes'] = sorted(list(set(nci_info['lig_atom_id'])))
    txn_new.put(
        key=key,
        value=pickle.dumps(data)
    )
    m += 1
txn_new.commit()
print(f"samples:{m}")
