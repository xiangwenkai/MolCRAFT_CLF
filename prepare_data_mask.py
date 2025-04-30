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
from rdkit.Chem import rdMMPA
from scipy.spatial import distance_matrix
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdRGroupDecomposition
# from openbabel import pybel
import warnings
warnings.filterwarnings("ignore")
random.seed(42)


def get_exits(mol):
    """
    Returns atoms marked as exits in DeLinker data
    """
    exits = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == '*':
            exits.append(atom)
    return exits

def remove_dummys_mol(molecule):
    '''
    Input: mol / str containing dummy atom
    Return: Removed mol, anchor_idx
    '''
    if type(molecule) == str:
        dum_mol = Chem.MolFromSmiles(molecule)
    else:
        dum_mol = molecule
    Chem.SanitizeMol(dum_mol)
    exits = get_exits(dum_mol)
    exit = exits[0]
    bonds = exit.GetBonds()
    if len(bonds) > 1:
        raise Exception('Exit atom has more than 1 bond')
    bond = bonds[0]
    exit_idx = exit.GetIdx()
    source_idx = bond.GetBeginAtomIdx()
    target_idx = bond.GetEndAtomIdx()
    anchor_idx = source_idx if target_idx == exit_idx else target_idx
    efragment = Chem.EditableMol(dum_mol)
    efragment.RemoveBond(source_idx, target_idx)
    efragment.RemoveAtom(exit_idx)

    return efragment.GetMol(), anchor_idx

def check_linker(fragmentation, verbose=False, linker_min=2,min_path_length=2,fragment_min=2):
    linker, frags = fragmentation
    if type(linker) == str:
        linker = Chem.MolFromSmiles(linker)
        frags = Chem.MolFromSmiles(frags)

    frag1, frag2 = Chem.GetMolFrags(frags, asMols=True)
    if min(frag1.GetNumHeavyAtoms(), frag2.GetNumHeavyAtoms()) < fragment_min:
        if verbose:
            print('These Fragments are too small')
        return False
    if linker.GetNumHeavyAtoms()< linker_min:
        if verbose:
            print('This linker are too small')
        return False
    dummy_atom_idxs = [atom.GetIdx() for atom in linker.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atom_idxs) != 2:
        if verbose:
            print('This linker is not the middle linker')
        return False
    path_length = len(Chem.rdmolops.GetShortestPath(linker, dummy_atom_idxs[0], dummy_atom_idxs[1]))-2
    if path_length < min_path_length:
        if verbose:
            print('This linker is too short')
        return False
    return True

def check_linkers(fragmentations,verbose=False):
    filter_fragmentations = []
    for fragmentation in fragmentations:
        if check_linker(fragmentation,verbose=verbose):
            filter_fragmentations.append(fragmentation)
    return filter_fragmentations

def get_mark(mol):
    '''
    The R Group Mark Finder
    '''
    marks = []
    for atom in mol.GetAtoms():
        atomicnum = atom.GetAtomicNum()
        if atomicnum == 0:
            marks.append(atom)
    return marks

def remove_mark_mol(molecule):
    '''
    Input: mol / str containing dummy atom
    Return: Removed mol, anchor_idx
    '''
    if type(molecule) == str:
        dum_mol = Chem.MolFromSmiles(molecule)
    else:
        dum_mol = molecule
    Chem.SanitizeMol(dum_mol)
    marks = get_mark(dum_mol)
    mark = marks[0]
    bonds = mark.GetBonds()
    if len(bonds) > 1:
        raise Exception('Exit atom has more than 1 bond')
    bond = bonds[0]
    mark_idx = mark.GetIdx()
    source_idx = bond.GetBeginAtomIdx()
    target_idx = bond.GetEndAtomIdx()
    anchor_idx = source_idx if target_idx == mark_idx else target_idx
    efragment = Chem.EditableMol(dum_mol)
    efragment.RemoveBond(source_idx, target_idx)
    efragment.RemoveAtom(mark_idx)

    return efragment.GetMol(), anchor_idx

def Murcko_decompose(mol, visualize=False):
    # The rdRGroupDecomposition may cause the kernel dumped in jupyter under the virtual environment
    # I don't know why but it works well in another virtual environment with the same version of rdkit
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    decompose = rdRGroupDecomposition.RGroupDecompose([scaffold], [mol])
    side_chains = []
    decompose = list(decompose[0][0].values())
    for i, rgroup in enumerate(decompose):
        if i >0:
            if visualize:
                side_chains.append(rgroup)
            else:
                rgroup, id = remove_mark_mol(rgroup)
                side_chains.append(rgroup)

    if visualize:
        scaffold = decompose[0]

    return scaffold, side_chains

class Mask():
    def __init__(self, mol, pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]"):
        super().__init__()

        self.pattern = pattern
        Chem.SanitizeMol(mol)
        self.mol = mol
        self.num_atoms = mol.GetNumAtoms()

    def get_frag_mask(self):
        fragmentations = rdMMPA.FragmentMol(self.mol, minCuts=1, maxCuts=1, maxCutBonds=100, pattern=self.pattern, resultsAsMols=False)
        fragmentation = random.choice(fragmentations)[1].replace('.', ',').split(',')  # no core
        id = random.randint(0, 1)
        masked_frag = remove_dummys_mol(fragmentation[id])[0]

        if masked_frag is not None:
            masked_id = self.mol.GetSubstructMatch(masked_frag)
            context_id = list(set(list(range(self.num_atoms))) - set(masked_id))
            context_idx = list(masked_id)
            masked_idx = list(context_id)
            return [masked_idx, context_idx]
        else:
            return []

    def get_link_mask(self):
        try:
            fragmentations = rdMMPA.FragmentMol(self.mol, minCuts=2, maxCuts=2, maxCutBonds=100, pattern=self.pattern, resultsAsMols=False)
            fragmentations = check_linkers(fragmentations)
            fragmentation = random.choice(fragmentations)
            core, chains = fragmentation
            masked_frag = remove_dummys_mol(core)[0]
            masked_frag = remove_dummys_mol(masked_frag)[0]
            masked_id = mol.GetSubstructMatch(masked_frag)
            context_id = list(set(list(range(self.num_atoms))) - set(masked_id))
            masked_idx = list(context_id)
            return masked_idx
        except:
            return []

    def get_single_link_mask(self):
        try:
            fragmentations = rdMMPA.FragmentMol(self.mol, minCuts=2, maxCuts=2, maxCutBonds=100, pattern=self.pattern, resultsAsMols=False)
            fragmentations = check_linkers(fragmentations)
            fragmentation = random.choice(fragmentations)
            core, chains = fragmentation
            frag = chains.split('.')
            id = random.randint(0, 1)
            masked_frag = remove_dummys_mol(frag[id])[0]

            masked_id = mol.GetSubstructMatch(masked_frag)
            context_id = list(set(list(range(self.num_atoms))) - set(masked_id))
            masked_idx = list(context_id)
            return masked_idx
        except:
            return []
    def get_scaffold_side_chain_mask(self):
        try:
            scaffold, side_chains = Murcko_decompose(self.mol)
            if len(side_chains) == 0:
                raise ValueError('Side Chains decomposition is None')
            masked_frag = scaffold

            masked_id = mol.GetSubstructMatch(masked_frag)
            context_id = list(set(list(range(self.num_atoms))) - set(masked_id))
            masked_idx = list(context_id)
            return [context_id, masked_idx]
        except:
            return []

    def get_all_mask(self):
        res = []
        try:
            frag_mask = self.get_frag_mask()
        except:
            frag_mask = []
        try:
            link_mask = self.get_link_mask()
        except:
            link_mask = []
        try:
            single_link_mask = self.get_single_link_mask()
        except:
            single_link_mask = []
        try:
            scaffold_mask = self.get_scaffold_side_chain_mask()
        except:
            scaffold_mask = []
        if frag_mask:
            res.extend(frag_mask)
        if link_mask:
            res.append(link_mask)
        if single_link_mask:
            res.append(single_link_mask)
        if scaffold_mask:
            res.extend(scaffold_mask)
        if res:
            res = [f'{sorted(x)}' for x in res]
            res = list(set(res))
            res = [eval(x) for x in res]
        return res


if __name__ == "__main__":
    db = lmdb.open('/data4/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb', map_size=10 * (1024 * 1024 * 1024),
                   create=False,subdir=False,readonly=True,lock=False,readahead=False,meminit=False)
    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))

    db_new = lmdb.open('/data4/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final_mask.lmdb',map_size=50 * (1024 * 1024 * 1024),  # 80GB
                       create=False,subdir=False,readonly=False,lock=False,readahead=False,meminit=False)
    txn_new = db_new.begin(write=True, buffers=True)

    # idx=1
    # key = keys[idx]
    # data = pickle.loads(db.begin().get(key))
    crossdock_path = '/data4/wenkai/MolCRAFT_CLF/data/crossdocked_v1.1_rmsd1.0_pocket10'

    m = 0
    idxes, all_atoms, all_coordinates, fail_idxes = [], [], [], []
    for idx in range(len(keys)):
        key = keys[idx]
        data = pickle.loads(db.begin().get(key))
        protein_filename = data['protein_filename']
        ligand_filename = data['ligand_filename']
        protein_path = os.path.join(crossdock_path, protein_filename)
        ligand_path = os.path.join(crossdock_path, ligand_filename)

        try:
            mol = Chem.MolFromMolFile(ligand_path, sanitize=False)
            # Chem.SanitizeMol(mol)
            mol.UpdatePropertyCache(strict=False)
            mask = Mask(mol)
            mask_idxes = mask.get_all_mask()

            # mol = Chem.RemoveHs(mol)
            # smi = Chem.MolToSmiles(mol)
            # name = smi
            # num_nodes = mol.GetNumAtoms()
        except:
            print(f"{idx} failed")
            mask_idxes = []
            fail_idxes.append(idx)
        data['mask_indexes'] = mask_idxes
        txn_new.put(
            key=str(idx).encode(),
            value=pickle.dumps(data)
        )
        idxes.append(idx)
        m += 1
    txn_new.commit()
    print(f"samples:{m}")

