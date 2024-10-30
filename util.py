from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import Descriptors
import numpy as np

REMOVER = SaltRemover()
ORGANIC_ATOM_SET = set([5, 6, 7, 8, 9, 15, 16, 17, 35, 53])

def remove_norm_layers(model, threshold):
    """Function used to randomly remove normalization layers from a model."""
    for name, module in model.named_children():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                            nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, 
                            nn.GroupNorm)):
            # Convert nn.Norm to nn.Identity
            if random.uniform(0, 1) <= threshold:
                setattr(model, name, nn.Identity())
        else:
            remove_norm_layers(module, threshold)

def preprocess_smiles(sml):
    """Function that preprocesses a SMILES string such that it is in the same format as
    the translation model was trained on. It removes salts and stereochemistry from the
    SMILES sequnce. If the sequnce correspond to an inorganic molecule or cannot be
    interpreted by RDKit nan is returned.

    Args:is
        sml: SMILES sequence.
    Returns:
        preprocessd SMILES sequnces or nan.
    """
    new_sml = remove_salt_stereo(sml, REMOVER)
    new_sml = normalize_smiles(new_sml, canonical=True, isomeric=False)
    return new_sml

def filter_smiles(sml):
    try:
        m = Chem.MolFromSmiles(sml)
        logp = Descriptors.MolLogP(m)
        mol_weight = Descriptors.MolWt(m)
        num_heavy_atoms = Descriptors.HeavyAtomCount(m)
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = set(atom_num_list) <= ORGANIC_ATOM_SET
        if ((logp > -5) & (logp < 7) &
            (mol_weight > 12) & (mol_weight < 600) &
            (num_heavy_atoms > 3) & (num_heavy_atoms < 50) &
            is_organic ):
            return Chem.MolToSmiles(m)
        else:
            return float('nan')
    except:
        return float('nan')
    
def remove_salt_stereo(sml, remover):
    """Function that strips salts and removes stereochemistry information from a SMILES.
    Args:
        sml: SMILES sequence.
        remover: RDKit's SaltRemover object.
    Returns:
        canonical SMILES sequnce without salts and stereochemistry information.
    """
    try:
        sml = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(sml),
                                                dontRemoveEverything=True),
                               isomericSmiles=False)
        if "." in sml:
            sml = keep_largest_fragment(sml)
    except:
        sml = np.float("nan")
    return(sml)

def keep_largest_fragment(sml):
    """Function that returns the SMILES sequence of the largest fragment for a input
    SMILES sequnce.

    Args:
        sml: SMILES sequence.
    Returns:
        canonical SMILES sequnce of the largest fragment.
    """
    mol_frags = Chem.GetMolFrags(Chem.MolFromSmiles(sml), asMols=True)
    largest_mol = None
    largest_mol_size = 0
    for mol in mol_frags:
        size = mol.GetNumAtoms()
        if size > largest_mol_size:
            largest_mol = mol
            largest_mol_size = size
    return Chem.MolToSmiles(largest_mol)

def normalize_smiles(smi, canonical=True, isomeric=False):
    """Function that normalizes a SMILES string."""
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
        return normalized
    except:
        print("Failed to normalize {} !".format(smi))
        return np.nan

def randomize_smiles(smiles, isomericSmiles=False):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        try:
            m = Chem.MolFromSmiles(smiles)
            ans = list(range(m.GetNumAtoms()))
            np.random.shuffle(ans)
            nm = Chem.RenumberAtoms(m,ans)
            return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=isomericSmiles)
        except:
            print("randomize_smiles failed on smiles: ", smiles)
            return smiles

def id2smile(tokens, tokenizer):
    """
    Args:
        tokens (BatchEncoding: Set like): {'input_ids': [], 'token_type_ids': [],'attention_mask': []}
        tokenizer (Tokenizer)
    """
    return tokenizer.batch_decode(tokens['input_ids'], skip_special_tokens=True)

def smile2id(smiles, tokenizer):    
    """
    Args:
        smiles (List like)
        tokenizer (Tokenizer)
    """
    return tokenizer.batch_encode_plus([smile for smile in smiles], padding=True, add_special_tokens=True)



if __name__ == "__main__":
    smiles = ["CN(C)CCC[C@]1(OCC2=C1C=CC(=C2)C#N)C1=CC=C(F)C=C1",
              "CC1COC2=C3N1C=C(C(O)=O)C(=O)C3=CC(F)=C2N1CCN(C)CC1",
              "C[C@H]1COC2=C3N1C=C(C(O)=O)C(=O)C3=CC(F)=C2N1CCN(C)CC1",
              "[H][C@@]12CCN(C[C@@H]1C=C)[C@]([H])(C2)[C@@H](O)C1=C2C=C(OC)C=CC2=NC=C1",
              "[H][C@]1(C[C@@H]2CC[N@]1C[C@@H]2C=C)[C@H](O)C1=CC=NC2=CC=C(OC)C=C12",
              "CN(C)CCCC1(OCC2=C1C=CC(=C2)C#N)C1=CC=C(F)C=C1"]
    new = [preprocess_smiles(s) for s in smiles]
    new = list(set(new))
    print(new)