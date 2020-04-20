# https://github.com/iwatobipen/chemo_info/blob/master/rdkit_notebook/drawmol_with_idx.ipynb
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolToImage

smile_list = [
    # 'CC(C)(C)Cl', 'CC(C)(C)O',
    # 'CC(C)Br', 'CC(C)C'
    # 'CCc1ccccc1C', 'CCc1ccccc1O'
    'Cc1ccccc1Cl', 'Cc1ccccc1N', 'Cc1ccccc1O'
    # 'CCCC#C', 'CCCC=C'
    # 'CCCC(=C)C', 'CCCC(=O)C', 'CCCC(=O)O'
    ]

for smile in smile_list:
    mol = Chem.MolFromSmiles(smile)
    plt = MolToImage(mol)
    plt.show()