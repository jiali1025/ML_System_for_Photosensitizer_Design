import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import rdFMCS
from matplotlib import colors
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
import random
import copy

'''
This is the help file include the commonly used functions in this project
'''

# general function to connect a single bond between 2 atom positions in 2 different rdkit molecules
'''
Chem.CombineMols() will convert two mol object into one mol object. It is like a composite class with two components each.
one is a mol object, they are still separate fragments

Chem.EditableMol() will make an editable copy of the molecule object for further edition

DrawingOptions.includeAtomNumbers=True this is a option function to use to see molecules with their atom indexes, this 
can help user when user want to utilize the code

the .AddBond() method is a method belong to the mol object, it can add the bonds at the indexed position between two mol fragments

newmol = newmol.GetMol() it is to copy the information back into a "normal" molecule object for further use

'''
def connectBond(mol1, mol2, pos1, pos2):
    newmol = Chem.EditableMol(Chem.CombineMols(mol1, mol2))
    newmol.AddBond(pos1, pos2, order = Chem.rdchem.BondType.SINGLE) # add single bond only
    newmol = newmol.GetMol()
    return newmol


'''
this function takes in a list of molecules (mol objects): [mol1,mol2,mol3] actually iterable tuples should work as well but we used list

used MolsToGridImage() in Draw package of rdkit 
'''

# Returns a grid of molecules (image)
def getGrid(mol_list = None, molsPerRow = 4):
    if mol_list:
        img = Draw.MolsToGridImage(mol_list, molsPerRow = molsPerRow)
        return img
    else:
        print("Please provide mol_list as a list of rdkit mol objects!")
        return

'''
this function has two optional arguments mol_list and session, mol_list is a list of molecules (mol objects)
use a for loop to convert all mol objects in mol_list into smiles with Chem.MolToSmiles() function
the function will return the smiles list
however, it is used mainly for combine molecule function so it will create a csv file with name = session with a list of 
smiles of combined molecules

'''
# Convert mols to smiles 
def getSMILES(mol_list = None, session = None):
    smiles_list = []
    for mol in mol_list:
        smiles = Chem.MolToSmiles(mol)
        smiles_list.append(smiles)
    smiles_dict = {"SMILES": smiles_list} # dont know what it for
    newdf = pd.DataFrame(smiles_list, columns = ['SMILES']) #I dont think have this inside this function is good
    filename = "Combined_Molecules_{}.csv".format(session)
    newdf.to_csv(filename, header = True, index = True)
    print("OUTPUT {} has been created!".format(filename))
    return smiles_list 
'''
the mol file is like:
     RDKit          2D

  8  8  0  0  0  0  0  0  0  0999 V2000
    1.0607    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0000   -1.0607    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0607    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.0607    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    4.1820    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.1213   -1.0607    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.0607    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.1213    1.0607    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  1  0
  5  6  1  0
  6  7  1  0
  7  8  1  0
  4  1  1  0
  8  5  1  0
M  END

the gjf file is like:
%chk=DAD_1880_td.chk
%mem=20000MB
%nproc=24
#p b3lyp/6-31g(d) td=(nstates=15, 50-50)

Title Card Required

0  1
C           9.29370        -0.09299         0.00048
N           8.59317        -1.17547         0.00040
C           7.27139        -0.72138         0.00026
C           6.07217        -1.44130         0.00018
C           4.94790        -0.63367         0.00003
O           3.63571        -1.02875        -0.00007
C           2.91862         0.14701        -0.00018
C           1.47089         0.06450        -0.00029
C           0.66687         1.22984        -0.00027
C           0.95954         2.63987        -0.00021
C          -0.23994         3.26844        -0.00018
O          -1.29124         2.38686        -0.00021
    

The function will read the mol file to get lines of the mol file (a list). In these lines it will construct
a molatomlist for each mol file. This is due to the strcuture of the mol file, at position 31 of the line with atom coordinate
information there is the element str. And only the atom position line has that length so can get num_atoms out

The coordinates is extracted from the second for loop by taking all coordinate lines into list and extract the coordinatation
information out

write_gjf_lines contains a list of atom and coordinates in string datatype:['C 1.0607 0.0000 0.0000 ', 'N -0.0000 -1.0607 0.0000 ',.....]

Then the gjf file will be wrote with the information


'''
# The following function is used to convert a MOL file with its coordinates to a GJF file with the same atomic coordinates.
def convertMOLtoGJF(molname):
    molfile = molname + ".mol"
    with open(molfile, 'r') as mol:
        mollines = mol.readlines()
    # get list of coordinates and respective atoms from MOL file
    coord_list = []
    atom_list = []

    molatomlist = [] # contain['C','N','H'.........]
    for i in range(4, len(mollines)):
        try:
            x = mollines[i][31]
            molatomlist.append(x)
        except IndexError:
            pass
    num_atoms = len(molatomlist)

    for i in range(4, 4 + num_atoms):
        ls = mollines[i].strip().split()
        try:
            coords = ls[0:3]
            atom = ls[3]
            coord_list.append(coords)
            atom_list.append(atom)
        except IndexError:
            pass
    write_gjf_lines = []
    for i in range(len(atom_list)):
        line_to_add = ""
        line_to_add += atom_list[i] + " "
        line_to_add += coord_list[i][0] + " "
        line_to_add += coord_list[i][1] + " "
        line_to_add += coord_list[i][2] + " "
        write_gjf_lines.append(line_to_add)

    first_few_lines_gjf = ["%chk={}.chk".format(molname), 
                        "%mem=20000MB",
                        "%nproc=24",
                        "#p b3lyp/6-31g(d) opt", # edited version: 1 changed to l !!!
                        "",
                        "Title Card Required",
                        "\n"
                        "0  1",]

    with open('{}.gjf'.format(molname), 'w') as f:
        for line in first_few_lines_gjf:
            f.write(line + "\n")
        for line in write_gjf_lines:
            data = line.split()    # Splits on whitespace to create a list of information of one atom
            f.write('{0[0]:<10}{0[1]:>8}{0[2]:>16}{0[3]:>16}'.format(data) + "\n") #format everything correctly


'''
This function is to simply check the conversion from MOL to GJF. It compares whether the atom element matches
'''
# This function is used to check the difference in MOL and GJF atomic coordinates simply
def checkGJFAtoms(molname, smiles): # input string SMILES
    gjffile = molname + ".gjf"
    molfile = molname + ".mol"
    with open(gjffile, 'r') as gjf:
        gjflines = gjf.readlines()
    gjfatomlist = []
    for i in range(9, len(gjflines)):
        x = gjflines[i][0]
        if x != '\n':
            gjfatomlist.append(x)
    with open(molfile, 'r') as mol:
        mollines = mol.readlines()
    molatomlist = []
    for i in range(4, len(mollines)):
        try:
            x = mollines[i][31]
            molatomlist.append(x)
        except IndexError:
            pass
    return(molatomlist == gjfatomlist)