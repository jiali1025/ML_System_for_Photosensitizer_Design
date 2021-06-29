"""
Ver 1.0 
combineDA 0.5.1 and combineDAD 0.6.2 concatenated 

Functions to combine Donor and Acceptor molecules with a Bridge in between
- getDAB converts a table of SMILES for DAB to 3 lists of rdkit mol objects 
- D1-A-D2 combination up next with bridge consistent between D1-A and A-D2 
- Batch by batch for D1AD2

"""
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

# Returns a grid of molecules (image)
'''
this function takes in a list of molecules (mol objects): [mol1,mol2,mol3] actually iterable tuples should work as well but we used list

used MolsToGridImage() in Draw package of rdkit 
'''
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
    smiles_dict = {"SMILES": smiles_list}
    newdf = pd.DataFrame(smiles_list, columns = ['SMILES'])
    filename = "Combined_Molecules_{}.csv".format(session)
    newdf.to_csv(filename, header = True, index = True)
    print("OUTPUT {} has been created!".format(filename))
    return smiles_list 
    

# Converts CSV of SMILES to dataframe with rdkit molecules of all donors, acceptors, bridges.
"""
Input: CSV source of SMILES for Donors, Acceptors, Bridges and their atom idx positions for bonding 
Example input:
   Unnamed: 0       Donor  Donor_Pos    Acceptor  Acceptor_Pos1  \
0           0           C        0.0         C#N              0   
1           1  c1c[nH]cn1        2.0       O=S=O              1   
2           2  c1cn[nH]c1        3.0     c1cnoc1              0   
3           3     c1cocn1        0.0     c1cocn1              1   
4           4  c1nnc[nH]1        4.0  c1nnc[nH]1              0   
   Acceptor_Pos2  Acceptor_Pos3    Bridge  Bridge_Pos1  Bridge_Pos2  
0            NaN            NaN       C#C          0.0          1.0  
1            1.0            NaN       C=C          0.0          1.0  
2            NaN            NaN       COC          0.0          2.0  
3            NaN            NaN       CSC          0.0          2.0  
4            NaN            NaN  c1ccccc1          2.0          5.0  

Note that the input CSV source must not have invalid or NaN SMILES. Only the last few rows are NaN by default.



function explain: get_mol(): it is to get mol object from smiles. Chem.Kekulize() is to convert the aromatic bond into
double bond-single bond-db-sb......

the for loop is aiming to get all mol object from the dataframe, the try function is to avoid possible invalid smiles in
input file. Also this is due to the len() in df columns will be the maximum length of the df, the white space is filled with NAN
And the data will be stored in dictionary form.

Then new columns of mol object will be created 

Output: Pandas Dataframe with rdkit molecules of Donors, Acceptors, Bridges


"""
def getDAB(source):

    df = pd.read_csv(source)

    def get_mol(smiles):
        mol = Chem.MolFromSmiles(smiles)    
        if mol is None:
            return None
        Chem.Kekulize(mol)
        return mol

    mol_list_dict = {
            'Donor': [],
            'Acceptor': [],
            'Bridge': []
        }
    # get all the mols for donors, acceptors, and bridges
    for mol_type in list(mol_list_dict.keys()):
        for i in range(len(df[mol_type])):
            smi = df[mol_type][i]
            try:
                mol = get_mol(smi)
                mol_list_dict[mol_type].append(mol)
            except TypeError:
                print(smi, " HAS TYPE ERROR")
                mol_list_dict[mol_type].append('NaN')
    
    df['Donor Mol'] = mol_list_dict['Donor']
    df['Acceptor Mol'] = mol_list_dict['Acceptor']
    df['Bridge Mol'] = mol_list_dict['Bridge']

    return df


"""
Input: Pandas dataframe with rdkit mol objects (D, A, and B) processed from getDAB
This dataframe is created from getDAB() function
Ordered list of necessary information is extracted from the df. Order is important. In this the NA information is treated with care
A dictionary is created to store generated molecules information temporarily
num_bridges is +1 since donor and acceptor can be directly connect with a bond(with a specific bond type between specific D and A pair)
1. consider single bond without any atom as a basic bridge 
use two for loops to iterate over all donors and acceptors, record information in output_df dictionary
with the try and except to check whether all molecules are valid with conversion and reconversion, the Chem.SanitizeMol() error
due to benzene ring, and possible runtime error.

2. permutate by keeping bridge constant first; each batch has the same bridge
this for loops are similar to 1. except has one more degree of variation of the bridge
Functions: 
- Combine every D and A together 
- Vary bridges B in between

Output: Pandas dataframe with all the combined DA molecules and their respective recipe  (this dataframe is created from the dictionary)

"""

# Ver 0.5.1 adjusts bonding positions for every donor, acceptor, and bridge to be versatile
def combineDA(df, session):

    bridge_smiles_list = [i for i in df['Bridge'] if i != 'NaN']
    donor_smiles_list = [i for i in df['Donor'] if i != 'NaN']
    acceptor_smiles_list = [i for i in df['Acceptor'] if i != 'NaN']
    bridge_mol_list = [i for i in df['Bridge Mol'] if i != 'NaN']
    donor_mol_list = [i for i in df['Donor Mol'] if i != 'NaN']
    acceptor_mol_list = [i for i in df['Acceptor Mol'] if i != 'NaN'] 
    bridge_pos1_list = [int(i) for i in df['Bridge_Pos1'].dropna()]
    bridge_pos2_list = [int(i) for i in df['Bridge_Pos2'].dropna()]
    donor_pos_list = [int(i) for i in df['Donor_Pos'].dropna()]
    acceptor_pos_list = [int(i) for i in df['Acceptor_Pos1'].dropna()] #only for pos 1 of acceptor
    CURRENT_VERSION = 'Ver 0.5.1'

    current_mol_num = 0
    output_df = {
        'Generated Molecule Name': [],
        'Generated Molecule': [],
        'Generated Molecule SMILES': [],
        'Donor SMILES': [],
        'Donor ID': [],
        'Acceptor SMILES': [],
        'Acceptor ID': [],
        'Bridge SMILES': [],
        'Bridge ID': [],
    }

    num_bridges = len(bridge_mol_list) + 1
    num_donors = len(donor_mol_list)
    num_acceptors = len(acceptor_mol_list)
    max_possible_molecules = num_acceptors * num_donors * num_bridges

    logs = open("DA_Combination_Logs_{}.txt".format(session), 'x') #open for exclusive creation, failing if the file already exists
    logs.write("[ LOGS TO COMBINE DONORS AND ACCEPTORS: VER {} ]\n".format(CURRENT_VERSION))
    logs.write("Total donors: {}\n".format(num_donors))
    logs.write("Total acceptors: {}\n".format(num_acceptors))
    logs.write("Total bridges: {}\n".format(num_bridges))
    logs.write("")
    logs.write("Total possible combined molecules: {}\n\n".format(max_possible_molecules))

    print("Beginning combinations...")

    # 1. consider single bond without any atom as a basic bridge as well 
    b = num_bridges - 1 # single bond bridge id is the last it is to get the bond bridge id
    for d in range(num_donors):
        for a in range(num_acceptors):
            donor, acceptor = donor_mol_list[d], acceptor_mol_list[a]
            donor_smiles, acceptor_smiles = donor_smiles_list[d], acceptor_smiles_list[a]
            donor_pos, acceptor_pos = donor_pos_list[d], acceptor_pos_list[a]

            current_mol_name = "DA_" + str(current_mol_num)
            output_df['Generated Molecule Name'].append(current_mol_name)
            output_df['Donor SMILES'].append(donor_smiles)
            output_df['Donor ID'].append(d)
            output_df['Acceptor SMILES'].append(acceptor_smiles)
            output_df['Acceptor ID'].append(a)
            output_df['Bridge SMILES'].append('SingleBond')
            output_df['Bridge ID'].append(b)
            logs.write("Donor {} + Acceptor {} + Bridge {}.\n".format(d, a, b)) 
            
            try:
                # formation of new molecule
                newmol = connectBond(mol1 = donor, mol2 = acceptor, pos1 = donor_pos, pos2 = acceptor_pos + donor.GetNumAtoms()) #+donor.GetNumAtoms() is due to when combine two fragments the atomnum increase
                try:
                    Chem.SanitizeMol(newmol)
                    FinalSmiles = Chem.MolToSmiles(newmol)
                    FinalMol = Chem.MolFromSmiles(FinalSmiles)
                    if FinalMol: # reconversion is fine 
                        output_df['Generated Molecule'].append(FinalMol)
                        output_df['Generated Molecule SMILES'].append(FinalSmiles)
                        logs.write("Success: Molecule {} formed.\n".format(current_mol_name))
                    
                    else: # Catch Error 2: Invalid Molecule (reconversion from smiles is invalid to rdkit)
                        logs.write("INVALID RECONVERSION ERROR: Failed Molecule {}.\n".format(current_mol_name))
                        output_df['Generated Molecule'].append('FAIL')
                        output_df['Generated Molecule SMILES'].append('FAIL')
                except ValueError as e: # Catch Error 1: Sanitization error due to benzene ring 
                    logs.write("SANITIZATION ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                    logs.write(str(e))
                    output_df['Generated Molecule'].append('FAIL')
                    output_df['Generated Molecule SMILES'].append('FAIL')
                    logs.write('\n')
                logs.write('\n')
            except RuntimeError as e:
                logs.write("RUNTIME ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                output_df['Generated Molecule'].append('FAIL')
                output_df['Generated Molecule SMILES'].append('FAIL')
                logs.write(str(e))
                logs.write('\n\n')
            
            # iter to next molecule 
            current_mol_num += 1
    print("Bridge ID {} - Single Bond - Combinations Done!".format(b))
    logs.write("\n--- Bridge ID {} - Single Bond - Combinations Done!!---\n\n".format(b))

    # 2. permutate by keeping bridge constant first; each batch has the same bridge
    for b in range(num_bridges - 1):
        print("Currently at Bridge ID {}".format(b))
        logs.write("\n\n********** BATCH Bridge ID {} **********\n\n".format(b))
        for d in range(num_donors):
            for a in range(num_acceptors):
                donor, acceptor, bridge = donor_mol_list[d], acceptor_mol_list[a], bridge_mol_list[b]
                donor_smiles, acceptor_smiles, bridge_smiles = donor_smiles_list[d], acceptor_smiles_list[a], bridge_smiles_list[b],
                donor_pos, acceptor_pos, bridge_pos1, bridge_pos2 = donor_pos_list[d], acceptor_pos_list[a], bridge_pos1_list[b], bridge_pos2_list[b]

                current_mol_name = "DA_" + str(current_mol_num)
                output_df['Generated Molecule Name'].append(current_mol_name)
                output_df['Donor SMILES'].append(donor_smiles)
                output_df['Donor ID'].append(d)
                output_df['Acceptor SMILES'].append(acceptor_smiles)
                output_df['Acceptor ID'].append(a)
                output_df['Bridge SMILES'].append(bridge_smiles)
                output_df['Bridge ID'].append(b)
                logs.write("Donor {} + Acceptor {} + Bridge {}.\n".format(d, a, b)) 
                
                try:
                    # formation of new molecule; versatile bonding regardless of positions and bridge type
                    tempmol = connectBond(mol1 = donor, mol2 = bridge, pos1 = donor_pos, pos2 = donor.GetNumAtoms() + bridge_pos1)
                    newmol = connectBond(mol1 = tempmol, mol2 = acceptor, pos1 = bridge_pos2 + donor.GetNumAtoms(), pos2 = acceptor_pos + bridge.GetNumAtoms() + donor.GetNumAtoms()) #the atom position index is changed due to same reason before
                    try:
                        Chem.SanitizeMol(newmol)
                        FinalSmiles = Chem.MolToSmiles(newmol)
                        FinalMol = Chem.MolFromSmiles(FinalSmiles)
                        if FinalMol: # reconversion is fine 
                            output_df['Generated Molecule'].append(FinalMol)
                            output_df['Generated Molecule SMILES'].append(FinalSmiles)
                            logs.write("Success: Molecule {} formed.\n".format(current_mol_name))
                        
                        else: # Catch Error 2: Invalid Molecule (reconversion from smiles is invalid to rdkit)
                            logs.write("INVALID RECONVERSION ERROR: Failed Molecule {}.\n".format(current_mol_name))
                            output_df['Generated Molecule'].append('FAIL')
                            output_df['Generated Molecule SMILES'].append('FAIL')
                    except ValueError as e: # Catch Error 1: Sanitization error due to benzene ring 
                        logs.write("SANITIZATION ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                        logs.write(str(e))
                        output_df['Generated Molecule'].append('FAIL')
                        output_df['Generated Molecule SMILES'].append('FAIL')
                    logs.write('\n')
                except RuntimeError as e:
                    logs.write("RUNTIME ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                    output_df['Generated Molecule'].append('FAIL')
                    output_df['Generated Molecule SMILES'].append('FAIL')
                    logs.write(str(e))
                    logs.write('\n\n')
                
                # iter to next molecule 
                current_mol_num += 1
        print("Bridge ID {} Combinations Done!".format(b))
        logs.write("\n--- Bridge ID {} Combinations Done!!---\n\n".format(b))

    output_df = pd.DataFrame.from_dict(output_df)

    logs.write("----------------- END -----------------\n\n")
    num_success = len([i for i in output_df['Generated Molecule'] if i != 'FAIL'])
    num_fail = len([i for i in output_df['Generated Molecule'] if i == 'FAIL'])
    final_text = "Total possible combined molecules: {}\n".format(max_possible_molecules)
    final_text += "Total successes: {}\n".format(num_success)
    final_text += "Total failures: {}\n".format(num_fail)
    final_text += "\nSuccess rate: {:.2f}".format((num_success/max_possible_molecules)* 100)
    logs.write(final_text)
    print("")
    print("COMBINATION COMPLETE:")
    print(final_text)

    return output_df

"""
Input: Pandas dataframe with rdkit mol objects (D, A, and B) processed from getDAB
This function is similar to the DA combinator function but with more complexity.
df['Acceptor_Pos2'].notnull() this one will return a series of boolen information
list(df[df['Acceptor_Pos2'].notnull()]['Acceptor_Pos2'].index), returns all the df index of acceptors with > 1 pos 
The first for loop is quite similar to the DA one
The second for loop is also quite similar to DA one
Functions: 
- Combine in this sequence: D + B + A + B + D
- Acceptor: Only acceptors with more than 1 positions for bonding will be included 
- Bridge: Both bridges are the same 
- Donor: Both donors are the same 

Output: Pandas dataframe with all the combined DAD molecules and their respective recipe 

Example execution:
df = getDAB('D_A_B_Labelled.csv')
DAD_outdf = combineDAD(df, current_session)
DAD_outdf.to_csv('./Generated_Molecules_DAD_{}.csv'.format(current_session))

"""

def combineDAD(df, session):

    donor_smiles_list = [i for i in df['Donor'].dropna()]
    donor_pos_list = [int(i) for i in df['Donor_Pos'].dropna()] 
    donor_mol_list = [i for i in df['Donor Mol'] if i != 'NaN']

    acceptor_smiles_list = [i for i in df['Acceptor'].dropna()]
    acceptor_pos1_list = [int(i) for i in df['Acceptor_Pos1']] # dont drop NA there is no NA, if there is NA cant use int()
    acceptor_pos2_list = [i for i in df['Acceptor_Pos2']] #2nd position for bonding with the same D # dont drop NA # cant conver NaN to integer so have to convert to int later
    acceptor_mol_list = [i for i in df['Acceptor Mol'] if i != 'NaN'] #actually no need to check i !='NaN' since Donor always have and if with NaN it will be wrong, since the idx will be different

    dual_acceptor_IDs_list = list(df[df['Acceptor_Pos2'].notnull()]['Acceptor_Pos2'].index) # returns all the df index of acceptors with > 1 pos 

    bridge_smiles_list = [i for i in df['Bridge'].dropna()]
    bridge_pos1_list = [int(i) for i in df['Bridge_Pos1'].dropna()]
    bridge_pos2_list = [int(i) for i in df['Bridge_Pos2'].dropna()]
    bridge_mol_list = [i for i in df['Bridge Mol'] if i != 'NaN']

    current_mol_num = 0
    output_df = {
        'Generated Molecule Name': [],
        'Generated Molecule': [],
        'Generated Molecule SMILES': [],
        'Donor 1 SMILES': [],
        'Donor 1 ID': [],
        'Donor 2 SMILES': [],
        'Donor 2 ID': [],
        'Acceptor SMILES': [],
        'Acceptor ID': [],
        'Bridge SMILES': [],
        'Bridge ID': [],
    }

    CURRENT_VERSION = 'ver.05'
    num_bridges = len(bridge_mol_list) + 1
    num_donors = len(donor_mol_list)
    num_acceptors = len(dual_acceptor_IDs_list) # only consider dual pos acceptors in DAD combi
    max_possible_molecules = num_acceptors * num_donors * num_bridges  # 56 * 96 * 14 = 75264 
 
    logs = open("DAD_Combination_Logs_{}.txt".format(session), 'x')
    logs.write("[ LOGS TO COMBINE DONORS AND ACCEPTORS - DAD: VER {} ]\n".format(CURRENT_VERSION))
    logs.write("Total donors: {}\n".format(num_donors))
    logs.write("Total acceptors: {}\n".format(num_acceptors))
    logs.write("Total bridges: {}\n".format(num_bridges))
    logs.write("")
    logs.write("Total possible combined molecules: {}\n\n".format(max_possible_molecules))

    print("Beginning combinations...")

    # 1. consider single bond without any atom as a basic bridge as well 
    b = num_bridges - 1 # single bond bridge id is the last 
    for d in range(num_donors):
        for a in dual_acceptor_IDs_list: # Note: for acceptor, a is the index in the df for acceptors that have dual pos
            donor, acceptor = donor_mol_list[d], acceptor_mol_list[a]
            donor_smiles, acceptor_smiles = donor_smiles_list[d], acceptor_smiles_list[a]
            donor_pos, acceptor_pos1, acceptor_pos2 = donor_pos_list[d], acceptor_pos1_list[a], int(acceptor_pos2_list[a])

            current_mol_name = "DAD_" + str(current_mol_num)
            output_df['Generated Molecule Name'].append(current_mol_name)
            output_df['Donor 1 SMILES'].append(donor_smiles)
            output_df['Donor 1 ID'].append(d)
            output_df['Donor 2 SMILES'].append(donor_smiles) # same donor in DAD symmetry
            output_df['Donor 2 ID'].append(d) # same donor in DAD symmetry
            output_df['Acceptor SMILES'].append(acceptor_smiles)
            output_df['Acceptor ID'].append(a)
            output_df['Bridge SMILES'].append('SingleBond')
            output_df['Bridge ID'].append(b)
            logs.write("Donor {} + Acceptor {} + Donor {} + Bridge {}.\n".format(d, a, d, b)) 

            try:
                # formation of new molecule with D-A and then D-A-D (single bond between each molecule)
                DA = connectBond(mol1 = donor, mol2 = acceptor, 
                                pos1 = donor_pos, 
                                pos2 = donor.GetNumAtoms() + acceptor_pos1)
                DAD = connectBond(mol1 = DA, mol2 = donor, 
                                    pos1 = donor.GetNumAtoms() + acceptor_pos2,
                                    pos2 = donor.GetNumAtoms() + acceptor.GetNumAtoms() + donor_pos)

                try:
                    Chem.SanitizeMol(DAD)
                    FinalSmiles = Chem.MolToSmiles(DAD)
                    FinalMol = Chem.MolFromSmiles(FinalSmiles)
                    if FinalMol: # reconversion is fine 
                        output_df['Generated Molecule'].append(FinalMol)
                        output_df['Generated Molecule SMILES'].append(FinalSmiles)
                        logs.write("Success: Molecule {} formed.\n".format(current_mol_name))
                    
                    else: # Catch Error 2: Invalid Molecule (reconversion from smiles is invalid to rdkit)
                        logs.write("INVALID RECONVERSION ERROR: Failed Molecule {}.\n".format(current_mol_name))
                        output_df['Generated Molecule'].append('FAIL')
                        output_df['Generated Molecule SMILES'].append('FAIL')
                except ValueError as e: # Catch Error 1: Sanitization error due to benzene ring 
                    logs.write("SANITIZATION ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                    logs.write(str(e))
                    output_df['Generated Molecule'].append('FAIL')
                    output_df['Generated Molecule SMILES'].append('FAIL')
                    logs.write('\n')
                logs.write('\n')
            except RuntimeError as e:
                logs.write("RUNTIME ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                output_df['Generated Molecule'].append('FAIL')
                output_df['Generated Molecule SMILES'].append('FAIL')
                logs.write(str(e))
                logs.write('\n\n')
            
            # iter to next molecule 
            current_mol_num += 1
    print("Bridge ID {} - Single Bond - Combinations Done!".format(b))
    logs.write("\n--- Bridge ID {} - Single Bond - Combinations Done!!---\n\n".format(b))

    # 2. permutate by keeping bridge constant first; each batch has the same bridge
    for b in range(num_bridges - 1):
        print("Currently at Bridge ID {}".format(b))
        logs.write("\n\n********** BATCH Bridge ID {} **********\n\n".format(b))
        for d in range(num_donors):
            for a in dual_acceptor_IDs_list: # Note: for acceptor, a is the index in the df for acceptors that have dual pos
                donor, acceptor, bridge = donor_mol_list[d], acceptor_mol_list[a], bridge_mol_list[b]
                donor_smiles, acceptor_smiles, bridge_smiles = donor_smiles_list[d], acceptor_smiles_list[a], bridge_smiles_list[b]
                donor_pos, acceptor_pos1, acceptor_pos2, bridge_pos1, bridge_pos2 = donor_pos_list[d], acceptor_pos1_list[a], int(acceptor_pos2_list[a]), bridge_pos1_list[b], bridge_pos2_list[b]

                current_mol_name = "DAD_" + str(current_mol_num)
                output_df['Generated Molecule Name'].append(current_mol_name)
                output_df['Donor 1 SMILES'].append(donor_smiles)
                output_df['Donor 1 ID'].append(d)
                output_df['Donor 2 SMILES'].append(donor_smiles) # same donor in DAD symmetry
                output_df['Donor 2 ID'].append(d) # same donor in DAD symmetry
                output_df['Acceptor SMILES'].append(acceptor_smiles)
                output_df['Acceptor ID'].append(a)
                output_df['Bridge SMILES'].append(bridge_smiles)
                output_df['Bridge ID'].append(b)
                logs.write("Donor {} + Acceptor {} + Donor {} + Bridge {}.\n".format(d, a, d, b)) 
                
                try:
                    # formation of new molecule; versatile bonding regardless of positions and bridge type
                    # step by step to combine D-B-A-B-D
                    DB = connectBond(mol1 = donor, mol2 = bridge, pos1 = donor_pos, pos2 = donor.GetNumAtoms() + bridge_pos1)

                    DBA = connectBond(mol1 = DB, mol2 = acceptor, 
                                    pos1 = bridge_pos2 + donor.GetNumAtoms(), 
                                    pos2 = acceptor_pos1 + bridge.GetNumAtoms() + donor.GetNumAtoms())

                    DBAB = connectBond(mol1 = DBA, mol2 = bridge, 
                                    pos1 = acceptor_pos2 + bridge.GetNumAtoms() + donor.GetNumAtoms(), 
                                    pos2 = donor.GetNumAtoms() + bridge.GetNumAtoms() + acceptor.GetNumAtoms() + bridge_pos1)

                    DBABD = connectBond(mol1 = DBAB, mol2 = donor, 
                                        pos1 = donor.GetNumAtoms() + bridge.GetNumAtoms() + acceptor.GetNumAtoms() + bridge_pos2,
                                        pos2 = donor.GetNumAtoms() + 2 * bridge.GetNumAtoms() + acceptor.GetNumAtoms() + donor_pos)

                    try:
                        Chem.SanitizeMol(DBABD)
                        FinalSmiles = Chem.MolToSmiles(DBABD)
                        FinalMol = Chem.MolFromSmiles(FinalSmiles)
                        if FinalMol: # reconversion is fine 
                            output_df['Generated Molecule'].append(FinalMol)
                            output_df['Generated Molecule SMILES'].append(FinalSmiles)
                            logs.write("Success: Molecule {} formed.\n".format(current_mol_name))
                        
                        else: # Catch Error 2: Invalid Molecule (reconversion from smiles is invalid to rdkit)
                            logs.write("INVALID RECONVERSION ERROR: Failed Molecule {}.\n".format(current_mol_name))
                            output_df['Generated Molecule'].append('FAIL')
                            output_df['Generated Molecule SMILES'].append('FAIL')
                    except ValueError as e: # Catch Error 1: Sanitization error due to benzene ring 
                        logs.write("SANITIZATION ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                        logs.write(str(e))
                        output_df['Generated Molecule'].append('FAIL')
                        output_df['Generated Molecule SMILES'].append('FAIL')
                    logs.write('\n')
                except RuntimeError as e:
                    logs.write("RUNTIME ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                    output_df['Generated Molecule'].append('FAIL')
                    output_df['Generated Molecule SMILES'].append('FAIL')
                    logs.write(str(e))
                    logs.write('\n\n')
                
                # iter to next molecule 
                current_mol_num += 1
        print("Bridge ID {} Combinations Done!".format(b))
        logs.write("\n--- Bridge ID {} Combinations Done!!---\n\n".format(b))

    output_df = pd.DataFrame.from_dict(output_df)

    logs.write("----------------- END -----------------\n\n")
    num_success = len([i for i in output_df['Generated Molecule'] if i != 'FAIL'])
    num_fail = len([i for i in output_df['Generated Molecule'] if i == 'FAIL'])
    final_text = "Total possible combined molecules: {}\n".format(max_possible_molecules)
    final_text += "Total successes: {}\n".format(num_success)
    final_text += "Total failures: {}\n".format(num_fail)
    final_text += "\nSuccess rate: {:.2f}".format((num_success/max_possible_molecules)* 100)
    logs.write(final_text)
    print("")
    print("COMBINATION COMPLETE:")
    print(final_text)

    return output_df

"""
Input: Pandas dataframe with rdkit mol objects (D, A, and B) processed from getDAB
batch_sess: int 0 to 13 (bridge ID, 13 for single bond bridge)
Here the difference is the batch_sess, it is to control only combine with one kind of bridge to avoid out of memory problem
of the hardware device
Functions: 
- Same as combineDAD but now D1 and D2 are two different donors 
- Combine in this sequence: D1 + B + A + B + D2
- Acceptor: Only acceptors with more than 1 positions for bonding will be included 
- Bridge: Both bridges are the same 
- Donor: Two different donors

Output: Pandas dataframe of all rdkit mol objects of DxAD combination iterations and their respective recipe
Example execution:

df = getDAB('D_A_B_Labelled.csv')
current_session = 3
for i in range(0, 14): # batch_sess takes values int 0 to 13 corresponding to each connecting bridge
    batch_sess = i
    DAD_outdf = combineD1AD2_by_batch(df, current_session, batch_sess)
    DAD_outdf.to_csv('./Generated_Molecules_DxAD_{}_{}.csv'.format(current_session, batch_sess))
    time.sleep(60)
"""


# breaking down the steps to ease on ram --> generate bridge by bridge, 14 batches in total
def combineD1AD2_by_batch(df, session, batch_sess):

    donor_smiles_list = [i for i in df['Donor'].dropna()]
    donor_pos_list = [int(i) for i in df['Donor_Pos'].dropna()] 
    donor_mol_list = [i for i in df['Donor Mol'] if i != 'NaN']

    acceptor_smiles_list = [i for i in df['Acceptor'].dropna()]
    acceptor_pos1_list = [int(i) for i in df['Acceptor_Pos1']] # dont drop NA 
    acceptor_pos2_list = [i for i in df['Acceptor_Pos2']] #2nd position for bonding with the same D # dont drop NA # cant conver NaN to integer so have to convert to int later
    acceptor_mol_list = [i for i in df['Acceptor Mol'] if i != 'NaN']

    dual_acceptor_IDs_list = list(df[df['Acceptor_Pos2'].notnull()]['Acceptor_Pos2'].index) # returns all the df index of acceptors with > 1 pos 

    bridge_smiles_list = [i for i in df['Bridge'].dropna()]
    bridge_pos1_list = [int(i) for i in df['Bridge_Pos1'].dropna()]
    bridge_pos2_list = [int(i) for i in df['Bridge_Pos2'].dropna()]
    bridge_mol_list = [i for i in df['Bridge Mol'] if i != 'NaN']

    CURRENT_VERSION = 'ver.05'

    current_mol_num = 0

    num_bridges = len(bridge_mol_list) + 1
    num_donors = len(donor_mol_list)
    num_acceptors = len(dual_acceptor_IDs_list) # only consider dual pos acceptors in D1AD2 combi
    max_possible_molecules = num_acceptors * num_donors * num_donors  # nett 56 * 95 * 96 * 14 = 7150080
 
    output_df = {
        'Generated Molecule Name': [],
        'Generated Molecule': [],
        'Generated Molecule SMILES': [],
        'Donor 1 SMILES': [],
        'Donor 1 ID': [],
        'Donor 2 SMILES': [],
        'Donor 2 ID': [],
        'Acceptor SMILES': [],
        'Acceptor ID': [],
        'Bridge SMILES': [],
        'Bridge ID': [],
    }

    logs = open("DxAD_Combination_Logs_{}_{}.txt".format(session, batch_sess), 'x')
    logs.write("[ LOGS TO COMBINE DONORS AND ACCEPTORS - D1AD2: VER {} - Batch {}]\n".format(CURRENT_VERSION, batch_sess))
    logs.write("Total donors: {}\n".format(num_donors))
    logs.write("Total acceptors: {}\n".format(num_acceptors))
    logs.write("")
    logs.write("Total possible combined molecules: {}\n\n".format(max_possible_molecules)) 

    print("Beginning combinations...")

    # 1. consider single bond without any atom as a basic bridge 
    if batch_sess == 13:
        b = 13
        for a in dual_acceptor_IDs_list: # Note: for acceptor, a is the index in the df for acceptors that have dual pos
            for d1 in range(num_donors):
                for d2 in range(num_donors):
                    if d1 == d2: 
                        pass # only combine molecules with different donors for this function
                    else:    
                        donor1, donor2, acceptor = donor_mol_list[d1], donor_mol_list[d2], acceptor_mol_list[a]
                        donor1_smiles, donor2_smiles, acceptor_smiles = donor_smiles_list[d1], donor_smiles_list[d2], acceptor_smiles_list[a]
                        donor1_pos, donor2_pos, acceptor_pos1, acceptor_pos2 = donor_pos_list[d1], donor_pos_list[d2], acceptor_pos1_list[a], int(acceptor_pos2_list[a])

                        current_mol_name = "DxAD{}_".format(b) + str(current_mol_num)
                        output_df['Generated Molecule Name'].append(current_mol_name)
                        output_df['Donor 1 SMILES'].append(donor1_smiles)
                        output_df['Donor 1 ID'].append(d1)
                        output_df['Donor 2 SMILES'].append(donor2_smiles)
                        output_df['Donor 2 ID'].append(d2) 
                        output_df['Acceptor SMILES'].append(acceptor_smiles)
                        output_df['Acceptor ID'].append(a)
                        output_df['Bridge SMILES'].append('SingleBond')
                        output_df['Bridge ID'].append(b)
                        logs.write("Donor {} + Acceptor {} + Donor {} + Bridge {}.\n".format(d1, a, d2, b)) 
                        
                        try:
                            # formation of new molecule with D1-A and then D1-A-D2 (single bond between each molecule)
                            D1A = connectBond(mol1 = donor1, mol2 = acceptor, 
                                            pos1 = donor1_pos, 
                                            pos2 = donor1.GetNumAtoms() + acceptor_pos1)
                            D1AD2 = connectBond(mol1 = D1A, mol2 = donor2, 
                                                pos1 = donor1.GetNumAtoms() + acceptor_pos2,
                                                pos2 = donor1.GetNumAtoms() + acceptor.GetNumAtoms() + donor2_pos)
                            try:
                                Chem.SanitizeMol(D1AD2)
                                FinalSmiles = Chem.MolToSmiles(D1AD2)
                                FinalMol = Chem.MolFromSmiles(FinalSmiles)
                                if FinalMol: # reconversion is fine 
                                    output_df['Generated Molecule'].append(FinalMol)
                                    output_df['Generated Molecule SMILES'].append(FinalSmiles)
                                    logs.write("Success: Molecule {} formed.\n".format(current_mol_name))
                                
                                else: # Catch Error 2: Invalid Molecule (reconversion from smiles is invalid to rdkit)
                                    logs.write("INVALID RECONVERSION ERROR: Failed Molecule {}.\n".format(current_mol_name))
                                    output_df['Generated Molecule'].append('FAIL')
                                    output_df['Generated Molecule SMILES'].append('FAIL')
                            except ValueError as e: # Catch Error 1: Sanitization error due to benzene ring 
                                logs.write("SANITIZATION ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                                logs.write(str(e))
                                output_df['Generated Molecule'].append('FAIL')
                                output_df['Generated Molecule SMILES'].append('FAIL')
                                logs.write('\n')
                            logs.write('\n')
                        except RuntimeError as e:
                            logs.write("RUNTIME ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                            output_df['Generated Molecule'].append('FAIL')
                            output_df['Generated Molecule SMILES'].append('FAIL')
                            logs.write(str(e))
                            logs.write('\n\n')
                        # iter to next molecule 
                        current_mol_num += 1
        print("Bridge ID {} - Single Bond - Combinations Done!".format(b))
        logs.write("\n--- Bridge ID {} - Single Bond - Combinations Done!!---\n\n".format(b))

    else:
        # 2. Every other bridges according to bridge ID
        b = batch_sess # each run of this function will only combine a batch of molecules by bridge ID
        print("Currently at Bridge ID {}".format(b))
        logs.write("\n\n********** BATCH Bridge ID {} **********\n\n".format(b))

        for a in dual_acceptor_IDs_list: # Note: for acceptor, a is the index in the df for acceptors that have dual pos
            for d1 in range(num_donors):
                for d2 in range(num_donors):
                    if d1 == d2:
                        pass # only combine molecules with different donors for this function
                    else:    
                        donor1, donor2, acceptor, bridge = donor_mol_list[d1], donor_mol_list[d2], acceptor_mol_list[a], bridge_mol_list[b]
                        donor1_smiles, donor2_smiles, acceptor_smiles, bridge_smiles = donor_smiles_list[d1], donor_smiles_list[d2], acceptor_smiles_list[a], bridge_smiles_list[b]
                        donor1_pos, donor2_pos, acceptor_pos1, acceptor_pos2, bridge_pos1, bridge_pos2 = donor_pos_list[d1], donor_pos_list[d2], acceptor_pos1_list[a], int(acceptor_pos2_list[a]), bridge_pos1_list[b], bridge_pos2_list[b]

                        current_mol_name = "DxAD{}_".format(b) + str(current_mol_num)
                        output_df['Generated Molecule Name'].append(current_mol_name)
                        output_df['Donor 1 SMILES'].append(donor1_smiles)
                        output_df['Donor 1 ID'].append(d1)
                        output_df['Donor 2 SMILES'].append(donor2_smiles) 
                        output_df['Donor 2 ID'].append(d2) 
                        output_df['Acceptor SMILES'].append(acceptor_smiles)
                        output_df['Acceptor ID'].append(a)
                        output_df['Bridge SMILES'].append(bridge_smiles)
                        output_df['Bridge ID'].append(b)
                        logs.write("Donor {} + Acceptor {} + Donor {} + Bridge {}.\n".format(d1, a, d2, b)) 
                        
                        try:
                            # formation of new molecule; versatile bonding regardless of positions and bridge type
                            # step by step to combine D1-B-A-B-D2
                            D1B = connectBond(mol1 = donor1, mol2 = bridge, 
                                            pos1 = donor1_pos, 
                                            pos2 = donor1.GetNumAtoms() + bridge_pos1)
                            D1BA = connectBond(mol1 = D1B, mol2 = acceptor, 
                                            pos1 = bridge_pos2 + donor1.GetNumAtoms(), 
                                            pos2 = acceptor_pos1 + bridge.GetNumAtoms() + donor1.GetNumAtoms())
                            D1BAB = connectBond(mol1 = D1BA, mol2 = bridge, 
                                            pos1 = acceptor_pos2 + bridge.GetNumAtoms() + donor1.GetNumAtoms(), 
                                            pos2 = donor1.GetNumAtoms() + bridge.GetNumAtoms() + acceptor.GetNumAtoms() + bridge_pos1)
                            D1BABD2 = connectBond(mol1 = D1BAB, mol2 = donor2, 
                                                pos1 = donor1.GetNumAtoms() + bridge.GetNumAtoms() + acceptor.GetNumAtoms() + bridge_pos2,
                                                pos2 = donor1.GetNumAtoms() + 2 * bridge.GetNumAtoms() + acceptor.GetNumAtoms() + donor2_pos)
                            try:
                                Chem.SanitizeMol(D1BABD2)
                                FinalSmiles = Chem.MolToSmiles(D1BABD2)
                                FinalMol = Chem.MolFromSmiles(FinalSmiles)
                                if FinalMol: # reconversion is fine 
                                    output_df['Generated Molecule'].append(FinalMol)
                                    output_df['Generated Molecule SMILES'].append(FinalSmiles)
                                    logs.write("Success: Molecule {} formed.\n".format(current_mol_name))
                                
                                else: # Catch Error 2: Invalid Molecule (reconversion from smiles is invalid to rdkit)
                                    logs.write("INVALID RECONVERSION ERROR: Failed Molecule {}.\n".format(current_mol_name))
                                    output_df['Generated Molecule'].append('FAIL')
                                    output_df['Generated Molecule SMILES'].append('FAIL')
                            except ValueError as e: # Catch Error 1: Sanitization error due to benzene ring 
                                logs.write("SANITIZATION ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                                logs.write(str(e))
                                output_df['Generated Molecule'].append('FAIL')
                                output_df['Generated Molecule SMILES'].append('FAIL')
                            logs.write('\n')
                        except RuntimeError as e:
                            logs.write("RUNTIME ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                            output_df['Generated Molecule'].append('FAIL')
                            output_df['Generated Molecule SMILES'].append('FAIL')
                            logs.write(str(e))
                            logs.write('\n\n')
                        
                        # iter to next molecule 
                        current_mol_num += 1

            print("Bridge ID {} Combinations Done!".format(b))
            logs.write("\n--- Bridge ID {} Combinations Done!!---\n\n".format(b))

    output_df = pd.DataFrame.from_dict(output_df)

    logs.write("----------------- END -----------------\n\n")
    num_success = len([i for i in output_df['Generated Molecule'] if i != 'FAIL'])
    num_fail = len([i for i in output_df['Generated Molecule'] if i == 'FAIL'])
    final_text = "Total possible combined molecules: {}\n".format(max_possible_molecules)
    final_text += "Total successes: {}\n".format(num_success)
    final_text += "Total failures: {}\n".format(num_fail)
    final_text += "\nSuccess rate: {:.2f}".format((num_success/max_possible_molecules)* 100)
    logs.write(final_text)
    print("")
    print("COMBINATION COMPLETE:")
    print(final_text)
    
    return output_df

"""
    Input: Pandas dataframe with rdkit mol objects (D, A, and B) processed from getDAB
    
    Functions: 
    - Same as combineDAD but now D1 and D2 are two different donors 
    - Combine in this sequence: D1 + B + A + B + D2
    - Acceptor: Only acceptors with more than 1 positions for bonding will be included 
    - Bridge: Both bridges are the same 
    - Donor: Two different donors

    Output: Pandas dataframe of all rdkit mol objects of DxAD combination iterations and their respective recipe

It is a one shot version for D1-A-D2 conversion where all bridges will be automatically iterated.
However, this may cause computer RAM error due to the large amount of data to generate. 
"""

#v one shot conversion of D1-A-D2 
def combineD1AD2(df, session):

    donor_smiles_list = [i for i in df['Donor'].dropna()]
    donor_pos_list = [int(i) for i in df['Donor_Pos'].dropna()] 
    donor_mol_list = [i for i in df['Donor Mol'] if i != 'NaN']

    acceptor_smiles_list = [i for i in df['Acceptor'].dropna()]
    acceptor_pos1_list = [int(i) for i in df['Acceptor_Pos1']] # dont drop NA 
    acceptor_pos2_list = [i for i in df['Acceptor_Pos2']] #2nd position for bonding with the same D # dont drop NA # cant conver NaN to integer so have to convert to int later
    acceptor_mol_list = [i for i in df['Acceptor Mol'] if i != 'NaN']

    dual_acceptor_IDs_list = list(df[df['Acceptor_Pos2'].notnull()]['Acceptor_Pos2'].index) # returns all the df index of acceptors with > 1 pos 

    bridge_smiles_list = [i for i in df['Bridge'].dropna()]
    bridge_pos1_list = [int(i) for i in df['Bridge_Pos1'].dropna()]
    bridge_pos2_list = [int(i) for i in df['Bridge_Pos2'].dropna()]
    bridge_mol_list = [i for i in df['Bridge Mol'] if i != 'NaN']

    current_mol_num = 0

    num_bridges = len(bridge_mol_list) + 1
    num_donors = len(donor_mol_list)
    num_acceptors = len(dual_acceptor_IDs_list) # only consider dual pos acceptors in D1AD2 combi
    max_possible_molecules = num_acceptors * num_donors * num_donors * num_bridges  # 56 * 95 * 96 * 14 = 7150080
 
    template_df = {
        'Generated Molecule Name': [],
        'Generated Molecule': [],
        'Generated Molecule SMILES': [],
        'Donor 1 SMILES': [],
        'Donor 1 ID': [],
        'Donor 2 SMILES': [],
        'Donor 2 ID': [],
        'Acceptor SMILES': [],
        'Acceptor ID': [],
        'Bridge SMILES': [],
        'Bridge ID': [],
    }
    list_of_output_dfs = [] # will create 14 different output_dfs so as to split the memory across 14 different csv

    logs = open("DxAD_Combination_Logs_{}.txt".format(session), 'x')
    logs.write("[ LOGS TO COMBINE DONORS AND ACCEPTORS - D1AD2: VER {} ]\n".format(CURRENT_VERSION))
    logs.write("Total donors: {}\n".format(num_donors))
    logs.write("Total acceptors: {}\n".format(num_acceptors))
    logs.write("Total bridges: {}\n".format(num_bridges))
    logs.write("")
    logs.write("Total possible combined molecules: {}\n\n".format(max_possible_molecules)) 

    print("Beginning combinations...")

    # 1. consider single bond without any atom as a basic bridge as well 
    output_df = copy.deepcopy(template_df)
    b = num_bridges - 1 # single bond bridge id is the last 
    for a in dual_acceptor_IDs_list: # Note: for acceptor, a is the index in the df for acceptors that have dual pos
        for d1 in range(num_donors):
            for d2 in range(num_donors):
                if d1 == d2: 
                    pass # only combine molecules with different donors for this function
                else:    
                    donor1, donor2, acceptor = donor_mol_list[d1], donor_mol_list[d2], acceptor_mol_list[a]
                    donor1_smiles, donor2_smiles, acceptor_smiles = donor_smiles_list[d1], donor_smiles_list[d2], acceptor_smiles_list[a]
                    donor1_pos, donor2_pos, acceptor_pos1, acceptor_pos2 = donor_pos_list[d1], donor_pos_list[d2], acceptor_pos1_list[a], int(acceptor_pos2_list[a])

                    current_mol_name = "DxAD_" + str(current_mol_num)
                    output_df['Generated Molecule Name'].append(current_mol_name)
                    output_df['Donor 1 SMILES'].append(donor1_smiles)
                    output_df['Donor 1 ID'].append(d1)
                    output_df['Donor 2 SMILES'].append(donor2_smiles)
                    output_df['Donor 2 ID'].append(d2) 
                    output_df['Acceptor SMILES'].append(acceptor_smiles)
                    output_df['Acceptor ID'].append(a)
                    output_df['Bridge SMILES'].append('SingleBond')
                    output_df['Bridge ID'].append(b)
                    logs.write("Donor {} + Acceptor {} + Donor {} + Bridge {}.\n".format(d1, a, d2, b)) 
                    
                    try:
                        # formation of new molecule with D1-A and then D1-A-D2 (single bond between each molecule)
                        D1A = connectBond(mol1 = donor1, mol2 = acceptor, 
                                        pos1 = donor1_pos, 
                                        pos2 = donor1.GetNumAtoms() + acceptor_pos1)
                        D1AD2 = connectBond(mol1 = D1A, mol2 = donor2, 
                                            pos1 = donor1.GetNumAtoms() + acceptor_pos2,
                                            pos2 = donor1.GetNumAtoms() + acceptor.GetNumAtoms() + donor2_pos)
                        try:
                            Chem.SanitizeMol(D1AD2)
                            FinalSmiles = Chem.MolToSmiles(D1AD2)
                            FinalMol = Chem.MolFromSmiles(FinalSmiles)
                            if FinalMol: # reconversion is fine 
                                output_df['Generated Molecule'].append(FinalMol)
                                output_df['Generated Molecule SMILES'].append(FinalSmiles)
                                logs.write("Success: Molecule {} formed.\n".format(current_mol_name))
                            
                            else: # Catch Error 2: Invalid Molecule (reconversion from smiles is invalid to rdkit)
                                logs.write("INVALID RECONVERSION ERROR: Failed Molecule {}.\n".format(current_mol_name))
                                output_df['Generated Molecule'].append('FAIL')
                                output_df['Generated Molecule SMILES'].append('FAIL')
                        except ValueError as e: # Catch Error 1: Sanitization error due to benzene ring 
                            logs.write("SANITIZATION ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                            logs.write(str(e))
                            output_df['Generated Molecule'].append('FAIL')
                            output_df['Generated Molecule SMILES'].append('FAIL')
                            logs.write('\n')
                        logs.write('\n')
                    except RuntimeError as e:
                        logs.write("RUNTIME ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                        output_df['Generated Molecule'].append('FAIL')
                        output_df['Generated Molecule SMILES'].append('FAIL')
                        logs.write(str(e))
                        logs.write('\n\n')
                    # iter to next molecule 
                    current_mol_num += 1

    list_of_output_dfs.append(output_df) # store the latest output_df to the list 
    print("Bridge ID {} - Single Bond - Combinations Done!".format(b))
    logs.write("\n--- Bridge ID {} - Single Bond - Combinations Done!!---\n\n".format(b))

    # 2. permutate by keeping bridge constant first; each batch has the same bridge
    for b in range(num_bridges - 1):
        print("Currently at Bridge ID {}".format(b))
        output_df = copy.deepcopy(template_df)
        logs.write("\n\n********** BATCH Bridge ID {} **********\n\n".format(b))

        for a in dual_acceptor_IDs_list: # Note: for acceptor, a is the index in the df for acceptors that have dual pos
            for d1 in range(num_donors):
                for d2 in range(num_donors):
                    if d1 == d2:
                        pass # only combine molecules with different donors for this function
                    else:    
                        donor1, donor2, acceptor, bridge = donor_mol_list[d1], donor_mol_list[d2], acceptor_mol_list[a], bridge_mol_list[b]
                        donor1_smiles, donor2_smiles, acceptor_smiles, bridge_smiles = donor_smiles_list[d1], donor_smiles_list[d2], acceptor_smiles_list[a], bridge_smiles_list[b]
                        donor1_pos, donor2_pos, acceptor_pos1, acceptor_pos2, bridge_pos1, bridge_pos2 = donor_pos_list[d1], donor_pos_list[d2], acceptor_pos1_list[a], int(acceptor_pos2_list[a]), bridge_pos1_list[b], bridge_pos2_list[b]

                        current_mol_name = "DxAD_" + str(current_mol_num)
                        output_df['Generated Molecule Name'].append(current_mol_name)
                        output_df['Donor 1 SMILES'].append(donor1_smiles)
                        output_df['Donor 1 ID'].append(d1)
                        output_df['Donor 2 SMILES'].append(donor2_smiles) 
                        output_df['Donor 2 ID'].append(d2) 
                        output_df['Acceptor SMILES'].append(acceptor_smiles)
                        output_df['Acceptor ID'].append(a)
                        output_df['Bridge SMILES'].append(bridge_smiles)
                        output_df['Bridge ID'].append(b)
                        logs.write("Donor {} + Acceptor {} + Donor {} + Bridge {}.\n".format(d1, a, d2, b)) 
                        
                        try:
                            # formation of new molecule; versatile bonding regardless of positions and bridge type
                            # step by step to combine D1-B-A-B-D2
                            D1B = connectBond(mol1 = donor1, mol2 = bridge, 
                                            pos1 = donor1_pos, 
                                            pos2 = donor1.GetNumAtoms() + bridge_pos1)
                            D1BA = connectBond(mol1 = D1B, mol2 = acceptor, 
                                            pos1 = bridge_pos2 + donor1.GetNumAtoms(), 
                                            pos2 = acceptor_pos1 + bridge.GetNumAtoms() + donor1.GetNumAtoms())
                            D1BAB = connectBond(mol1 = D1BA, mol2 = bridge, 
                                            pos1 = acceptor_pos2 + bridge.GetNumAtoms() + donor1.GetNumAtoms(), 
                                            pos2 = donor1.GetNumAtoms() + bridge.GetNumAtoms() + acceptor.GetNumAtoms() + bridge_pos1)
                            D1BABD2 = connectBond(mol1 = D1BAB, mol2 = donor2, 
                                                pos1 = donor1.GetNumAtoms() + bridge.GetNumAtoms() + acceptor.GetNumAtoms() + bridge_pos2,
                                                pos2 = donor1.GetNumAtoms() + 2 * bridge.GetNumAtoms() + acceptor.GetNumAtoms() + donor2_pos)
                            try:
                                Chem.SanitizeMol(D1BABD2)
                                FinalSmiles = Chem.MolToSmiles(D1BABD2)
                                FinalMol = Chem.MolFromSmiles(FinalSmiles)
                                if FinalMol: # reconversion is fine 
                                    output_df['Generated Molecule'].append(FinalMol)
                                    output_df['Generated Molecule SMILES'].append(FinalSmiles)
                                    logs.write("Success: Molecule {} formed.\n".format(current_mol_name))
                                
                                else: # Catch Error 2: Invalid Molecule (reconversion from smiles is invalid to rdkit)
                                    logs.write("INVALID RECONVERSION ERROR: Failed Molecule {}.\n".format(current_mol_name))
                                    output_df['Generated Molecule'].append('FAIL')
                                    output_df['Generated Molecule SMILES'].append('FAIL')
                            except ValueError as e: # Catch Error 1: Sanitization error due to benzene ring 
                                logs.write("SANITIZATION ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                                logs.write(str(e))
                                output_df['Generated Molecule'].append('FAIL')
                                output_df['Generated Molecule SMILES'].append('FAIL')
                            logs.write('\n')
                        except RuntimeError as e:
                            logs.write("RUNTIME ERROR: Failed Molecule {}. Error as such:\n".format(current_mol_name))
                            output_df['Generated Molecule'].append('FAIL')
                            output_df['Generated Molecule SMILES'].append('FAIL')
                            logs.write(str(e))
                            logs.write('\n\n')
                        
                        # iter to next molecule 
                        current_mol_num += 1

        list_of_output_dfs.append(output_df) # store the latest output_df to the list 
        print("Bridge ID {} Combinations Done!".format(b))
        logs.write("\n--- Bridge ID {} Combinations Done!!---\n\n".format(b))

    list_of_final_dfs = []
    for df in list_of_output_dfs:
        list_of_final_dfs.append(pd.DataFrame.from_dict(df))

    logs.write("----------------- END -----------------\n\n")
    num_success = len([i for i in output_df['Generated Molecule'] if i != 'FAIL'])
    num_fail = len([i for i in output_df['Generated Molecule'] if i == 'FAIL'])
    final_text = "Total possible combined molecules: {}\n".format(max_possible_molecules)
    final_text += "Total successes: {}\n".format(num_success)
    final_text += "Total failures: {}\n".format(num_fail)
    final_text += "\nSuccess rate: {:.2f}".format((num_success/max_possible_molecules)* 100)
    logs.write(final_text)
    print("")
    print("COMBINATION COMPLETE:")
    print(final_text)

    return list_of_final_dfs



##################################################
###### Deprecated Old Code for Ver 0.4.1 #########
##################################################

# Deprecated: 0.4.1 Extract smiles from csv and returns 2 lists: both D and A molecules 
def getDA(source = 'D_A_bridge_SMILE.csv'):
    df = pd.read_csv(source)

    def get_mol(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.Kekulize(mol)
        return mol

    donor_mol_list = []
    for i in range(len(df['Donor'])):
        d = df['Donor'][i]
        try:
            donor_mol = get_mol(d)
            donor_mol_list.append(donor_mol)
        except TypeError:
            print(d, " HAS TYPE ERROR")

    acceptor_mol_list = []
    for i in range(len(df['Acceptor'])):
        a = df['Acceptor'][i]
        try:
            acceptor_mol = get_mol(a)
            acceptor_mol_list.append(acceptor_mol)
        except TypeError:
            print(a, " HAS TYPE ERROR")

    return donor_mol_list, acceptor_mol_list 

# Deprecated: 0.4.1 now no longer using this func to script donor atom bonding positions
def selectPos(mol_list = None, acceptor_atom_pos_list = None):
    # Rule to select donor connecting atom:
    # Select Nitrogen --> if 3 bonds, select para benzene --> if no Nitrogen, select Oxygen
    """
    From current dataset of 18 donors and 19 acceptors, the indexed positions are:
    donor_atom_pos_list = [13, 13, 13, 1, 12, 10, 24, 16, 10, 4, 4, 12, 1, 2, 0, 6, 5, 5]
    acceptor_atom_pos_list = [9, 5, 5, 4, 6, 6, 11, 8, 3, 4, 7, 8, 8, 5, 13, 5, 1, 5, 6]
    # getting donor atom pos manually as well due to difficulty in general rule scripting
    """

    donor_atom_pos_list = []
    N_O_present = False
    for mol in mol_list:
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'N': # rule 1: N prioritised
                donor_atom_pos_list.append(i)
                N_O_present = True 
        if N_O_present == False:
            for i in range(mol.GetNumAtoms()):
                if mol.GetAtomWithIdx(i).GetSymbol() == 'O': # rule 2: O next 
                    donor_atom_pos_list.append(i)
                    N_O_present = True 
        N_O_present = False
    # To select acceptor connecting atom, select manually 
    # here, we input a list of idx positions 
    acceptor_atom_pos_list = acceptor_atom_pos_list
    return donor_atom_pos_list, acceptor_atom_pos_list

# Ver 0.4.1 for combining DA and writing files; now use combineDA and combineDAD instead 
def combineMol(session, donor_mol_list, acceptor_mol_list, connecting_bridge_list, donor_atom_pos_list, acceptor_atom_pos_list):
    f = open("Combination_Logs_{}.txt".format(session), 'x')
    f.write("[ LOGS TO COMBINE DONORS AND ACCEPTORS: VER 0.5.1]\n")

    f.write("Total donors: {}\n".format(len(donor_mol_list)))
    f.write("Total acceptors: {}\n".format(len(acceptor_mol_list)))
    f.write("")
    max_possible_molecules = len(acceptor_mol_list) * len(donor_mol_list) * 4 # 4 different types of connecting bridges
    f.write("Total possible combined molecules: {}\n\n".format(max_possible_molecules))

    combined_mols_list = []
    sanitized_fail_list = []
    reconversion_fail_list = []
    m = 0 # success count
    sf = 0 # sanitization fail count
    rf = 0 # reconversion fail count

    CDoubleBond = Chem.MolFromSmiles('C=C') # id 0 and 1
    CTripleBond = Chem.MolFromSmiles("C#C") # id 0 and 1
    CBenzene = Chem.MolFromSmiles('c1ccccc1') # id 0 to 5; size 6

    for d in range(len(donor_mol_list)):
        donor = donor_mol_list[d]
        for a in range(len(acceptor_mol_list)):
            acceptor = acceptor_mol_list[a]
            f.write("Combining donor {} and acceptor {} together.\n".format(d, a))
            for c in range(len(connecting_bridge_list)):
                cmethod = connecting_bridge_list[c]
                try:
                    donor_pos = donor_atom_pos_list[d]
                    acceptor_pos = acceptor_atom_pos_list[a]
                    # varying connecting bridge type
                    if cmethod == 1: # single bond 
                        newmol = connectBond(mol1 = donor, mol2 = acceptor, pos1 = donor_pos, pos2 = acceptor_pos + donor.GetNumAtoms())

                    elif cmethod == 2: # double bond
                        # connect donor with C double bond first
                        newmol1 = connectBond(mol1 = donor, mol2 = CDoubleBond, pos1 = donor_pos, pos2 = donor.GetNumAtoms())
                        # then connect C double bond with acceptor 
                        newmol = connectBond(mol1 = newmol1, mol2 = acceptor, pos1 = 1 + donor.GetNumAtoms(), pos2 = acceptor_pos + 1 + donor.GetNumAtoms())

                    elif cmethod == 3: # triple bond
                        # connect donor with C triple bond first
                        newmol1 = connectBond(mol1 = donor, mol2 = CTripleBond, pos1 = donor_pos, pos2 = donor.GetNumAtoms())
                        # then connect C triple bond with acceptor 
                        newmol = connectBond(mol1 = newmol1, mol2 = acceptor, pos1 = 1 + donor.GetNumAtoms(), pos2 = acceptor_pos + 1 + donor.GetNumAtoms())

                    elif cmethod == 4: # benzene ring 
                        # connect donor with benzene first
                        newmol1 = connectBond(mol1 = donor, mol2 = CBenzene, pos1 = donor_pos, pos2 = donor.GetNumAtoms())
                        # then connect C double bond with acceptor 
                        newmol = connectBond(mol1 = newmol1, mol2 = acceptor, pos1 = 3 + donor.GetNumAtoms(), pos2 = acceptor_pos + 6 + donor.GetNumAtoms())

                    else:
                        print("INVALID CONNECTING BRIDGE LIST! Must be like [1,2,3,4] for 4 different connecting bridge types.")
                        return

                    try:
                        Chem.SanitizeMol(newmol)
                        TempSmi = Chem.MolToSmiles(newmol)
                        FinalMol = Chem.MolFromSmiles(TempSmi)
                        if FinalMol: # reconversion is fine 
                            combined_mols_list.append(FinalMol)
                            f.write("DONE: Molecule {}: Donor {} + Acceptor {} + Bridge {}.\n".format(m, d, a, c))
                            m += 1
                        else: # Catch Error 2: Invalid Molecule (reconversion from smiles is invalid to rdkit)
                            print("INVALID RECONVERSION ERROR: Donor {} + Acceptor {} + Bridge {}".format(d, a, c))
                            f.write("INVALID RECONVERSION ERROR: Failed Molecule R{}: Donor {} + Acceptor {} + Bridge {}.\n".format(rf, d, a, c))
                            reconversion_fail_list.append(newmol)
                            rf += 1
                    except ValueError as e: # Catch Error 1: Sanitization error due to benzene ring 
                        print("SANITIZATION ERROR: Donor {} + Acceptor {} + Bridge {}".format(d, a, c))
                        f.write("SANITIZATION ERROR: Failed Molecule S{}: Donor {} + Acceptor {} + Bridge {}; Error as such:\n".format(sf, d, a, c))
                        f.write(str(e))
                        sanitized_fail_list.append(newmol)
                        sf += 1
                    f.write('\n')
                except RuntimeError as e:
                    f.write("RUNTIME ERROR: Cannot combine Donor {} + Acceptor {} + Bridge {}; Error as such:\n.".format(d, a, c))
                    f.write(str(e))
                    f.write('\n\n')

    f.write("----------------- END -----------------\n\n")
    final_text = "Total possible combined molecules: {}\n".format(max_possible_molecules)
    final_text += "Total molecules created: {}\n".format(len(combined_mols_list))
    final_text += "Sanitization failed molecules: {}\n".format(len(sanitized_fail_list))
    final_text += "Reconversion failed molecules: {}\n".format(len(reconversion_fail_list))
    final_text += "Total failures: {}\n".format(max_possible_molecules - len(combined_mols_list))
    final_text += "\nSuccess rate: {:.2f}".format((len(combined_mols_list)/(max_possible_molecules)) * 100)
    f.write(final_text)
    print("")
    print("COMBINATION COMPLETE:")
    print(final_text)
    return combined_mols_list, sanitized_fail_list, reconversion_fail_list