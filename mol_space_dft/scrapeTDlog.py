"""
# Conversion of \_td.log files to Labelled Data Ver 1.0

Input: td.log files after TD-DFT for all generated molecules 
Output: csv file with the following columns:
- Molecule given name
- SMILES 
- S1 
- T1 (might be NA)
- S1-T1 gap (might be NA)
- HOMO
- LUMO
- Eg band gap


## Workflow for each molecule:

a) Molecule name and SMILES to be found from original csv of all generated molecules and their SMILES. Dont need to convert log to SMILES using Open Babel.

b) Scrape td.log files for S1 T1 values
- If have both S1 T1, take both 
- If dont have S1, ignore.
- If dont have T1 only, just take S1.

c) Scrape td.log files for HOMO LUMO with new instructions. 

d) Compile all values, get S1-T1 gap and HOMO LUMO gap, then output to final csv

"""

import os 
import pandas as pd 
'''
This is the main function it will call scrapeS1T1_from_dir() and scrapeHL_from_dir()
'''
def get_labelled_df(DirFiles, outdf):
    outdf = scrapeS1T1_from_dir(DirFiles, outdf)
    print("------------------ All S1 T1 values are scraped! ------------------")
    outdf = scrapeHL_from_dir(DirFiles, outdf)
    print("------------------ All HOMO LUMO values are scraped! ------------------")
    return outdf
'''
Here is to S1T1 value out, the input need to be a data frame at the outdf argument. The return will be the df with the S1
T1 and S1-T1 gap information
'''
# input a list of file names of all _td.log files in the directory to be scraped
def scrapeS1T1_from_dir(DirFiles, outdf):
    start_indicators = ['Input orientation', 'Standard orientation', 'Excitation energies and oscillator strengths']
    break_indicators = ['Rotational constants (GHZ)', 'Population analysis using the SCF density']

    full_energy_dict = {
        'Name': [],
        'EnergyType' : [],
        'EnergyValue' : []
    }
    for i in range(len(DirFiles)):
        current_mol_name = DirFiles[i][:-7]
        with open(DirFiles[i]) as f:
            lines = f.readlines()
            cont = False
            nonewritten = True
            energy_type_list = []
            energy_value_list = []

            for i in range(len(lines)):
                line = lines[i]
                if cont == True:
                    if any(phrase in line for phrase in break_indicators): 
                        # find any ends and stop writing 
                        cont = False
                    else: # continue writing the segment 
        #                 print(line)
        #                 print('')
                        nonewritten = False 
                        if ('Excited State') and ('Singlet-A' in line or 'Triplet-A' in line): 
                            # side processing of energy list
                            line_list = line.split()
                            try:
                                indi_index = line_list.index('eV')
                                energy_value = line_list[indi_index - 1] # find the energy value before eV
                                energy_type = line_list[indi_index - 2]
                                energy_index = line_list[indi_index - 3]
                                energy_type_list.append(energy_type)
                                energy_value_list.append(energy_value)
                            except ValueError:
                                print("ERROR")
                                energy_type_list.append("ERROR")
                                energy_value_list.append("ERROR")
                else:
                    if any(phrase in line for phrase in start_indicators): 
                        # properly write from the first part 
                        cont = True
        #                 print(line)
        #                 print('')
                        nonewritten = False

            if nonewritten == True:
                print('NOTHING FOUND')
            else:
                # append name
                full_energy_dict['Name'].append(current_mol_name)
                # append energy list
                full_energy_dict['EnergyType'].append(energy_type_list)
                full_energy_dict['EnergyValue'].append(energy_value_list)

    for k in range(len(full_energy_dict['Name'])):
        molname = full_energy_dict['Name'][k]
        typelist = full_energy_dict['EnergyType'][k]
        valuelist = full_energy_dict['EnergyValue'][k]
        try:
            S1_index = typelist.index('Singlet-A')
            S1_value = float(valuelist[S1_index])
        except ValueError: # if no S1
            S1_value = 'NA'
        try:
            T1_index = typelist.index('Triplet-A')
            T1_value = float(valuelist[T1_index])
        except ValueError: # if no T1
            T1_value = 'NA'
        # compute S1T1 gap
        if T1_value != 'NA' and S1_value != 'NA':
            S1T1_gap = abs(float(S1_value) - float(T1_value))
        else:
            S1T1_gap = 'NA'
        # now append these values to the df 
        outdf['S1'][outdf.index[outdf['Generated Molecule Name'] == molname]] = S1_value
        outdf['T1'][outdf.index[outdf['Generated Molecule Name'] == molname]] = T1_value
        outdf['S1T1 Gap'][outdf.index[outdf['Generated Molecule Name'] == molname]] = S1T1_gap
        print("Done for ", molname)

    return outdf
'''
This one is to extract the homo-lumo information and write into the data frame. 
'''

# input a list of file names of all _td.log files in the directory to be scraped
def scrapeHL_from_dir(DirFiles, outdf):
    for k in range(len(DirFiles)):
        current_mol_name = DirFiles[k][:-7]
        with open(DirFiles[k]) as f:
            lines = f.readlines()
            lastline = lines[-1]
            cont = True
            curr_line_index = 0
            while cont:
                line = lines[curr_line_index]
                if "Alpha virt. eigenvalues --" in line:
                    LUMO_line_index = curr_line_index
                    HOMO_line_index = curr_line_index - 1
                    LUMO_line = lines[LUMO_line_index]
                    HOMO_line = lines[HOMO_line_index]
                    LUMO_value = float(LUMO_line.split()[4]) # LUMO is first value of the line 
                    HOMO_value = float(HOMO_line.split()[-1]) # HOMO is last value of the line 
                    bandgap = abs(HOMO_value - LUMO_value)
                    outdf['HOMO'][outdf.index[outdf['Generated Molecule Name'] == current_mol_name]] = HOMO_value
                    outdf['LUMO'][outdf.index[outdf['Generated Molecule Name'] == current_mol_name]] = LUMO_value
                    outdf['Bandgap'][outdf.index[outdf['Generated Molecule Name'] == current_mol_name]] = bandgap
                    cont = False
                    print("Done ", current_mol_name)
                if line == lastline:
                    print("Last line reached, failure: ", current_mol_name)
                    outdf['HOMO'][outdf.index[outdf['Generated Molecule Name'] == current_mol_name]] = 'NA'
                    outdf['LUMO'][outdf.index[outdf['Generated Molecule Name'] == current_mol_name]] = 'NA'
                    outdf['Bandgap'][outdf.index[outdf['Generated Molecule Name'] == current_mol_name]] = 'NA'
                    cont = False
                curr_line_index += 1
    return outdf

