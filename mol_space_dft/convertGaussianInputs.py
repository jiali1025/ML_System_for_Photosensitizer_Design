"""
refactored: getGaussianInput.py ver 1.0 + batch conversions of both DA and DAD to GJF + runBatchConvertLog

Dependency: Open Babel's command prompt scripts 
- Open Babel 3.1 https://anaconda.org/conda-forge/openbabel; remember to set path to Anaconda 

Two processes here to define as arguments in CLI
1. SMILES --> GJF and TXT for structural optimizations
2. LOG --> GJF and TXT for TD DFT

"""
import openbabel
import numpy as np
import pandas as pd 
import os 
import time
import sys
import glob

# 1. Convert SMILES to gjf files with 3d coordinates first 
def convert_smiles_to_gjf(smiles_list, molecule_name_list):
    for i in range(len(smiles_list)):
        smi = smiles_list[i]
        filename = molecule_name_list[i]
        print("Currently converting SMILES {} to file {}.gjf".format(smi, filename))
        sys_text = 'obabel -:"{}" -O {}.gjf --gen3d'.format(smi, filename)    
        os.system(sys_text)
    print("Initial conversion to GJFs DONE!\nTotal number of molecule files in this batch: {}\n\n".format(len(smiles_list)))
    return  

# 1. Convert log files to .gjf again using Open Babel, but now for TD-DFT calculations 
def convert_log_to_gjf_td(log_list):
    """
    Given log list in the directory, convert them to _td.gjf files 
    Then also return the molecule_name_list to be used for later steps
    """
    molecule_name_list = []
    for i in range(len(log_list)):
        filename = log_list[i][:-4] # DA_24877.log to DA_24877 is filename
        newfilename = filename + '_td' # append _td to filename
        molecule_name_list.append(newfilename)
        print("Currently converting {}.log to {}.gjf".format(filename, newfilename))
        sys_text = 'obabel "{}".log -O {}.gjf"'.format(filename, newfilename) 
        os.system(sys_text)
    print("Initial conversion to TD GJFs DONE!\nTotal number of molecule files in this batch: {}\n\n".format(len(log_list)))
    return molecule_name_list

# 2. Edit gjf files to be runnable and create txt job file for every molecule 
# refactored version for both structural opt and td dft gjf files
def edit_gjf_file(filename, process = "TD"):
    """
    process
    - SO for structural opt
    - TD for TD-DFT 
    """
    with open('{}.gjf'.format(filename), 'r') as f:
        lines = f.readlines()
    with open('{}.gjf'.format(filename), 'w') as f:
        for line in lines[4:]: # only take the lines from line 4 onwards and remove the rest on top
            f.write(line)

    # lines to be added to the top of every gjf file for the job to run subsequently
    if process == 'SO':
        gjf_lines_to_add = ["%chk={}.chk".format(filename), 
                            "%mem=20000MB",
                            "%nproc=24",
                            "#p b3lyp/6-31g(d) opt", 
                            "",
                            "Title Card Required"] 
    elif process == 'TD':
        gjf_lines_to_add = ["%chk={}.chk".format(filename), 
                            "%mem=20000MB",
                            "%nproc=24",
                            "#p b3lyp/6-31g(d) td=(nstates=15, 50-50)",
                            "",
                            "Title Card Required"]
    else:
        print("No such method for GJF defined.")
        return
        
    def prepend_multiple_lines(file_name, list_of_lines): 
        # to add lines to the top of every gjf file 
        dummy_file = file_name + '.gjf'
        with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
            for line in list_of_lines:
                write_obj.write(line + '\n')
            for line in read_obj:
                write_obj.write(line)
        os.remove(file_name)
        os.rename(dummy_file, file_name)
        return

    prepend_multiple_lines('{}.gjf'.format(filename), gjf_lines_to_add)
    return

# 3. Create a job txt file for PBS 
def create_job_txt(filename):
    default_reference_lines = ['#!/bin/bash \n','\n','\n','#PBS -P Project_Name_of_Job  \n','\n','#PBS -q parallel24 \n',
                         '\n','#PBS -l select=1:ncpus=24:mpiprocs=24:mem=160GB \n','\n','#PBS -j oe \n','\n',
                         '#PBS -N O46_td1\n', # line 11 to edit
                         '###  -N Job_Name: set filename for standard output/error message.\n','\n','\n','\n',
                         'cd $PBS_O_WORKDIR;   ## Change to the working dir in the exec host \n','\n','\n','\n',
                         '##--- Put your exec/application commands below ---  \n','\n',
                         'g09 <O46_td1.gjf > O46_td1.log\n', # line 22 to edit
                         '\n','\n','\n','##--- END HERE --- \n','\n','\n']
    job_ref_lines = default_reference_lines[:]
    job_ref_lines[11] = '#PBS -N {}\n'.format(filename)
    job_ref_lines[22] = 'g09 <{}.gjf > {}.log\n'.format(filename, filename)
    
    with open('{}.txt'.format(filename), 'w') as f:
        for line in job_ref_lines:
            f.write(line)
    return

# Together: combining both gjf and txt steps
# given molecule name for the generated molecule, get completed gjf and txt files in directory
def generate_gjf_txt_batch(molecule_name_list, process = "SO"):
    for filename in molecule_name_list:
        create_job_txt(filename)
        if process in ["SO", "TD"]:
            edit_gjf_file(filename, process)
        else:
            print("No such method for GJF conversion.")
            return
        print('Both gjf and txt files created for: {} !'.format(filename))
    return 


#################################################

# command line executions 
"""
python convertGaussianInputs.py X 
- X = 1: Convert SMILES from CSV in the directory to GJF and TXT files
- X = 2: Convert all LOG files in the directory to td.GJF and td.TXT files

"""

process = int(sys.argv[1])
print(process)
if process == 1:
    print("Get files for Structural Optimization: Converting SMILES in CSV to GJF and TXT files...")
    csv_file = glob.glob("*.csv")[0]
    sample_df = pd.read_csv(csv_file)
    print("CSV selected: {}".format(csv_file))
    convert_smiles_to_gjf(list(sample_df['SMILES']), list(sample_df['MOLNAME']))
    generate_gjf_txt_batch(list(sample_df['MOLNAME']), process = "SO")
    print("DONE! All GJF and TXT files are generated in this directory.")

elif process == 2:
    print("Get files for TD DFT: Converting all LOG in directory to td.GJF and td.TXT files...")
    dir_log_list = []
    for file in os.listdir():
        if file.endswith(".log"):
            dir_log_list.append(file)
    print("Total LOG files in this directory to be converted: ", len(dir_log_list))
    molecule_name_list = convert_log_to_gjf_td(dir_log_list)
    generate_gjf_txt_batch(molecule_name_list, process = "TD")
    print("DONE! All td.GJF and td.TXT files are generated in this directory.")

else:
    print("There is no such method. Please recheck script.")
