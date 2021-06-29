import numpy as np
import pandas as pd
# set random seed
np.random.seed(42)
from sklearn.manifold import TSNE
import glob
import sys

SELECTED_STRUCTURE = sys.argv[1]

if SELECTED_STRUCTURE == "DA":
    all_DA_AL_nfps = []
    for i in range(9):
        filename = glob.glob(f"DA_nfps/DA_AL_nfp_{i}.npy")[0]
        with open(filename, "rb") as f:
            all_DA_AL_nfps.append(np.load(f))

    filename = glob.glob(f"DA_nfps/DA_unlabelled_nfps.npy")[0]
    with open(filename, "rb") as f:
        DA_unlabelled_nfp = np.load(f)

    filename = glob.glob(f"DA_nfps/DA_final_3_nfps.npy")[0]
    with open(filename, "rb") as f:
        final_3_nfp = np.load(f)

    # indiv: 142452, 7101, 119, 112, 120, 119, 120, 120, 120, 120, 3
    # total: 142452, 149553, 149672, 149784, 149904, 150023, 150143, 150263, 150383, 150503
    # slicing to remove all the trailing 1s and 0s of every nfp array 

    all_sequenced_nfps = np.concatenate((DA_unlabelled_nfp[:-8],
                                        all_DA_AL_nfps[0][:-9],  
                                        all_DA_AL_nfps[1][:-1],
                                        all_DA_AL_nfps[2][:-8],
                                        all_DA_AL_nfps[3][:],
                                        all_DA_AL_nfps[4][:-1],
                                        all_DA_AL_nfps[5][:],
                                        all_DA_AL_nfps[6][:],
                                        all_DA_AL_nfps[7][:],
                                        all_DA_AL_nfps[8][:],
                                        final_3_nfp[:-7]))
    print(all_sequenced_nfps.shape)
    print()
    sequences = [142452, 149553, 149672, 149784, 149904, 150023, 150143, 150263, 150383, 150503]
    print(all_sequenced_nfps[:sequences[0]].shape, 
        all_sequenced_nfps[sequences[0]:sequences[1]].shape, 
        all_sequenced_nfps[sequences[1]:sequences[2]].shape,
        all_sequenced_nfps[sequences[2]:sequences[3]].shape,
        all_sequenced_nfps[sequences[3]:sequences[4]].shape,
        all_sequenced_nfps[sequences[4]:sequences[5]].shape,
        all_sequenced_nfps[sequences[5]:sequences[6]].shape,
        all_sequenced_nfps[sequences[6]:sequences[7]].shape,
        all_sequenced_nfps[sequences[7]:sequences[8]].shape,
        all_sequenced_nfps[sequences[8]:sequences[9]].shape,
        all_sequenced_nfps[sequences[9]:].shape,
        )
elif SELECTED_STRUCTURE == "DAD":
    all_DAD_AL_nfps = []
    for i in range(11):
        filename = glob.glob(f"DAD_nfps/DAD_AL_nfp_{i}.npy")[0]
        with open(filename, "rb") as f:
            all_DAD_AL_nfps.append(np.load(f))

    all_DAD_unlabelled_nfps = []
    for i in range(4):
        filename = glob.glob(f"DAD_nfps/DAD_unlabelled_nfps_{i}.npy")[0]
        with open(filename, "rb") as f:
            all_DAD_unlabelled_nfps.append(np.load(f))

    filename = glob.glob(f"DAD_nfps/DADx_1098.npy")[0]
    with open(filename, "rb") as f:
        DADx_1098_nfp = np.load(f)

    # indiv: 115455, 4914, 119, 120, 120, 120, 120, 120, 120, 120, 120, 120, 1
    # total: 115455, 120369, 120488, 120608, 120728, 120848, 120968, 121088, 121208, 121328, 121448, 121568
    # slicing to remove all the trailing 1s and 0s of every nfp array 

    all_sequenced_nfps = np.concatenate((
        all_DAD_unlabelled_nfps[0],
        all_DAD_unlabelled_nfps[1],
        all_DAD_unlabelled_nfps[2],
        all_DAD_unlabelled_nfps[3][:-5],
        all_DAD_AL_nfps[0][:-6], 
        all_DAD_AL_nfps[1][:-1],
        all_DAD_AL_nfps[2],
        all_DAD_AL_nfps[3],
        all_DAD_AL_nfps[4],
        all_DAD_AL_nfps[5],
        all_DAD_AL_nfps[6],
        all_DAD_AL_nfps[7],
        all_DAD_AL_nfps[8],
        all_DAD_AL_nfps[9],
        all_DAD_AL_nfps[10],
        DADx_1098_nfp[:-9]))
    print(all_sequenced_nfps.shape)
    print()
    sequences = [115455, 120369, 120488, 120608, 120728, 120848, 120968, 121088, 121208, 121328, 121448, 121568]
    print(all_sequenced_nfps[:sequences[0]].shape, 
        all_sequenced_nfps[sequences[0]:sequences[1]].shape, 
        all_sequenced_nfps[sequences[1]:sequences[2]].shape,
        all_sequenced_nfps[sequences[2]:sequences[3]].shape,
        all_sequenced_nfps[sequences[3]:sequences[4]].shape,
        all_sequenced_nfps[sequences[4]:sequences[5]].shape,
        all_sequenced_nfps[sequences[5]:sequences[6]].shape,
        all_sequenced_nfps[sequences[6]:sequences[7]].shape,
        all_sequenced_nfps[sequences[7]:sequences[8]].shape,
        all_sequenced_nfps[sequences[8]:sequences[9]].shape,
        all_sequenced_nfps[sequences[9]:sequences[10]].shape,
        all_sequenced_nfps[sequences[10]:sequences[11]].shape,
        all_sequenced_nfps[sequences[11]:].shape,
        )

elif SELECTED_STRUCTURE == "FIG4":
    with open("Combined_nfps/combined_initial_DA_labelled_nfps.npy", "rb") as f:
        initial_DA_nfps = np.load(f)
    with open("Combined_nfps/combined_initial_DAD_labelled_nfps.npy", "rb") as f:
        initial_DAD_nfps = np.load(f)
    with open("Combined_nfps/combined_selected_4_data.npy", "rb") as f:
        selected_nfps = np.load(f)
    
    all_sequenced_nfps = np.concatenate((initial_DA_nfps[:-9],
                                        initial_DAD_nfps[:-6],
                                        selected_nfps[:4]))
    print(all_sequenced_nfps.shape)
    print()
    sequences = [7101, 12015, 12019]
    print(all_sequenced_nfps[:sequences[0]].shape, 
        all_sequenced_nfps[sequences[0]:sequences[1]].shape, 
        all_sequenced_nfps[sequences[1]:sequences[2]].shape)

print()
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=200)
all_features_embedded = tsne.fit_transform(all_sequenced_nfps)

# save embedding 
with open(f"all_{SELECTED_STRUCTURE}_features_embedded.npy", "wb") as f:
    np.save(f, all_features_embedded)
print()
print("Final features shape:")
print(all_features_embedded.shape)
