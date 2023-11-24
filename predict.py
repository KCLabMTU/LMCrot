import pyfiglet
text = "LMCrot"
font = "banner"
logo = pyfiglet.figlet_format(text)
print(logo)

"""
Create an argument parser
"""
import argparse
parser = argparse.ArgumentParser(description='Command-line arguments for LMCrot.')

# Define arguments with both long and short forms, and descriptive help messages
parser.add_argument('-d', '--device', choices=['CPU', 'GPU'], default='CPU', 
                    help='Specify the device for transformer model (ProtT5-U50-XL) to run on. Choices are "CPU" and "GPU". Default is "CPU".')
parser.add_argument('-m', '--mode', default='full-precision', 
                    help='Precision mode for computations of ProtT5. Default is "half-precision". If any other value is given, "full-precision" will be used for embeddings generation.')

# Parse arguments
args = parser.parse_args()

"""
import required libraries
"""
import os
import sys
import pickle
import tqdm
import plotext as plt
import numpy as np
import pandas as pd
import Bio
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
# for ProtT5 model
import torch
import transformers
from transformers import T5EncoderModel, T5Tokenizer
from utils.FEPS import get_FEPS_features
import re
import gc

print('\n\nVersion information:-')
print('-'*55)
print("Python version:", sys.version)
print("Numpy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("pyfiglet version:", pyfiglet.__version__)
print("Bio (biopython) version:", Bio.__version__)
print("Tensorflow version:", tf.__version__)
print("keras version inside tensorflow:", tf.keras.__version__)
print("Torch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print('-'*55)

print('\nBuilding Enviroment....')

"""
define file paths and other parameters
"""
# Access the parsed arguments
device_choice = args.device
mode = args.mode

#define paths for window size, model, input and output
win_size = 31 
input_fasta_file = "./input/sequence.fasta" # load your fasta sequence
output_csv_file = "./results.csv"   #save the output 
model_path = './models/LMCrot.h5' #LMCrot Model

print('\nwindow size: {}'.format(win_size))
print('Device Choice: {}'.format(device_choice))
print('Mode Selected: {}'.format(mode)) 

"""
Load tokenizer and pretrained model ProtT5
"""
# install SentencePiece transformers and bio-embeddings if not installed already
# !pip install -q SentencePiece transformers


tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
pretrained_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
if mode=='half-precision':
	pretrained_model = pretrained_model.half() 	

gc.collect()

# define devices
if device_choice == 'GPU' and torch.cuda.is_available():
	device = torch.device('cuda:0')
else:
	device = torch.device('cpu')

pretrained_model = pretrained_model.to(device)
pretrained_model = pretrained_model.eval()

def get_protT5_features(sequence): 
    """
    Extracts protein embeddings for a given protein sequence using the protT5 transformer.
    
    Input:
        sequence (str): A protein sequence represented as a string of amino acid characters.
                        Expected to have a length of 'l'.
    
    Returns:
        numpy.ndarray: A matrix of shape (l, 1024) containing the embeddings for the protein sequence.
                       Each row corresponds to an amino acid in the sequence, and each column corresponds 
                       to a feature in the embedding.
    
    Notes:
    - The function adds spaces between amino acid characters to prepare the sequence for the protT5 tokenizer.
    - Rare amino acids (U, Z, O, B) are replaced with the placeholder 'X' before feature extraction.
    - The function assumes the presence of a pretrained protT5 model and tokenizer, 
      and that the necessary configurations are set for them.
    """
    # add space in between amino acids
    sequence = [' '.join(e) for e in sequence]
    
    # replace rare amino acids with X
    sequence = [re.sub(r"[UZOB]", "X", seq) for seq in sequence]
    
    # set configurations and extract features
    ids = tokenizer.batch_encode_plus(sequence, add_special_tokens = True, padding = True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    
    with torch.no_grad():
        embedding = pretrained_model(input_ids = input_ids, attention_mask = attention_mask)
    embedding = embedding.last_hidden_state.cpu().numpy()

    seq_len = (attention_mask[0] == 1).sum()
    seq_emd = embedding[:,0,:]
    return seq_emd


def get_input_for_embedding(window):
    """
    Converts a window of protein sequence characters into corresponding integer encodings.
    
    Parameters:
    - window (str): A sequence of characters representing a window in a protein sequence. 
                    The characters should belong to the set 'ARNDCQEGHILKMFPSTWYVX-'.
    
    Returns:
    - numpy.array: An array of integer encodings corresponding to the characters in the given window.
                   If any character in the window is not in the defined set, the function returns None.
    
    Note:
    - The function defines a universe of possible input characters and their corresponding integer encodings.
      Any character outside this universe is considered invalid for encoding.
    """
    # define universe of possible input values
    alphabet = 'ARNDCQEGHILKMFPSTWYVX-'
    
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    
    for char in window:
        if char not in alphabet:
            return
    integer_encoded = np.array([char_to_int[char] for char in window])
    return integer_encoded


def extract_one_windows_position(sequence,site,window_size):
    
    '''
    Description: Extract a window from the given string at given position of given size
                (Need to test more conditions, optimizations)
    Parameters:
        protein_id (str): just used for debug purpose
        sequence (str): 
        site_residue (chr):
        window_size(int):
    Returns:
        string: a window/section
    '''
    if(sequence[site-1] != 'K'):
        print('there is no lsyine at {} site'.format(site-1))
        return
    
    half_window = int((window_size-1)//2)
    # if window is greater than seq length, make the sequence long by introducing virtual amino acids
    # To avoid different conditions for virtual amino acids, add half window everywhere
    #sequence = "X" * half_window + sequence + "X" * half_window
    sequence = sequence[::-1][:half_window][::-1] + sequence + sequence[:half_window]
    site=site+half_window
    section = sequence[site - 1-half_window : site + half_window]
    return section

def window_embeddings(site_position, protein_embeddings, window_size): 
    """
    Extracts a window of embeddings centered around a given site position within a protein sequence.
    
    Parameters:
    - site_position (int): The position of the site in the protein sequence around which the window of embeddings will be extracted.
    - protein_embeddings (numpy.array): The embeddings for each position in the protein sequence.
    - window_size (int): The size of the window to be extracted. Should be an odd number.
    
    Returns:
    - numpy.array: A window of embeddings centered around the given site position. 
                   If window size is even, an error message is printed and None is returned.
    
    Note:
    - The function uses periodic boundary conditions. That is, if the window extends beyond the beginning or end of the protein sequence, 
      it wraps around to the other end.
    - The window size should be odd to ensure an equal number of positions before and after the site position.
    """    
    if window_size % 2 == 0:
        print('[Error] Invalid window size. Window size should be odd')
        return

    half_window = (window_size - 1) // 2
    sequence_length = len(protein_embeddings)
    window_embedding = []

    for offset in range(-half_window, half_window + 1):
        index = (site_position - 1 + offset) % sequence_length
        window_embedding.append(protein_embeddings[index])

    window_embedding = np.array(window_embedding)
    
    return window_embedding

def get_physicochemical_features(peptide):
    """
    Extracts FEPS (Feature Extraction from Protein Sequence) features from a given peptide sequence.
    
    Parameters:
    - peptide (str): The peptide sequence for which the FEPS features are to be extracted.
    
    Returns:
    - numpy.ndarray: A reshaped array containing the FEPS features for the given peptide. 
                     The returned array has a shape of (1, number of features).
    
    Note:
    - The function first extracts the FEPS features using the `get_FEPS_features` method. 
      It then reshapes the extracted features to ensure compatibility with PhysicoDNN model.
    - The provided peptide should be a valid amino acid sequence.
    """
    feps_features = get_FEPS_features(peptide)
    return feps_features.reshape(1,-1)


def get_predictions(model, data):
    """
    Extracts the output from the second last layer of a given model for the provided data.
    
    Parameters:
    - model (keras.Model): The model from which the output is to be extracted.
    - data (numpy.array): The input data for which the output is required.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the output from the second last layer for the given data.
    
    Note:
    The function assumes that the model has at least two layers.
    """
    layer_name = model.layers[len(model.layers)-2].name #-1 for last layer, -2 for second last and so on"
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(data, verbose=0)

    return pd.DataFrame(intermediate_output)



# initialize empty result dataframe
results_df = pd.DataFrame(columns = ['prot_desc', 'position','site_residue', 'probability', 'prediction'])

# load base models
ProtT5_model = load_model('./models/T5ResConvBiLSTM.h5', compile = False)
Embedding_model = load_model('./models/EmbedCNN.h5', custom_objects={"K": K}, compile = False)
Physico_model = load_model('./models/PhysicoDNN.h5', custom_objects={"K": K}, compile = False)

#load saved scaler to scale the physicochemical and stacked features
scaler_phy = pickle.load(open('./utils/scaler_physico.sav', 'rb'))
scaler_fused = pickle.load(open('./utils/scaler_stacked.sav', 'rb')) 

print('\nLoading and Inititalization Done.. Intitiating Prediction....')
for seq_record in tqdm(SeqIO.parse(input_fasta_file, "fasta"),desc='Processing Sequences'):
    prot_id = seq_record.id
    sequence = seq_record.seq
    positive_predicted,negative_predicted = [],[]
    
    # extract protT5 for full sequence and store in temporary dataframe 
    pt5_all = get_protT5_features(sequence)
    # generate embedding features and window for each amino acid in sequence
    for index, amino_acid in enumerate(sequence):
        # check if AA is 'K' (lysine)
        if amino_acid in ['K']:
            site = index + 1

            # extract window
            window = extract_one_windows_position(sequence, site,win_size)
            
            # extract embedding_encoding
            X_test_embedding = np.reshape(get_input_for_embedding(window), (1, win_size))
            # get ProtT5 features extracted above
            X_test_pt5 = window_embeddings(site, pt5_all, win_size)
            # get FEPS physciochemical properties 
            #print(str(window))
            X_test_phy = get_physicochemical_features(str(window))
            X_test_phy_scaled = scaler_phy.transform(X_test_phy)
            
            prot_pred_test = get_predictions(ProtT5_model,  np.expand_dims(X_test_pt5, axis=0))
            emb_pred_test = get_predictions(Embedding_model, [X_test_embedding])
            phy_pred_test = get_predictions(Physico_model, X_test_phy_scaled)
            
            
            X_stacked_test = pd.concat([prot_pred_test, emb_pred_test,phy_pred_test],axis=1) #intermediate fusion
            X_stacked_test_scaled = scaler_fused.transform(X_stacked_test)

            # load combined model
            combined_model = load_model(model_path)
            y_pred = combined_model.predict(X_stacked_test_scaled, verbose = 0)[0][0]

            # append results to results_df
            results_df.loc[len(results_df)] = [prot_id, site, amino_acid, y_pred, int(y_pred > 0.5)]  #decision threshold cut-off is 0.5. You can change it a/c to your need

print('Prediction of all sites completed....')

# Check if output file exists
if os.path.exists(output_csv_file):
    os.remove(output_csv_file)

# Export results 
print('Saving results ...')
results_df.to_csv(output_csv_file, index = False)
print('Results saved to ' + output_csv_file+'\n')

grouped = results_df.groupby("prot_desc")

prot_descs = []
kcr_counts = []
non_kcr_counts = []

for prot_desc, group in grouped:
    prot_descs.append(prot_desc)
    kcr_counts.append(sum(group["prediction"] > 0.5))
    non_kcr_counts.append(len(group) - kcr_counts[-1])

colors = ["cyan", "magenta"]
plt.simple_multiple_bar(prot_descs, [kcr_counts, non_kcr_counts], width=100, labels=["Kcr", "non-Kcr"], title='Distribution of Predicted Crotonylation by Protein ID')
plt.show()

print('\n')
# Plotting the distribution of probability values in stacked form
num_bins = 10
bins = [i/num_bins for i in range(num_bins+1)]
labels = [f"{round(bins[i]*100)}-{round(bins[i+1]*100)}%" for i in range(num_bins)]

# Count Kcr and non-Kcr for each bin
kcr_counts = []
non_kcr_counts = []

for i in range(num_bins):
    bin_group = results_df[(results_df["probability"] >= bins[i]) & (results_df["probability"] < bins[i+1])]
    kcr_counts.append(sum(bin_group["prediction"] > 0.5))
    non_kcr_counts.append(len(bin_group) - kcr_counts[-1])

plt.simple_stacked_bar(labels, [kcr_counts, non_kcr_counts], width=100, title='Distribution of Probability Values', labels=["Kcr", "non-Kcr"])
plt.show()

print('\nDone..')
print('Exiting the program...')
