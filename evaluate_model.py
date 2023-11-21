#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:
import pyfiglet
text = "LMCrot"
font = "banner"

logo = pyfiglet.figlet_format(text)
print(logo)

print('Importing all the required libraries..')
import os
import sys
import random
import warnings
import pickle
import Bio
from Bio import SeqIO
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    auc,
    confusion_matrix,
    matthews_corrcoef,
    accuracy_score,
    classification_report,
    f1_score,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K, regularizers, Model, Sequential, layers, optimizers, losses, callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    Conv2D, Embedding, MaxPooling2D, Conv1D, Dense, MaxPooling1D, Input,
    Flatten, LSTM, Dropout, Bidirectional, Normalization, Reshape,
    Lambda, LeakyReLU
)
print('Importing Done!!\n')
print('Version information:-')
print('-'*55)
print("Python version:", sys.version)
print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
print("seaborn version:", sns.__version__)
print("tqdm version:", tqdm.__version__)
print("pyfiglet version:", pyfiglet.__version__)
print("matplotlib version:", matplotlib.__version__)
print("scikit-learn version:", sklearn.__version__)
print("BioPython version:", Bio.__version__)
print("tensorflow version:", tf.__version__)
print("keras version inside tensorflow:", tf.keras.__version__)
print('-'*55)


# In[2]:


# Set the random seeds
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# # Helper functions

# In[3]:


def extract_one_windows_position(protein_id,sequence,site_residue,site,window_size): 
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
    if (window_size%2)==0:
        print('Error: Enter odd number window size')
        return 0
    
    half_window = int((window_size-1)//2)
    
    if(sequence is None):
        print('No sequence for [protein_id,site_residue,window_size] ='+str([protein_id,site_residue,window_size]))
        return 0
    else:
        seq_length = len(sequence)
    
    if(sequence[site-1] not in site_residue): # check if site_residue at position site is valid
        print('Given site-residue and site does not match [protein_id,site_residue, site] ='+str([protein_id,site_residue,site]))
        return 0
    
    
    # if window is greater than seq length, make the sequence long by introducing virtual amino acids
    # To avoid different conditions for virtual amino acids, add half window everywhere
    
    sequence = sequence[::-1][:half_window][::-1] + sequence + sequence[:half_window] #circular permutation
    site = site + half_window
    section = sequence[site - 1 - half_window : site + half_window]
    
    return section


# In[4]:


def trim_seq(df): #handy when seq. length is too large for ProtT5 to handle
    """
    Trims or slices protein sequences to a specified size.

    This function processes protein sequences longer than a specified window 
    (default 4499). Depending on the position of a specified site within the 
    sequence, the sequence may be trimmed from the start, end, or around the site.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing protein information. Expected columns are:
        - UniProt: The protein identifier.
        - Position: The position of a specific site within the protein sequence.
        - sequences: The protein sequence string.
        - Target: Target label or information.
        
    Returns:
    -------
    newdf : pd.DataFrame
        A DataFrame with the same columns as the input, but with potentially modified 
        sequences and positions. The columns are:
        - UniProt: The protein identifier (unchanged).
        - Position_New: The potentially updated position of the site after sequence trimming.
        - Position_Old: The original position of the site in the protein sequence.
        - sequences: The potentially trimmed protein sequence string.
        - Target: Target label or information (unchanged).

    Notes:
    -----
    The function also prints statistics about how many sequences were processed 
    and the type of trimming operation performed on them.

    The default window size is 4499, and the site will be at the center of the window 
    if the sequence is trimmed around it.
    """
    protid=[]
    position_new=[]
    position_old=list(df.Position)
    sequences =[]
    target=[]
    window=4499 #seq. of length 4499 
    c,n,wo,sb,se=0,0,0,0,0
    
    print("Truncating large sequences.....")
    print("Current max seq size allowed: "+str(window)+'\n')
    print("So, site will be at: "+str((window//2) + 1)+'\n')
  
    for index,row in tqdm.tqdm(df.iterrows(), desc="Processing sequences"):
        if len(row.sequences)>window: #do some processing if length > 4499 (win size)
            if row.Position<window: #slicing from start
                peptide=row.sequences[:window]
                site=row.Position
                n+=1
                sb+=1
            elif row.Position-1 > (len(row.sequences)-window): #slicing from end
                peptide=row.sequences[-window:]
                site= row.Position - (len(row.sequences)-window)
                n+=1
                se+=1
            else: #if the site is somewhere in the middle (take window of 4459 in this case)
                peptide=extract_one_windows_position(row.UniProt,row.sequences,['K'],row.Position,window) #S,T 
                site=(window//2) + 1
                n+=1
                wo+=1      
        else: #do nothing if length is < 4999
            peptide = row.sequences
            site=row.Position
        protid.append(row.UniProt)
        position_new.append(site)
        sequences.append(peptide)
        target.append(row.Target)
    newdf=pd.DataFrame([protid,position_new,position_old,sequences,target]).T
    newdf.columns=['UniProt','Position_New','Position_Old','sequences','Target']
    print('\nTotal Sampes: '+str(df.shape[0]))
    print("Processed "+str(n)+" sequences from total samples:")
    print("> Slicing sequence (from start) operation done on "+str(sb)+" samples.")
    print("> Slicing sequence (from end) operation done on "+str(se)+" samples.")
    print("> Window operation done on "+str(wo)+" samples\n")
    print("Done!!")
    print('-'*50)
    return newdf


# In[5]:


# extract W x n embeddings where W length window is centered around 'K' residues
def window_embeddings(df, protein_embeddings, window_size): 
    """
    Extracts embeddings for sequences with a specified window size centered around 'K' residues.

    This function returns a window of specified size centered around sites specified 
    in the input DataFrame. It operates in a circular manner, meaning if the window 
    goes beyond the sequence's start or end, it wraps around.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing protein information. It is expected to have a column 'Position_New' 
        which indicates the position of 'K' residues in the protein sequence.

    protein_embeddings : list of lists
        A nested list where each inner list contains ProtT5 embeddings for a protein sequence. 
        The order should correspond to the sequences in the `df`.

    window_size : int
        The size of the window to extract. Must be an odd number to have a center.

    Returns:
    -------
    window_embeddings : np.array
        A numpy array containing windowed embeddings. Each row corresponds to a protein sequence, 
        and the columns contain the embeddings for the window around the 'K' residue.

    Notes:
    -----
    The function will print an error message and return None if an even window size is provided.
    """
    
    if window_size % 2 == 0:
        print('[Error] Invalid window size. Window size should be odd')
        return
    col = 'Position_New'
    df[col] = df[col].astype(int)
    site_indices = list(df[col])  # List of site indices
    half_window = (window_size - 1) // 2

    # Extract embeddings for the window around the site in a circular manner
    window_embeddings = []
    for i, site_index in enumerate(site_indices):
        window = []
        sequence_length = len(protein_embeddings[i])
        for offset in range(-half_window, half_window + 1):
            # Subtract 1 from the site index when calculating the start of the window
            index = (site_index - 1 + offset) % sequence_length
            window.append(protein_embeddings[i][index])
        window_embeddings.append(window)

    window_embeddings = np.array(window_embeddings)
    
    print('Extraction of window embeddings successful!!')
    
    return window_embeddings


# In[6]:


def get_input_for_embedding(fasta_file):
    """
    Converts protein sequences from a FASTA file into integer encodings.

    This function reads protein sequences from a provided FASTA file and 
    returns their integer encodings, suitable for input into an embedding layer. 
    The integer encodings are based on a predefined alphabet of amino acid characters.

    Parameters:
    ----------
    fasta_file : str
        Path to the FASTA file containing protein sequences.

    Returns:
    -------
    encodings : np.array
        A numpy array where each row is an integer-encoded protein sequence. 

    Notes:
    -----
    If the sequence contains a character not in the predefined alphabet, 
    the function will return None.
    """
    
    encodings = []
    
    # define universe of possible input values
    alphabet = 'ARNDCQEGHILKMFPSTWYVUX-'
    
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        data = seq_record.seq
        for char in data:
            if char not in alphabet:
                return
        integer_encoded = [char_to_int[char] for char in data]
        encodings.append(integer_encoded)
    encodings = np.array(encodings)
    return encodings


# In[7]:


def load_all_models(model_names):
    """
    Load multiple Keras models from specified filenames.

    Given a list of model names, this function attempts to load each model 
    from the corresponding filename. The filenames are constructed using a 
    predefined path pattern and the provided model names. 
    
    Parameters:
    ----------
    model_names : list
        List of strings, where each string is the name of a Keras model 
        (without file extension). 

    Returns:
    -------
    all_models : list
        A list containing the loaded Keras models.

    Notes:
    -----
    The function assumes a filename pattern of 'final_models_v2/{model_name}.h5'
    for the models and will attempt to load each model using this pattern.
    
    If there's an issue loading a model, an error will be raised by the 
    underlying `load_model` function.
    """
    print('Initiating Intermediate-Fusion. Loading base models...\n')
    all_models = list()
    for model in model_names:
        filename = './models/'+ model + '.h5'
        model = load_model(filename, custom_objects={"K": K},compile = False)
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# In[8]:


def get_predictions(model,data,mode=1): 
    """
    Extracts either intermediate features or decision-level outputs from a given model.

    For a given Keras model, this function can either extract features from an intermediate 
    layer (typically the second last layer) or produce decision-level outputs for given data.

    Parameters:
    ----------
    model : keras.Model
        The trained Keras model from which predictions or features are to be extracted.

    data : array-like
        The input data for the model. Shape should match the model's input shape.

    mode : int, optional, default=1
        Determines the type of extraction:
        - 1: Features from the final hidden layer.
        - 2: Decision level output.

    Returns:
    -------
    features_or_output : pd.DataFrame
        A dataframe containing either the extracted features or decision-level outputs.

    Notes:
    -----
    When mode is set to 1, the function assumes that the second last layer of the model 
    is a hidden layer from which features are to be extracted.

    When mode is set to 2, the function returns the raw prediction values without applying 
    any threshold for classification.
    """
    
    if (mode==1):
        print('Generating features from final hidden layer\n')
        layer_name = model.layers[len(model.layers)-2].name #-1 for last layer, -2 for second last and so on"
        print('\nGetting outputs from layer: '+layer_name)
        intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(data)
        print('\nObtained feature vector shape: '+str(intermediate_output.shape[1]))
        print('\nShape of returned data: '+str(intermediate_output.shape))
        print('-'*80)
        return pd.DataFrame(intermediate_output)
        
    else:
        print('\nGenerating decison level output\n')
        Y_pred = model.predict(data)
        #print(Y_pred)
        #db=0.50
        #print('Decision boundary: '+str(db))
        #Y_pred = (Y_pred > db)
        #y_pred1 = [int(y) for y in Y_pred]
        y_pred_prob = (Y_pred)
        print('-'*100)
        return pd.DataFrame(y_pred_prob)


# In[9]:


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the given model on test data and provides a comprehensive report of various metrics.

    Parameters:
    - model (tensorflow.keras.Model): The model to be evaluated.
    - X_test (np.array): Test data (features).
    - y_test (np.array): True labels for the test data.

    Returns:
    None. Prints the evaluation metrics and plots the ROC and Precision-Recall curves.

    Plots:
    - ROC Curve: A plot of the True Positive Rate (sensitivity) against the False Positive Rate.
    - Precision-Recall Curve: A plot of Precision against Recall.

    Note:
    Assumes a binary classification problem and a threshold of 0.50 to distinguish between classes.
    """
    
    
    # Predict probabilities and get binary predictions
    Y_pred_prob = model.predict(X_test)
    Y_pred = (Y_pred_prob > 0.50)
    y_pred1 = np.array(Y_pred).ravel()
    r_test_y = y_test.astype(int)

    # Calculate metrics
    mcc = matthews_corrcoef(r_test_y, y_pred1)
    accuracy = accuracy_score(r_test_y, y_pred1)
    f1 = f1_score(r_test_y, y_pred1)
    cm = confusion_matrix(r_test_y, y_pred1)
    sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
    specificity = cm[0][0] / (cm[0][0] + cm[0][1])
    gmean = (sensitivity * specificity) ** 0.5
    roc_auc = roc_auc_score(r_test_y, Y_pred_prob)
    aupr = average_precision_score(r_test_y, Y_pred_prob)
   
    #print the metrics
    print("\n----------RESULTS----------")
    print("Accuracy: ", accuracy)
    print("F1 Score: ", f1)
    print("Matthews Correlation: ", mcc)
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    print("G-mean: ", gmean)
    print("AUROC: ", roc_auc)
    print("AUPR: ", aupr)
    print("Confusion Matrix: \n", cm)
    print('-'*30)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 5))

    # ROC Curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(r_test_y, Y_pred_prob)
    plt.plot(fpr, tpr, label="AUROC = {:.3f}".format(roc_auc))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(r_test_y, Y_pred_prob)
    plt.plot(recall, precision, label="AUPR = {:.3f}".format(aupr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()


# In[10]:


def rename_duplicate_columns(df):
    """
    Function to rename duplicate columns in a pandas DataFrame
    Arguments
    df : pandas DataFrame
    Returns
    df : pandas DataFrame with renamed columns
    """
    cols=[i for i in range(df.shape[1])]
    
    df.columns = cols
    return df


# # Load ProtT5 Features

# In[11]:


test=pd.read_csv('./independent_data/test_data.csv')


# In[12]:


#truncate sequence if the length >= 5000
#this step is optional if you have enough computational resource
test=trim_seq(test)


# In[13]:


#load protT5 FSWE features
test_feature=np.load('./independent_data/test_ProtT5_FSWE_features.npy',allow_pickle=True)


# ### Window-residue Extraction

# In[14]:


window_size=31
#prepare X,y
test_feature_window_protT5=window_embeddings(test,test_feature,window_size)
X_test_protT5=test_feature_window_protT5
X_test_protT5=np.asarray(X_test_protT5).astype('float32')
y_test=test.Target


# # Integer Enocoding for Keras Embedding Layer 

# In[15]:


X_test_embedding = get_input_for_embedding('./independent_data/test_data.fasta')


# # Physicochemical features from FEPS

# In[16]:


test_physico=pd.read_csv('./independent_data/test_data_FEPS_physico.csv')
X_test_physico =test_physico.drop(['uniprotID','label','position'],axis=1)


# In[17]:


# Transform the data into uniform scale
scaler = pickle.load(open('./utils/scaler_physico.sav', 'rb')) #load the saved scaler
X_test_physico_scaled = scaler.transform(X_test_physico)


# 
# # Perform Intermediate Fusion and Prepare Input for Meta-classifer

# In[18]:


#load base models
members = load_all_models(['T5ResConvBiLSTM','EmbedCNN','PhysicoDNN'])

#make sure the layers of base models are freezed
for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            layer.trainable = False

#Get intermediate features
prot_pred_test=get_predictions(members[0],X_test_protT5)
emb_pred_test=get_predictions(members[1],X_test_embedding)
phy_pred_test=get_predictions(members[2],X_test_physico_scaled)

#Intermediate features concatenation
X_stacked_test=pd.concat([prot_pred_test,emb_pred_test,phy_pred_test],axis=1)
y_stacked_test=y_test
print('\nIntermediate-fusion of base models compleleted!!!\n')

# In[19]:


#renaming the columns after fusion (to avoid duplicate column names)
X_stacked_test = rename_duplicate_columns(X_stacked_test)
#again scale the fused data
scaler = pickle.load(open('./utils/scaler_stacked.sav', 'rb'))
X_stacked_test_scaled = scaler.transform(X_stacked_test)


# In[20]:


#load meta-classifer (or LMCrot)
print('Loading LMCrot....')
filename = './models/LMCrot.h5'
model = load_model(filename, custom_objects={"K": K},compile = False)
print('Done!!')

# In[21]:


#evaluate the stacked (or meta-classifer or LMCrot) 
evaluate_model(model, X_stacked_test_scaled, y_stacked_test)

