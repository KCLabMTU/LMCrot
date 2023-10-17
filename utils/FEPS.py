import pandas as pd
import numpy as np
from propy import CTD
#pip install propy3
import re, sys, os
from collections import Counter


#Xs1
aadf = pd.read_csv("./utils/aa.csv", sep=",")
aadict = aadf.set_index('residue').T.to_dict('list')
def features1(seq):
     mass   = []
     atom   = []
     volume = []
     hindex = []
     #for aa in seq:
     for i in range(len(seq)):
        k = (len(seq)+1)/2
        if i != k and seq[i] != "X": 
          mass.append(aadict[seq[i]][0])
          atom.append(aadict[seq[i]][1])
          volume.append(aadict[seq[i]][2])
          hindex.append(aadict[seq[i]][3])
     
     features = np.concatenate([mass, atom, volume, hindex])
     return features

"""
Computes AAIndex
"""

#The AAIndex
TSAJ990101={'A': 89.3, 'R': 190.3, 'N': 122.4, 'D': 114.4, \
            'C': 102.5, 'E': 138.8, 'Q': 146.9, 'G': 63.8, \
            'H': 157.5, 'I': 163.0, 'L': 163.1, 'K': 165.1, \
            'M': 165.8, 'F': 190.8, 'P': 121.6, 'S': 94.2, \
            'T': 119.6, 'W': 226.4, 'Y': 194.6, 'V': 138.2, 'O':0.0, 'U':0.0}
MAXF760101={'A': 1.43, 'R': 1.18, 'N': 0.64, 'D': 0.92, \
            'C': 0.94, 'E': 1.67, 'Q': 1.22, 'G': 0.46, \
            'H': 0.98, 'I': 1.04, 'L': 1.36, 'K': 1.27, \
            'M': 1.53, 'F': 1.19, 'P': 0.49, 'S': 0.7, \
            'T': 0.78, 'W': 1.01, 'Y': 0.69, 'V': 0.98, 'O':0.0, 'U':0.0}
NAKH920108={'A': 9.36, 'R': 0.27, 'N': 2.31, 'D': 0.94, \
            'C': 2.56, 'E': 0.94, 'Q': 1.14, 'G': 6.17, \
            'H': 0.47, 'I': 13.73, 'L': 16.64, 'K': 0.58, \
            'M': 3.93, 'F': 10.99, 'P': 1.96, 'S': 5.58, \
            'T': 4.68, 'W': 2.2, 'Y': 3.13, 'V': 12.43, 'O':0.0, 'U':0.0}
BLAM930101={'A': 0.96, 'R': 0.77, 'N': 0.39, 'D': 0.42, \
            'C': 0.42, 'E': 0.53, 'Q': 0.8, 'G': 0.0, \
            'H': 0.57, 'I': 0.84, 'L': 0.92, 'K': 0.73, \
            'M': 0.86, 'F': 0.59, 'P': -2.5, 'S': 0.53, \
            'T': 0.54, 'W': 0.58, 'Y': 0.72, 'V': 0.63, 'O':0.0, 'U':0.0}
BIOV880101={'A': 16.0, 'R': -70.0, 'N': -74.0, 'D': -78.0, \
            'C': 168.0, 'E': -106.0, 'Q': -73.0, 'G': -13.0, \
            'H': 50.0, 'I': 151.0, 'L': 145.0, 'K': -141.0, \
            'M': 124.0, 'F': 189.0, 'P': -20.0, 'S': -70.0, \
            'T': -38.0, 'W': 145.0, 'Y': 53.0, 'V': 123.0, 'O':0.0, 'U':0.0}
CEDJ970104={'A': 7.9, 'R': 4.9, 'N': 4.0, 'D': 5.5, 'C': 1.9, \
            'E': 7.1, 'Q': 4.4, 'G': 7.1, 'H': 2.1, 'I': 5.2,\
            'L': 8.6, 'K': 6.7, 'M': 2.4, 'F': 3.9, 'P': 5.3, \
            'S': 6.6, 'T': 5.3, 'W': 1.2, 'Y': 3.1, 'V': 6.8, 'O':0.0, 'U':0.0}
NOZY710101={'A': 0.5, 'R': 0.0, 'N': 0.0, 'D': 0.0, 'C': 0.0, \
            'E': 0.0, 'Q': 0.0, 'G': 0.0, 'H': 0.5, 'I': 1.8, \
            'L': 1.8, 'K': 0.0, 'M': 1.3, 'F': 2.5, 'P': 0.0, \
            'S': 0.0, 'T': 0.4, 'W': 3.4, 'Y': 2.3, 'V': 1.5, 'O':0.0, 'U':0.0}
KLEP840101={'A': 0.0, 'R': 1.0, 'N': 0.0, 'D': -1.0, 'C': 0.0, \
            'E': -1.0, 'Q': 0.0, 'G': 0.0, 'H': 0.0, 'I': 0.0, \
            'L': 0.0, 'K': 1.0, 'M': 0.0, 'F': 0.0, 'P': 0.0, \
            'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0, 'O':0.0, 'U':0.0}
NAKH900109={'A': 9.25, 'R': 3.96, 'N': 3.71, 'D': 3.89, 'C': 1.07, \
            'E': 4.8, 'Q': 3.17, 'G': 8.51, 'H': 1.88, 'I': 6.47, \
            'L': 10.94, 'K': 3.5, 'M': 3.14, 'F': 6.36, 'P': 4.36, \
            'S': 6.26, 'T': 5.66, 'W': 2.22, 'Y': 3.28, 'V': 7.55, 'O':0.0, 'U':0.0}
LIFS790101={'A': 0.92, 'R': 0.93, 'N': 0.6, 'D': 0.48, 'C': 1.16, \
            'E': 0.61, 'Q': 0.95, 'G': 0.61, 'H': 0.93, 'I': 1.81,\
            'L': 1.3, 'K': 0.7, 'M': 1.19, 'F': 1.25, 'P': 0.4, \
            'S': 0.82, 'T': 1.12, 'W': 1.54, 'Y': 1.53, 'V': 1.81, 'O':0.0, 'U':0.0}
HUTJ700103={'A': 154.33, 'R': 341.01, 'N': 207.9, 'D': 194.91, \
            'C': 219.79, 'E': 223.16, 'Q': 235.51, 'G': 127.9, \
            'H': 242.54, 'I': 233.21, 'L': 232.3, 'K': 300.46, \
            'M': 202.65, 'F': 204.74, 'P': 179.93, 'S': 174.06, \
            'T': 205.8, 'W': 237.01, 'Y': 229.15, 'V': 207.6, 'O':0.0, 'U':0.0}
MIYS990104={'A': -0.04, 'R': 0.07, 'N': 0.13, 'D': 0.19, 'C': -0.38,\
            'E': 0.23, 'Q': 0.14, 'G': 0.09, 'H': -0.04, 'I': -0.34,\
            'L': -0.37, 'K': 0.33, 'M': -0.3, 'F': -0.38, 'P': 0.19, \
            'S': 0.12, 'T': 0.03, 'W': -0.33, 'Y': -0.29, 'V': -0.29, 'O':0.0, 'U':0.0}

def features2(seq):
  features = {}
  counter = 0
  alldf = pd.DataFrame()
  for i in range(len(seq)):
     if seq[i] !="X":
       features['TSAJ'+str(i+1)] = TSAJ990101[seq[i]]
       features['MAXF'+str(i+1)] = MAXF760101[seq[i]]
       features['NAKH'+str(i+1)] = NAKH920108[seq[i]]
       features['BLAM'+str(i+1)] = BLAM930101[seq[i]]
       features['BIOV'+str(i+1)] = BIOV880101[seq[i]]
       features['CEDJ'+str(i+1)] = CEDJ970104[seq[i]]
       features['NOZY'+str(i+1)] = NOZY710101[seq[i]]
       features['KLEP'+str(i+1)] = KLEP840101[seq[i]]
       features['NAKH'+str(i+1)] = NAKH900109[seq[i]] 
       features['LIFS'+str(i+1)] = LIFS790101[seq[i]]
       features['HUTJ'+str(i+1)] = HUTJ700103[seq[i]]
       features['MIYS'+str(i+1)] = MIYS990104[seq[i]]
  df_dictionary = pd.DataFrame([features])
  features = df_dictionary.to_numpy()[0]
  return features

"""
Computes HAAC
"""


def features3(seq):
     features = {}
     alldf = pd.DataFrame()
     NpolAlip  = 0
     Aromatic  = 0
     PolUncha  = 0  
     PosiChar  = 0
     #LL = (len(seq)-1)/2
     for i in range(len(seq)):
        if seq[i] in ["G","A", "L","V","M","I"]:
            NpolAlip = NpolAlip + 1
        elif seq[i] in ["F", "Y", "W"]:
            Aromatic = Aromatic + 1
        elif seq[i] in ["S", "T","C", "P", "N", "Q"]:
            PolUncha = PolUncha + 1
        elif seq[i] in ["E", "D"]:
            PosiChar = PosiChar + 1

     features['NpolAlip'] = NpolAlip
     features['Aromatic'] = Aromatic
     features['PolUncha'] = PolUncha
     features['PosiChar'] = PosiChar
     df_dictionary = pd.DataFrame([features])
     #alldf = pd.concat([alldf, df_dictionary], ignore_index=True)
     features_np = df_dictionary.to_numpy()[0]
     return features_np
     


"""
Computes CTD
"""
def getCTD(proseq):
       '''
       * CTD: Composition Translation Distribution
       * Calculate all CTD descriptors based seven different properties of AADs. 
       * 21 feature / sequence
       * Composition
       '''
       return CTD.CalculateCTD(proseq)

def features4(seq):
   features = {}
   counter = 0
   alldf = pd.DataFrame()
   features = getCTD(seq)   
   df_dictionary = pd.DataFrame([features])
   #alldf = pd.concat([alldf, df_dictionary], ignore_index=True)
   features = df_dictionary.to_numpy()[0]
   return features
import numpy as np
def features5(sequence):  #ZSCALE
	zscale = {
		'A': [0.24,  -2.32,  0.60, -0.14,  1.30], # A
		'C': [0.84,  -1.67,  3.71,  0.18, -2.65], # C
		'D': [3.98,   0.93,  1.93, -2.46,  0.75], # D
		'E': [3.11,   0.26, -0.11, -0.34, -0.25], # E
		'F': [-4.22,  1.94,  1.06,  0.54, -0.62], # F
		'G': [2.05,  -4.06,  0.36, -0.82, -0.38], # G
		'H': [2.47,   1.95,  0.26,  3.90,  0.09], # H
		'I': [-3.89, -1.73, -1.71, -0.84,  0.26], # I
		'K': [2.29,   0.89, -2.49,  1.49,  0.31], # K
		'L': [-4.28, -1.30, -1.49, -0.72,  0.84], # L
		'M': [-2.85, -0.22,  0.47,  1.94, -0.98], # M
		'N': [3.05,   1.62,  1.04, -1.15,  1.61], # N
		'P': [-1.66,  0.27,  1.84,  0.70,  2.00], # P
		'Q': [1.75,   0.50, -1.44, -1.34,  0.66], # Q
		'R': [3.52,   2.50, -3.50,  1.99, -0.17], # R
		'S': [2.39,  -1.07,  1.15, -1.39,  0.67], # S
		'T': [0.75,  -2.18, -1.12, -1.46, -0.40], # T
		'V': [-2.59, -2.64, -1.54, -0.85, -0.02], # V
		'W': [-4.36,  3.94,  0.59,  3.44, -1.59], # W
		'Y': [-2.54,  2.44,  0.43,  0.04, -1.47], # Y
		'-': [0.00,   0.00,  0.00,  0.00,  0.00], # -
	}
	features = []
	for aa in sequence:
	    code = zscale[aa]
	    features.append(code)
	return np.array(features).reshape(155)


def features6(sequence, window=5, **kw):  #EAAC
     kw = {'order': 'ACDEFGHIKLMNPQRSTVWY'}
     AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
     #AA = 'ARNDCQEGHILKMFPSTWYV'
     encodings = []
     code = []
     for j in range(len(sequence)):
         if j < len(sequence) and j + window <= len(sequence):
             count = Counter(re.sub('-', '', sequence[j:j+window]))
             for key in count:
                count[key] = count[key] / len(re.sub('-', '', sequence[j:j+window]))
             for aa in AA:
                code.append(count[aa])
     encodings.append(code)
     return np.array(encodings).reshape(540)

"""
Computes Hydrophobicity polar
"""
def features7(seq):
   features = {}
   counter = 0
   alldf = pd.DataFrame()
   hydrophobic  = 0
   polar        = 0
   PolUncha  = 0  
   PosiChar  = 0
   #LL = (len(seq)-1)/2
   for i in range(len(seq)):
        if seq[i] in ["G","A", "V","L","I","P", "F", "M", "W"]:
            hydrophobic = hydrophobic + 1
        elif seq[i] in ["T", "C", "N", "Q", "Y"]:
            polar = polar + 1

   features['hydrophobic'] = hydrophobic
   features['polar'] = polar
   df_dictionary = pd.DataFrame([features])
   features = df_dictionary.to_numpy()[0]
   return features



"""
Computes OS
"""
def features8(seq):
  features = {}
  counter = 0
  alldf = pd.DataFrame()
  polar        = 0
  positive     = 0
  negative     = 0  
  charged      = 0
  hydrophobic  = 0
  aliphatic    = 0
  aromatic     = 0
  small        = 0
  tiny         = 0
  proline      = 0

  #LL = (len(seq)-1)/2
  for i in range(len(seq)):
     if seq[i] in ["N","Q", "S","D","E","C", "T", "K", "R", "H", "Y", "W"]:
         polar = polar + 1
     elif seq[i] in ["K", "H", "R"]:
         positive = positive + 1
     elif seq[i] in ["D", "E"]:
         negative = negative + 1
     elif seq[i] in ["K", "H", "R", "D", "E"]:
         charged = charged + 1
     elif seq[i] in ["A", "G", "C", "T", "I", "V", "L", "K", "H", "F", "W", "Y", "M"]:
         hydrophobic = hydrophobic + 1
     elif seq[i] in ["I", "V", "L"]:
         aliphatic = aliphatic + 1
     elif seq[i] in ["F", "Y", "W", "H"]:
         aromatic = aromatic + 1
     elif seq[i] in ["P", "N", "D", "T", "C", "A", "G", "S", "V"]:
         small = small + 1
     elif seq[i] in ["A", "S", "G", "C"]:
         tiny = tiny + 1
     elif seq[i] == "P":
         proline = proline + 1

  #features['polar']        = polar
  features['positive']     = positive
  features['negative']     = negative
  #features['charged']      = charged
  #features['hydrophobic']  = hydrophobic
  #features['aliphatic']    = aliphatic
  #features['aromatic']     = aromatic
  #features['small']        = small
  features['tiny']         = tiny
  features['proline']      = proline

  df_dictionary = pd.DataFrame([features])
  features = df_dictionary.to_numpy()[0]
  return features


"""
Computes ACH - JA
"""
#The AAIndex
TSAJ990101={'A': 89.3, 'R': 190.3, 'N': 122.4, 'D': 114.4, \
            'C': 102.5, 'E': 138.8, 'Q': 146.9, 'G': 63.8, \
            'H': 157.5, 'I': 163.0, 'L': 163.1, 'K': 165.1, \
            'M': 165.8, 'F': 190.8, 'P': 121.6, 'S': 94.2, \
            'T': 119.6, 'W': 226.4, 'Y': 194.6, 'V': 138.2}
MAXF760101={'A': 1.43, 'R': 1.18, 'N': 0.64, 'D': 0.92, \
            'C': 0.94, 'E': 1.67, 'Q': 1.22, 'G': 0.46, \
            'H': 0.98, 'I': 1.04, 'L': 1.36, 'K': 1.27, \
            'M': 1.53, 'F': 1.19, 'P': 0.49, 'S': 0.7, \
            'T': 0.78, 'W': 1.01, 'Y': 0.69, 'V': 0.98}
NAKH920108={'A': 9.36, 'R': 0.27, 'N': 2.31, 'D': 0.94, \
            'C': 2.56, 'E': 0.94, 'Q': 1.14, 'G': 6.17, \
            'H': 0.47, 'I': 13.73, 'L': 16.64, 'K': 0.58, \
            'M': 3.93, 'F': 10.99, 'P': 1.96, 'S': 5.58, \
            'T': 4.68, 'W': 2.2, 'Y': 3.13, 'V': 12.43}
BLAM930101={'A': 0.96, 'R': 0.77, 'N': 0.39, 'D': 0.42, \
            'C': 0.42, 'E': 0.53, 'Q': 0.8, 'G': 0.0, \
            'H': 0.57, 'I': 0.84, 'L': 0.92, 'K': 0.73, \
            'M': 0.86, 'F': 0.59, 'P': -2.5, 'S': 0.53, \
            'T': 0.54, 'W': 0.58, 'Y': 0.72, 'V': 0.63}
BIOV880101={'A': 16.0, 'R': -70.0, 'N': -74.0, 'D': -78.0, \
            'C': 168.0, 'E': -106.0, 'Q': -73.0, 'G': -13.0, \
            'H': 50.0, 'I': 151.0, 'L': 145.0, 'K': -141.0, \
            'M': 124.0, 'F': 189.0, 'P': -20.0, 'S': -70.0, \
            'T': -38.0, 'W': 145.0, 'Y': 53.0, 'V': 123.0}
CEDJ970104={'A': 7.9, 'R': 4.9, 'N': 4.0, 'D': 5.5, 'C': 1.9, \
            'E': 7.1, 'Q': 4.4, 'G': 7.1, 'H': 2.1, 'I': 5.2,\
            'L': 8.6, 'K': 6.7, 'M': 2.4, 'F': 3.9, 'P': 5.3, \
            'S': 6.6, 'T': 5.3, 'W': 1.2, 'Y': 3.1, 'V': 6.8}
NOZY710101={'A': 0.5, 'R': 0.0, 'N': 0.0, 'D': 0.0, 'C': 0.0, \
            'E': 0.0, 'Q': 0.0, 'G': 0.0, 'H': 0.5, 'I': 1.8, \
            'L': 1.8, 'K': 0.0, 'M': 1.3, 'F': 2.5, 'P': 0.0, \
            'S': 0.0, 'T': 0.4, 'W': 3.4, 'Y': 2.3, 'V': 1.5}
KLEP840101={'A': 0.0, 'R': 1.0, 'N': 0.0, 'D': -1.0, 'C': 0.0, \
            'E': -1.0, 'Q': 0.0, 'G': 0.0, 'H': 0.0, 'I': 0.0, \
            'L': 0.0, 'K': 1.0, 'M': 0.0, 'F': 0.0, 'P': 0.0, \
            'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0}
NAKH900109={'A': 9.25, 'R': 3.96, 'N': 3.71, 'D': 3.89, 'C': 1.07, \
            'E': 4.8, 'Q': 3.17, 'G': 8.51, 'H': 1.88, 'I': 6.47, \
            'L': 10.94, 'K': 3.5, 'M': 3.14, 'F': 6.36, 'P': 4.36, \
            'S': 6.26, 'T': 5.66, 'W': 2.22, 'Y': 3.28, 'V': 7.55}
LIFS790101={'A': 0.92, 'R': 0.93, 'N': 0.6, 'D': 0.48, 'C': 1.16, \
            'E': 0.61, 'Q': 0.95, 'G': 0.61, 'H': 0.93, 'I': 1.81,\
            'L': 1.3, 'K': 0.7, 'M': 1.19, 'F': 1.25, 'P': 0.4, \
            'S': 0.82, 'T': 1.12, 'W': 1.54, 'Y': 1.53, 'V': 1.81}
HUTJ700103={'A': 154.33, 'R': 341.01, 'N': 207.9, 'D': 194.91, \
            'C': 219.79, 'E': 223.16, 'Q': 235.51, 'G': 127.9, \
            'H': 242.54, 'I': 233.21, 'L': 232.3, 'K': 300.46, \
            'M': 202.65, 'F': 204.74, 'P': 179.93, 'S': 174.06, \
            'T': 205.8, 'W': 237.01, 'Y': 229.15, 'V': 207.6}
MIYS990104={'A': -0.04, 'R': 0.07, 'N': 0.13, 'D': 0.19, 'C': -0.38,\
            'E': 0.23, 'Q': 0.14, 'G': 0.09, 'H': -0.04, 'I': -0.34,\
            'L': -0.37, 'K': 0.33, 'M': -0.3, 'F': -0.38, 'P': 0.19, \
            'S': 0.12, 'T': 0.03, 'W': -0.33, 'Y': -0.29, 'V': -0.29}

BaMe={'A': 0.75, 'R': -0.02, 'N': -0.16, 'D': -0.50, 'C': 2.60,\
            'E': -0.54, 'Q': -0.11, 'G': 0.0, 'H': 0.57, 'I': 2.19,\
            'L': 1.97, 'K': -0.90, 'M': 1.22, 'F': 1.92, 'P': 0.72, \
            'S': 0.11, 'T': 0.47, 'W': 1.51, 'Y': 1.36, 'V': 1.8}

BLMO={'A': 0.37, 'R': -1.52, 'N': -0.79, 'D': -1.43, 'C': 0.55,\
            'E': -1.4, 'Q': -0.76, 'G': 0.0, 'H': -0.10, 'I': 1.34,\
            'L': 1.34, 'K': -0.67, 'M': 0.73, 'F': 1.52, 'P': 0.64, \
            'S': -0.43, 'T': -0.15, 'W': 1.16, 'Y': 1.16, 'V': 1.0}


EI={'A': 0.15, 'R': -3.09, 'N': -1.29, 'D': -1.42, 'C': -0.19,\
            'E': -0.54, 'Q': -1.37, 'G': 0.0, 'H': -0.90, 'I': 0.92,\
            'L': 0.60, 'K': -2.03, 'M': 0.16, 'F': 1.10, 'P': -0.41, \
            'S': -0.68, 'T': -0.55, 'W': 0.34, 'Y': -0.23, 'V': 0.61}

KYDO={'A': 0.76, 'R': -1.41, 'N': -1.06, 'D': -1.06, 'C': 1.0,\
            'E': -1.06, 'Q': -1.06, 'G': 0.0, 'H': -0.96, 'I': 1.68,\
            'L': 1.44, 'K': -1.20, 'M': 0.79, 'F': 1.10, 'P': -0.41, \
            'S': -0.14, 'T': -0.10, 'W': -0.17, 'Y': -0.31, 'V': 1.58}




ME={'A': 0.07, 'R': 0.11, 'N': 0.11, 'D': -1.08, 'C': -0.90,\
            'E': -2.23, 'Q': -0.63, 'G': 0.0, 'H': -0.46, 'I': 1.83,\
            'L': 1.16, 'K': 0.01, 'M': 0.63, 'F': 1.74, 'P': 0.80, \
            'S': 0.16, 'T': 0.36, 'W': 1.96, 'Y': 0.80, 'V': 0.36}

RO={'A': 0.18, 'R': -0.71, 'N': -0.80, 'D': 0.89, 'C': 1.69,\
            'E': -0.89, 'Q': -0.89, 'G': 0.00, 'H': 0.53, 'I': 1.42,\
            'L': 1.16, 'K': -1.78, 'M': 1.16, 'F': 1.42, 'P': -0.71, \
            'S': -0.53, 'T': -0.18, 'W': 1.16, 'Y': 0.36, 'V': 1.25}

WIWH={'A': -0.20, 'R': -0.41, 'N': -0.51, 'D': -1.53, 'C': 0.31,\
            'E': -2.51, 'Q': -0.71, 'G': 0, 'H': -0.20, 'I': 0.35,\
            'L': 0.71, 'K': -0.59, 'M': 0.30, 'F': 1.43, 'P': -0.55, \
            'S': -0.15, 'T': -0.16, 'W': 2.33, 'Y': 1.19, 'V': -0.08}

JA={'A': 0.06, 'R': -0.32, 'N': 0.13, 'D': -0.43, 'C': 0.55,\
            'E': -0.72, 'Q': 0.46, 'G': 0, 'H': 0.03, 'I': 1.54,\
            'L': 1.54, 'K': -1.07, 'M': 0.49, 'F': 2.48, 'P': 0.44, \
            'S': 0.07, 'T': 0.16, 'W': 2.81, 'Y': 1.84, 'V': 0.97, 'O': 0, 'U':0}

MI={'A': 0.40, 'R': -0.15, 'N': -0.37, 'D': -0.43, 'C': 1.65,\
            'E': -0.40, 'Q': -0.29, 'G': 0, 'H': 0.30, 'I': 2.08,\
            'L': 1.91, 'K': -0.74, 'M': 2.14, 'F': 2.18, 'P': -0.29, \
            'S': -0.19, 'T': 0, 'W': 1.52, 'Y': 0.68, 'V': 1.51, 'O': 0, 'U':0}



"""
EI={'A': , 'R': , 'N': , 'D': , 'C': ,\
            'E': , 'Q': , 'G': , 'H': , 'I': ,\
            'L': , 'K': , 'M': , 'F': , 'P': , \
            'S': , 'T': , 'W': , 'Y': , 'V': }
"""
def features9(seq):
    features = {}
    counter = 0
    alldf = pd.DataFrame()
    #seq = "SNLEEKQRSLQ"
    j = 0
    m = int((len(seq)-1)/2)
    for i in range(m):
       sumi = 0
       avg  = 0
       for j in range(i,len(seq)):
         if j != m:
             if j < len(seq)-i:
                 if seq[j] != "X":  
                   aa  = seq[j]
                   aai = JA[aa]
                   sumi = sumi + aai 
             else:
                 break
       avg = sumi/(len(seq) - 2*i -1)  
       avg = round(avg,3)
       features['JA'+str(i+1)] = avg    
    df_dictionary = pd.DataFrame([features])
    return df_dictionary.to_numpy()[0]



"""
Computes ACH - MI
"""

def features10(seq):
    features = {}
    counter = 0
    alldf = pd.DataFrame()
    #seq = "SNLEEKQRSLQ"
    j = 0
    m = int((len(seq)-1)/2)
    for i in range(m):
       sumi = 0
       avg  = 0
       for j in range(i,len(seq)):
         if j != m:
             if j < len(seq)-i:
                 if seq[j] != "X":
                   aa  = seq[j]
                   aai = JA[aa]
                   sumi = sumi + aai
             else:
                 break
       avg = sumi/(len(seq) - 2*i -1)
       avg = round(avg,3)
       features['MI'+str(i+1)] = avg
    df_dictionary = pd.DataFrame([features])
    return df_dictionary.to_numpy()[0]


def get_FEPS_features(peptide):
  X1 = features1(peptide)
  #print(X1.shape)
  X2 = features2(peptide)
  #print(X2.shape)
  X3 = features3(peptide)
  #print(X3.shape)
  X4 = features4(peptide)
  #print(X4.shape)
  X5 = features5(peptide)
  #print(X5.shape)
  X6 = features6(peptide)
  #print(X6.shape)
  X7 = features7(peptide)
  #print(X7.shape)
  X8 = features8(peptide)
  #print(X8.shape)
  X9 = features9(peptide)
  #print(X9.shape)
  X10 = features10(peptide)
  #print(X10.shape)
  features = np.hstack((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10))
  return features


