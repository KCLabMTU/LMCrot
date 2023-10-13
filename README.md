<div align="center">

# <span style="color:blue;">LMCrot</span> [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20repository&url=https://github.com/KCLabMTU/pLMSNOSite&via=YourTwitterHandle&hashtags=github,transformers,ptmprediction,proteins)

</div>


 <p align="center">
An Interpretable Approach to Predict Crotonylation Modification in Proteins Using Transformer-based Protein Language Model and Residual Network 
 </p>
 
---
<p align="center">
<!---
<img src="images/Screenshot from 2023-06-22 15-32-45.png"/> 
-->
<img src="images/animation.gif"/ alt="Animation"> 
</p>

<p align="center">
<a href="https://www.python.org/"><img alt="python" src="https://img.shields.io/badge/Python-3.9.7-blue.svg"/></a>
<a href="https://www.tensorflow.org/"><img alt="tensorflow" src="https://img.shields.io/badge/TensorFlow-2.9.1-orange.svg"/></a>
<a href="https://keras.io/"><img alt="Keras" src="https://img.shields.io/badge/Keras-2.9.0-red.svg"/></a>
<a href="https://huggingface.co/transformers/"><img alt="Transformers" src="https://img.shields.io/badge/Transformers-4.18.0-yellow.svg"/></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.11.0-orange.svg"/></a>
<a href="https://biopython.org/"><img alt="Bio" src="https://img.shields.io/badge/Bio-1.5.2-brightgreen.svg"/></a>
<a href="https://scikit-learn.org/"><img alt="scikit_learn" src="https://img.shields.io/badge/scikit_learn-1.2.0-blue.svg"/></a>
<a href="https://matplotlib.org/"><img alt="matplotlib" src="https://img.shields.io/badge/matplotlib-3.5.1-blueviolet.svg"/></a>
<a href="https://numpy.org/"><img alt="numpy" src="https://img.shields.io/badge/numpy-1.23.5-red.svg"/></a>
<a href="https://pandas.pydata.org/"><img alt="pandas" src="https://img.shields.io/badge/pandas-1.5.0-yellow.svg"/></a>
<a href="https://docs.python-requests.org/en/latest/"><img alt="requests" src="https://img.shields.io/badge/requests-2.27.1-green.svg"/></a>
<a href="https://seaborn.pydata.org/"><img alt="seaborn" src="https://img.shields.io/badge/seaborn-0.11.2-lightgrey.svg"/></a>
<a href="https://tqdm.github.io/"><img alt="tqdm" src="https://img.shields.io/badge/tqdm-4.63.0-blue.svg"/></a>
<a href="https://xgboost.readthedocs.io/en/latest/"><img alt="xgboost" src="https://img.shields.io/badge/xgboost-1.5.0-purple.svg"/></a>
<a href="https://github.com/KCLabMTU/pLMSNOSite/commits/main"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/KCLabMTU/pLMSNOSite.svg?style=flat&color=blue"></a>
<a href="https://github.com/KCLabMTU/pLMSNOSite/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/KCLabMTU/pLMSNOSite.svg?style=flat&color=blue"></a>
<a href="https://github.com/KCLabMTU/pLMSNOSite/pulls"><img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/KCLabMTU/pLMSNOSite.svg?style=flat&color=blue"></a>


 
</p>

##  About
pLMSNOSite is a robust predictor of S-nitrosylation modification sites in protein sequences. It employs an intermediate-fusion-based stacked generalization approach to harness the representational power of global contextual embeddings obtained from a transformer protein language model called [`ProtT5-XL-UniRef50`](https://github.com/agemagician/ProtTrans) combined with local contextual embeddings from the supervised word embedding layer.

#### About ProtT5-XL-UniRef50
[`ProtT5-XL-UniRef50`](https://github.com/agemagician/ProtTrans) is a transformer-based protein language model that was developed by Rostlab. This model uses Google's T5 (Text-to-Text Transfer Transformer) architecture. Using the Masked Language Modelling (MLM) objective, ProtT5 was trained on the UniRef50 dataset (consisting of 45 million protein sequences) in a self-supervised fashion. This comprehensive training allows the model to effectively capture and understand the context within protein sequences, proving valuable for tasks like predicting PTM sites. More details about ProtT5 are as follows:
| Dataset | No. of Layers | Hidden Layer Size | Intermediate Size | No. of Heads | Dropout | Target Length | Masking Probability | Local Batch Size | Global Batch Size | Optimizer | Learning Rate | Weight Decay | Training Steps | Warm-up Steps | Mixed Precision | No. of Parameters | System | No. of Nodes | No. of GPUs/TPUs |
|---------|---------------|-------------------|-------------------|--------------|---------|---------------|---------------------|------------------|-------------------|-----------|---------------|--------------|----------------|---------------|-----------------|------------------|--------|--------------|-----------------|
| UniRef50 | 24 | 1024 | 65536 | 16 | 0.1 | 512 | 15% | 84 | 4096 | AdaFactor | 0.01 | 0.0 | 400K/400K | 40K/40K | None | 3B | TPU Pod | 32 | 1024 |

Note: Info. in the table adopted from "ProtTrans: Towards Cracking the Language of Life’s Code Through Self-Supervised Learning," by A. Elnaggar et al., 2023, *IEEE Transactions on Pattern Analysis & Machine Intelligence*, 14(8).

## Webserver  :globe_with_meridians:

You can access the webserver of LMCrot at [kcdukkalab.org/pLMSNOSite/](http://kcdukkalab.org/pLMSNOSite/).

``

## Getting Started  :rocket: 

To get a local copy of the repository, you can either clone it or download it directly from GitHub.

### Clone the Repository

If you have Git installed on your system, you can clone the repository by running the following command in your terminal:

```shell
git clone git@github.com:KCLabMTU/LMCrot.git
```
### Download the Repository
Alternatively, if you don't have Git or prefer not to use it, you can download the repository directly from GitHub. Click [here](https://github.com/KCLabMTU/LMCrot/archive/refs/heads/main.zip) to download the repository as a zip file.

Note: In the 'Download the Repository' section, the link provided is a direct download link to the repository's `main` branch as a zip file. This may differ if your repository's default branch is named differently.

## Install Libraries

Python version: `3.9.7`

To install the required libraries, run the following command:

```shell
pip install -r requirements.txt
```

Required libraries and versions: 
<code>
Bio==1.5.2
keras==2.9.0
matplotlib==3.5.1
numpy==1.23.5
pandas==1.5.0
requests==2.27.1
scikit_learn==1.2.0
seaborn==0.11.2
tensorflow==2.9.1
torch==1.11.0
tqdm==4.63.0
transformers==4.18.0
xgboost==1.5.0
</code>

## Install Transformers
```shell
pip install -q SentencePiece transformers
```
## Evaluate LMCrot on Independent Test Set
To evaluate our model on the independent test set, we have already placed the test sequences and corresponding ProtT5 features in `data/test/` folder. After installing all the requirements, run the following command:
<br>
```shell
 python evaluate_model.py
```

## Predict Crotonylatiom modification in your own sequence
1. Place your FASTA file in the `input/sequence.fasta` directory.
2. Run the following command:
   ```shell
   python3 predict.py
   ```
3. Find the results in the current directory.

## Notes  :memo: 
1. The prediction runtime directly depends on the length of the input sequence. Longer sequences require more time for ProtT5 to generate feature vectors, and consequently, more time is needed for prediction.
2. In order to tailor the system to your specific requirements, we have ensured that modifying the decision threshold cut-off value is simple and straightforward. Here's what you need to do:
   - Open the `predict.py` file 
     - Navigate to line `171`
     - You'll find the current cut-off value is set at `0.5`
     - Adjust this to any preferred cut-off value 

   By following these simple steps, you can easily customize the decision threshold cut-off value to better meet the needs of your project.


## Funding 
<p>
  <a href="https://www.nsf.gov/">
    <img src="images/NSF_Official_logo.svg" alt="NSF Logo" width="110" height="110" style="margin-right: 20px;">
  </a>
</p>




## Contact  :mailbox: 
Should you have any inquiries related to this project, please feel free to reach out via email. Kindly CC all of the following recipients in your communication for a swift response:

- Main Contact: [dbkc@mtu.edu](mailto:dbkc@mtu.edu)
- CC: [ppratyush@mtu.edu](mailto:ppratyush@mtu.edu)

We look forward to addressing your queries and concerns.
