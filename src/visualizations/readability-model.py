
from pathlib import Path
import pandas as pd
import textstat as ts
import spacy
import textdescriptives as td
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import gzip
from tqdm import tqdm
from numba import jit, njit, prange
import ir_datasets as ir
import gzip
import json
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import torch
import pyterrier as pt
import os

data_path='data/visualizations/train-val/'

if __name__ == "__main__":
    
    data_fkgl = pd.read_pickle('data/torch/loss/jina/data_flesch_kincaid_grade.pkl.gzip', compression='gzip')
    data_smog = pd.read_pickle('data/torch/loss/jina/data_smog.pkl.gzip', compression='gzip')
    data_ari = pd.read_pickle('data/torch/loss/jina/data_automated_readability_index.pkl.gzip', compression='gzip')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
    
    ax1.plot(data_fkgl['epochs'], data_fkgl['train_losses'], label='Training FKGL', color='#f7a941')
    ax1.plot(data_fkgl['epochs'], data_fkgl['val_losses'], label='Validation FKGL', color='#0077ae')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Mean Squared Error (MSE)')
    ax1.set_ylim(top=17)
    ax1.legend(loc='upper right')
    ax1.set_title('FKGL')
    
    ax2.plot(data_smog['epochs'], data_smog['train_losses'], label='Training SMOG', color='#f7a941')
    ax2.plot(data_smog['epochs'], data_smog['val_losses'], label='Validation SMOG', color='#0077ae')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Mean Squared Error (MSE)')
    ax2.set_ylim(top=17)
    ax2.legend(loc='upper right')
    ax2.set_title('SMOG')

    ax3.plot(data_ari['epochs'], data_ari['train_losses'], label='Training ARI', color='#f7a941')
    ax3.plot(data_ari['epochs'], data_ari['val_losses'], label='Validation ARI', color='#0077ae')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Mean Squared Error (MSE)')
    ax3.set_ylim(top=17)
    ax3.legend(loc='upper right')
    ax3.set_title('ARI')
   


    

    fig.suptitle('Mean Squared Error (MSE) over 200 Epochs for Training and Validation', fontsize=16)
    

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(data_path+'classic.svg')