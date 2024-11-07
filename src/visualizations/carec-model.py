
from pathlib import Path
import pandas as pd
import textstat as ts
import spacy
import textdescriptives as td
import matplotlib.pyplot as plt
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
    data =pd.read_pickle('data/torch/loss/jina/data_CAREC.sc.pkl.gzip', compression='gzip')
    #val =pd.read_pickle('data/torch/mae/jina/val_smog.pkl.gzip', compression='gzip')
    print(data)
    
    # Plotting the accuracy (MAE)
    plt.plot(data['epochs'], data['train_losses'],label='Training MSE', color='#f7a941' )
    plt.plot(data['epochs'], data['val_losses'], label='Validation MSE', color='#0077ae')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    #plt.legend()
    plt.title('Mean Squared Error (MSE) over 200 Epochs for Training and Validation')
    plt.savefig(data_path+'carec.svg')
    