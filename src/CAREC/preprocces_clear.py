import pandas as pd 
import numpy as np
import csv
import json
import os

MAPPING = {
    'ID': 'doc_id',
    'Excerpt': 'text',
    'Flesch-Reading-Ease': 'FRE.sc',
    'Flesch-Kincaid-Grade-Level': 'FKGL.sc',
    'Automated Readability Index': 'ARI.sc',
    'SMOG Readability': 'SMOG.sc',
    'CAREC': 'CAREC.sc',
    'CAREC_M': 'CAREC_M.sc',
    'CML2RI': 'CML2RI.sc'
}

if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer='data/api/clear-corpus/CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv')
    df = df[MAPPING.keys()]
    df = df.rename(columns=MAPPING)
    
    df.to_csv(path_or_buf='data/api/clear-corpus/all_batches.csv')
    df.to_json(path_or_buf='data/api/clear-corpus/all_batches.json', orient='records', lines=True)