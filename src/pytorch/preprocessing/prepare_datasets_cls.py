import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch as nn
import gzip
from sklearn.utils import shuffle

cwd = os.getcwd()

DATASETS = ['beir/arguana',
            'beir/nfcorpus',
            'beir/scifact', 
            'clear/corpus']

SCORES = ["flesch_reading_ease",
          "flesch_kincaid_grade",
          "smog",
          "gunning_fog",
          "automated_readability_index",
          "coleman_liau_index",
          "lix","rix",]

ARTE_SCORES=["FRE.sc",
          "FKGL.sc",
          "ARI.sc",
          "SMOG.sc",
          "CAREC.sc",
          "CAREC_M.sc",
          "CML2RI.sc"]

transformer = 'jina'

if __name__ == '__main__':
    embedding_list = []
    spacy_scores_list = []
    arte_scores_list= []
    for dsname in DATASETS:
        embeddings_path = os.path.join(cwd, 'data/transformer', dsname.replace('/', '-'), transformer+ '.pkl.gzip')
        spacy_scores_path = os.path.join(cwd, 'data/spacy/all_tokens', dsname.replace('/','-')+'.jsonl')
        arte_scores_path = os.path.join(cwd, 'data/api', dsname.replace('/', '-'), 'all_batches.json')
        embeddings = pd.read_pickle(embeddings_path, compression='gzip')

        # for column in embeddings.columns:
            # if column != 'doc_id':
            #     embeddings[column].apply(lambda x: x.item())
        embeddings = embeddings[embeddings.columns.drop(list(embeddings.filter(regex='sep_')))]
        spacy_scores = pd.read_json(spacy_scores_path, lines=True).rename(columns={'docno': 'doc_id'})
        arte_scores = pd.read_json(arte_scores_path, lines=True)
        embedding_list.append(embeddings)
        spacy_scores_list.append(spacy_scores)
        arte_scores_list.append(arte_scores)

    
    embeddings = pd.concat(embedding_list)
    embeddings['doc_id'] = embeddings['doc_id'].astype(str)
    spacy_scores = pd.concat(spacy_scores_list)
    spacy_scores['doc_id'] = spacy_scores['doc_id'].astype(str)
    arte_scores = pd.concat(arte_scores_list)
    arte_scores['doc_id'] = arte_scores['doc_id'].astype(str)
        
    for score in SCORES:
        
        spacy_score = spacy_scores.filter(['doc_id',score])
        merged_df = pd.merge(embeddings, spacy_score, on='doc_id', how='outer')
        
        merged_df = shuffle(merged_df)
        print(merged_df.shape)
        '''
        PROBLEM scifact: doc_id ist in embeddings und in spacy score unterschiedlich... :(
        Höchstwahrscheinlich ein mal string und ein mal int     
        '''
        merged_df = merged_df.reset_index(drop=True)
        print(merged_df.shape)
        
        outputpath = os.path.join(cwd, 'data/torch/datasets/', transformer)
        os.makedirs(outputpath, exist_ok=True)
        merged_df.to_pickle(os.path.join(outputpath, score+'.pkl.gzip'), compression='gzip')
    
    for score in ARTE_SCORES:
        arte_score = arte_scores.filter(['doc_id', score])
        merged_df = pd.merge(embeddings, arte_score, on='doc_id', how='inner') 
        
        merged_df = shuffle(merged_df)
        merged_df = merged_df.reset_index(drop=True)
        print(merged_df.shape)
        
        outputpath = os.path.join(cwd, 'data/torch/datasets/', transformer)
        os.makedirs(outputpath, exist_ok=True)
        merged_df.to_pickle(os.path.join(outputpath, score+'.pkl.gzip'), compression='gzip')