import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch as nn
import gzip

cwd = os.getcwd()

dsname = 'beir/nfcorpus'

bert_embeddings_path = os.path.join(cwd, 'data/transformer', dsname.replace('/', '-'), 'bert.pkl.gzip')
spacy_scores_path = os.path.join(cwd, 'data/spacy/512token', dsname.replace('/','-')+'.jsonl')
arte_scores_path = os.path.join(cwd, 'data/api', dsname.replace('/', '-'), 'all_batches.json')


if __name__ == '__main__':
    bert_embeddings = pd.read_pickle(bert_embeddings_path, compression='gzip')

    for column in bert_embeddings.columns:
        if column != 'doc_id':
            bert_embeddings[column].apply(lambda x: x.item())
    spacy_scores = pd.read_json(spacy_scores_path, lines=True)
    arte_scores = pd.read_json(arte_scores_path, lines=True)
    
    merged_df = pd.merge(bert_embeddings, spacy_scores, on='doc_id', how='inner')
    merged_df = pd.merge(merged_df, arte_scores, on='doc_id', how='inner')
    merged_df = merged_df.set_index('doc_id').drop(columns=['text'])
    
    print(merged_df.head())
    
    correlation=merged_df.corr()
    print(merged_df.head(5))
    plt.figure(figsize=(15, 10))
    sns.set_theme(font_scale=0.5)
    sns.heatmap(correlation, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    #plt.savefig(data_path+'corr.pdf')
    plt.show()