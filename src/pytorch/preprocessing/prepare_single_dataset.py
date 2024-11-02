import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch as nn
import gzip

cwd = os.getcwd()

dsname = 'beir/arguana'

score = 'flesch_reading_ease'

bert_embeddings_path = os.path.join(cwd, 'data/transformer', dsname.replace('/', '-'), 'bert.pkl.gzip')
spacy_scores_path = os.path.join(cwd, 'data/spacy/512token', dsname.replace('/','-')+'.jsonl')
arte_scores_path = os.path.join(cwd, 'data/api', dsname.replace('/', '-'), 'all_batches.json')


if __name__ == '__main__':
    bert_embeddings = pd.read_pickle(bert_embeddings_path, compression='gzip')

    for column in bert_embeddings.columns:
        if column != 'doc_id':
            bert_embeddings[column].apply(lambda x: x.item())
    bert_embeddings = bert_embeddings[bert_embeddings.columns.drop(list(bert_embeddings.filter(regex='sep_')))]
    spacy_scores = pd.read_json(spacy_scores_path, lines=True)
    spacy_scores = spacy_scores.filter(['doc_id',score])
    merged_df = pd.merge(bert_embeddings, spacy_scores, on='doc_id', how='inner')
    outputpath = os.path.join(cwd, 'data/torch', dsname.replace('/', '-'), score)
    os.makedirs(outputpath, exist_ok=True)
    merged_df.to_pickle(os.path.join(outputpath, 'bert-512.pkl.gzip'), compression='gzip')