import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch as nn
import gzip

cwd = os.getcwd()

dsname = 'beir/nfcorpus'
data_path='data/visualizations/heatmap/'
EMBEDDING_PATHS = [
    'data/transformer/beir-arguana/jina.pkl.gzip',
    'data/transformer/beir-scifact/jina.pkl.gzip',
    'data/transformer/beir-nfcorpus/jina.pkl.gzip'
]

metrics = [ 'flesch_kincaid_grade', 'smog', 'automated_readability_index', 'CAREC.sc',]

readability_datasets = ['data/spacy/all_tokens/beir-arguana.jsonl', 'data/spacy/all_tokens/beir-nfcorpus.jsonl', 'data/spacy/all_tokens/beir-scifact.jsonl']
carec_datasets = ['data/api/beir-arguana/all_batches.json', 'data/api/beir-nfcorpus/all_batches.json', 'data/api/beir-scifact/all_batches.json']
if __name__ == '__main__':
    bertlist = []
    for item in EMBEDDING_PATHS:
        embeddings = pd.read_pickle(item, compression='gzip')
        bertlist.append(embeddings)
    bert_embeddings = pd.concat(bertlist)
    
    readabilitydatalist =[]
    for item in readability_datasets: 
        df = pd.read_json(item, lines=True)
        df['docno'] = df['docno'].astype(str)  
        df.rename(columns={'docno': 'doc_id'}, inplace=True)
        readabilitydatalist.append(df)
    readabilitydf = pd.concat(readabilitydatalist)

    
    careclist =[]
    for item in carec_datasets:
        df = pd.read_json(item, lines=True)
        
        df['doc_id'] = df['doc_id'].astype(str) 
        careclist.append(df)
    carecdf = pd.concat(careclist)
    
    
    # for column in bert_embeddings.columns:
    #     if column != 'doc_id':
    #         bert_embeddings[column].apply(lambda x: x.item())


    merged_df = pd.merge(readabilitydf, carecdf, on='doc_id', how='inner')
    merged_df = merged_df.loc[:, ['doc_id']+metrics]
    merged_df = pd.merge(merged_df,bert_embeddings, on= 'doc_id', how='inner')
    merged_df = merged_df.set_index('doc_id')
    
    print(merged_df.head())
    
    correlation=merged_df.corr()
  
    correlation = correlation.loc[:, metrics].rename(columns={'flesch_kincaid_grade': 'FKGL', 'smog': 'SMOG', 'automated_readability_index': 'ARI', 'CAREC.sc': 'CAREC'})
    correlation = correlation.T.drop(columns=metrics)
    

    plt.figure(figsize=(15, 4))
    # xticks = correlation.columns
    # keptticks = xticks[::int(len(xticks)/10)]
    # xticks = ['' for y in xticks]
    # xticks[::int(len(xticks)/10)] = keptticks
    sns.set_theme(font_scale=1.5)
    sns.heatmap(correlation, xticklabels=[], annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.yticks(rotation=0) 
    plt.xlabel('JinaAI Embedding')
    plt.savefig(data_path+'corr.svg')
    #plt.show()