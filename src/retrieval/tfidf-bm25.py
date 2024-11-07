import ir_datasets as ir 
import pandas as pd 
import torch 
import pyterrier as pt 
# import pt_datasets
# from pt_datasets import load_dataset, create_dataloader
import os
import re


DATASETS=['beir/arguana','beir/scifact','beir/nfcorpus']
DATAPATH = './data/retrieval/classic/'

def preprocess_query(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
    return text

if __name__ == '__main__':
    allretrieved = []
    top10 = []
    for datasetname in DATASETS:
        dataset = pt.datasets.get_dataset('irds:'+datasetname)
        indexerpath = './'+datasetname.replace('/', '-')
        if not os.path.exists(indexerpath):
            indexer = pt.IterDictIndexer(indexerpath, meta={'docno': 50, 'text': 8192})
            index = indexer.index(dataset.get_corpus_iter())
        else:
            index = pt.IndexFactory.of(indexerpath+'/data.properties')
        tfidf = pt.terrier.Retriever(index, wmodel='TF_IDF')
        bm25 = pt.terrier.Retriever(index, wmodel='BM25')
        topics = dataset.get_topics()
        if 'url' in topics.columns:
            topics['query'] = topics['text']
            topics = topics.drop(columns=['url','text'])
            
        topics['query'] = topics['query'].apply(preprocess_query)
        retrieved_bm25 = bm25(topics)
        retrieved_tfidf = tfidf(topics)
        retrieved_bm25['system'] = 'bm25'
        retrieved_tfidf['system'] = 'tfidf'
        
        allretrieved.append(retrieved_bm25)
        allretrieved.append(retrieved_tfidf)
        top10.append(retrieved_bm25.loc[retrieved_bm25['rank'] < 10].copy())
        top10.append(retrieved_tfidf.loc[retrieved_tfidf['rank'] < 10].copy())
        
    dfallretrieved = pd.concat(allretrieved)
    dftop10 = pd.concat(top10)
    
    print(dfallretrieved.head())
    print(dftop10.head())
    
    if not os.path.exists(DATAPATH):
        os.makedirs(DATAPATH)
    
    dfallretrieved.to_pickle(path=DATAPATH +'allretrieved.pkl.gzip', compression='gzip')
    dftop10.to_pickle(path=DATAPATH+'top10.pkl.gzip', compression='gzip')
        
        