import ir_datasets as ir
import pandas as pd
import torch
import pyterrier as pt
from sentence_transformers import SentenceTransformer, util
import os
import re

DATASETS = ['beir/arguana', 'beir/scifact', 'beir/nfcorpus']
DATAPATH = './data/retrieval/roberta/'

def preprocess_query(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
    return text

if __name__ == '__main__':
    embedder = SentenceTransformer("msmarco-roberta-base-v3")

    allretrieved = []
    top10 = []

    for datasetname in DATASETS:
        dataset = pt.datasets.get_dataset('irds:' + datasetname)
        corpus = list(dataset.get_corpus_iter())
        corpus_texts = [doc['text'] for doc in corpus]  
        doc_ids = [doc['docno'] for doc in corpus]    

        corpus_embeddings = embedder.encode(corpus_texts, convert_to_tensor=True)

        topics = dataset.get_topics()
        if 'text' in topics.columns:
            topics = topics.rename(columns={'text': 'query'})
        elif 'title' in topics.columns:
            topics = topics.rename(columns={'title': 'query'})
        topics = topics[['qid', 'query']]
         
        for _, topic in topics.iterrows():
            query_text = preprocess_query(topic['query'])
            query_embedding = embedder.encode(query_text, convert_to_tensor=True)


            similarity_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_k = 10  
            scores, indices = torch.topk(similarity_scores, k=top_k)

            results = []
            for rank, (score, idx) in enumerate(zip(scores, indices)):
                results.append({
                    'qid': topic['qid'],
                    'docno': doc_ids[idx],
                    'rank': rank + 1,
                    'score': score.item(),
                    'system': 'roberta'
                })
            
            allretrieved.extend(results)
            top10.extend(results[:10])  

    dfallretrieved = pd.DataFrame(allretrieved)
    dftop10 = pd.DataFrame(top10)

    if not os.path.exists(DATAPATH):
        os.makedirs(DATAPATH)
    
    dfallretrieved.to_pickle(path=DATAPATH + 'allretrieved_roberta.pkl.gzip', compression='gzip')
    dftop10.to_pickle(path=DATAPATH + 'top10_roberta.pkl.gzip', compression='gzip')

    print(dfallretrieved.head())
    print(dftop10.head())