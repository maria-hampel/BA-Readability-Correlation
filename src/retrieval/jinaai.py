import ir_datasets as ir
import pandas as pd
import torch
import pyterrier as pt
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import util
import os
import re

DATASETS = ['beir/arguana', 'beir/scifact', 'beir/nfcorpus']
EMBEDDING_PATHS = {
    'beir/arguana': 'data/transformer/beir-arguana/jina.pkl.gzip',
    'beir/scifact': 'data/transformer/beir-scifact/jina.pkl.gzip',
    'beir/nfcorpus': 'data/transformer/beir-nfcorpus/jina.pkl.gzip'
}
DATAPATH = './data/retrieval/jina/'

def preprocess_query(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
    return text

if __name__ == '__main__':
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en')

    allretrieved = []
    top10 = []

    for datasetname in DATASETS:
        dataset = pt.datasets.get_dataset('irds:' + datasetname)

        
        corpus_embeddings_df = pd.read_pickle(EMBEDDING_PATHS[datasetname], compression='gzip')
        doc_ids = corpus_embeddings_df['doc_id'].tolist()
        corpus_embeddings = torch.tensor(corpus_embeddings_df.drop(columns=['doc_id']).values)

        
        topics = dataset.get_topics()
        if 'text' in topics.columns:
            topics = topics.rename(columns={'text': 'query'})
        elif 'title' in topics.columns:
            topics = topics.rename(columns={'title': 'query'})
        topics = topics[['qid', 'query']]

        for _, topic in topics.iterrows():
            query_text = preprocess_query(topic['query'])

            
            with torch.no_grad():
                query_embedding = model(**tokenizer(query_text, padding=True, truncation=True, return_tensors="pt")).last_hidden_state.mean(dim=1)

            
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
                    'system': 'jina'
                })

            allretrieved.extend(results)
            top10.extend(results[:10])

    
    dfallretrieved = pd.DataFrame(allretrieved)
    dftop10 = pd.DataFrame(top10)

    if not os.path.exists(DATAPATH):
        os.makedirs(DATAPATH)
    
    dfallretrieved.to_pickle(path=DATAPATH + 'allretrieved_jina.pkl.gzip', compression='gzip')
    dftop10.to_pickle(path=DATAPATH + 'top10_jina.pkl.gzip', compression='gzip')

    print(dfallretrieved.head())
    print(dftop10.head())