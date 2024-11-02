from transformers import AutoModel
from numpy.linalg import norm
import os
import ir_datasets as ir
from tqdm import tqdm
import numpy 
import pandas as pd

model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
dsname = 'clear-corpus'
datapath = 'data/transformer'

def get_embeddings(text, model, ):
    
    embedding = model.encode([text])

    return embedding[0]

if __name__ == '__main__':
    dataset = pd.read_json(path_or_buf='data/api/clear-corpus/all_batches.json', lines=True)

    dataset = dataset[['doc_id', 'text']]
    results = []
    for i, row in tqdm(dataset.iterrows()):
        # if i >= 10:  
        #     break
        
        doc_id = row['doc_id']
        text = row['text']
        
        jina_embedding = get_embeddings(text, model)

        results.append({
            'doc_id': doc_id,
            **{f'cls_{j}': jina_embedding[j] for j in range(jina_embedding.shape[0])}
        })

    embeddings_df = pd.DataFrame(results)
    print(embeddings_df.head())
    
    embeddings_df.to_pickle(path=os.path.join(datapath, dsname.replace("/", "-"), 'jina.pkl.gzip'), compression='gzip')
    