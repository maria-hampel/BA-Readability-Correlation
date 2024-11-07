from transformers import AutoModel
from numpy.linalg import norm
import os
import ir_datasets as ir
from tqdm import tqdm
import numpy 
import pandas as pd

dsname = 'beir/arguana'
datapath=os.path.join(os.getcwd(), 'data/transformer')

model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) 


def get_embeddings(text, model, ):
    
    embedding = model.encode([text])

    return embedding[0]

if __name__ == '__main__':
    dataset = ir.load(dsname)
    results = []
    for i, doc in tqdm(enumerate(dataset.docs_iter())):
        if i >= 10:  
            break

        doc_id = doc.doc_id
        doc_text = doc.text
        
        jina_embedding = get_embeddings(doc_text, model)

        results.append({
            'doc_id': doc_id,
            **{f'cls_{j}': jina_embedding[j] for j in range(jina_embedding.shape[0])}
        })

    embeddings_df = pd.DataFrame(results)
    print(embeddings_df.head())
    
    embeddings_df.to_pickle(path=os.path.join(datapath, dsname.replace("/", "-"), 'jina.pkl.gzip'), compression='gzip')
    