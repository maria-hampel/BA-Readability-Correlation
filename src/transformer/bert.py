import torch
from transformers import AutoTokenizer, BertModel, BertTokenizer, BertForMaskedLM
import ir_datasets as ir
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pyarrow.parquet as pq

dsname = 'beir/arguana'
datapath=os.path.join(os.getcwd(), 'data/transformer')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

def get_cls_sep_embeddings(text, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    #print(last_hidden_states.shape)
    cls_embedding = last_hidden_states[0, 0, :]

    sep_embedding = last_hidden_states[0, -1, :]
    
    return cls_embedding, sep_embedding


if __name__ == '__main__':
    dataset = ir.load(dsname)
    results = []
    
    for i, doc in tqdm(enumerate(dataset.docs_iter())):
        if i >= 100:  # Limit to first 100 documents for demonstration; remove this for full dataset
             break

        #print(f"Processing document {i+1}")
        doc_id = doc.doc_id
        doc_text = doc.text
        
        cls_embedding, sep_embedding = get_cls_sep_embeddings(doc_text)
        combined_embedding = np.concatenate((cls_embedding, sep_embedding))
        #print(cls_embedding, sep_embedding)
        results.append({
            'doc_id': doc_id,
            **{f'cls_{j}': cls_embedding[j].numpy() for j in range(cls_embedding.shape[0])},
            **{f'sep_{j}': sep_embedding[j].numpy() for j in range(sep_embedding.shape[0])}
        })
        #print(f"Results accumulated: {len(results)}")

    embeddings_df = pd.DataFrame(results)
    print(embeddings_df.head())
    
    embeddings_df.to_pickle(path=os.path.join(datapath, dsname.replace("/", "-"), 'bert.pkl.gzip'), compression='gzip')