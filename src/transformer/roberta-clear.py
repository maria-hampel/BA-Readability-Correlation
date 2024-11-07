import ir_datasets as ir
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch
import os


dsname = 'clear-corpus'
datapath = 'data/transformer'

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')


def get_roberta_embeddings(text, model, tokenizer, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state

    roberta_embedding = last_hidden_states[0, 0, :]

    return roberta_embedding

if __name__ == '__main__':
    dataset = pd.read_json(path_or_buf='data/api/clear-corpus/all_batches.json', lines=True)

    dataset = dataset[['doc_id', 'text']]
    results = []
    for i, row in tqdm(dataset.iterrows()):
        if i >= 10:  # Limit to first 100 documents for demonstration; remove this for full dataset
            break

        doc_id = row['doc_id']
        text = row['text']
        
        roberta_embedding = get_roberta_embeddings(text, model, tokenizer)
        
        results.append({
            'doc_id': doc_id,
            **{f'cls_{j}': roberta_embedding[j].numpy() for j in range(roberta_embedding.shape[0])}# Convert PyTorch tensor to NumPy array
        })

    embeddings_df = pd.DataFrame(results)
    print(embeddings_df.head())
    
    embeddings_df.to_pickle(path=os.path.join(datapath, dsname.replace("/", "-"), 'roberta.pkl.gzip'), compression='gzip')
    
 
