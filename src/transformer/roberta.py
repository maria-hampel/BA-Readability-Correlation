import ir_datasets as ir
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch
import os


dsname = 'beir/arguana'
datapath=os.path.join(os.getcwd(), 'data/transformer')

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
    dataset = ir.load(dsname)
    results = []
    for i, doc in tqdm(enumerate(dataset.docs_iter())):
        if i >= 10:  # Limit to first 100 documents for demonstration; remove this for full dataset
            break

        doc_id = doc.doc_id
        doc_text = doc.text
        
        roberta_embedding = get_roberta_embeddings(doc_text, model, tokenizer)
        
        results.append({
            'doc_id': doc_id,
            **{f'cls_{j}': roberta_embedding[j].numpy() for j in range(roberta_embedding.shape[0])}# Convert PyTorch tensor to NumPy array
        })

    embeddings_df = pd.DataFrame(results)
    print(embeddings_df.head())
    
    embeddings_df.to_pickle(path=os.path.join(datapath, dsname.replace("/", "-"), 'roberta.pkl.gzip'), compression='gzip')
    