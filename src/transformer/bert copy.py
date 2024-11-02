import ir_datasets as ir
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch
from transformers import AutoTokenizer, BertModel, BertTokenizer, BertForMaskedLM

import os


dsname = 'clear-corpus'
datapath = 'data/transformer'

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
    dataset = pd.read_json(path_or_buf='data/api/clear-corpus/all_batches.json', lines=True)

    dataset = dataset[['doc_id', 'text']]
    results = []
    for i, row in tqdm(dataset.iterrows()):
        # if i >= 10:  # Limit to first 100 documents for demonstration; remove this for full dataset
        #     break

        doc_id = row['doc_id']
        text = row['text']
        
        # Get RoBERTa embeddings for the entire document text
        cls_embedding, sep_embedding = get_cls_sep_embeddings(text)
        # Store results
        results.append({
            'doc_id': doc_id,
            **{f'cls_{j}': cls_embedding[j].numpy() for j in range(cls_embedding.shape[0])},
            **{f'sep_{j}': sep_embedding[j].numpy() for j in range(sep_embedding.shape[0])}
        })

    # Create a DataFrame from the results
    embeddings_df = pd.DataFrame(results)
    print(embeddings_df.head())
    
    embeddings_df.to_pickle(path=os.path.join(datapath, dsname.replace("/", "-"), 'bert.pkl.gzip'), compression='gzip')
 
    #embedding_matrix = np.concatenate(embeddings_df['roberta_embedding'].to_numpy(), axis=0)

    # # Flatten the embeddings into separate columns
    # embedding_columns = []
    # for i in range(embedding_matrix.shape[1]):
    #     embedding_columns.append(f'roberta_embedding_{i}')

    # # Create DataFrame with embedding matrix and column names
    # embeddings_df[embedding_columns] = pd.DataFrame(embedding_matrix, columns=embedding_columns)
    
    
    # embeddings_df = pd.DataFrame(results)
    
    # embeddings_df.to_pickle(path=os.path.join(datapath, dsname.replace("/", "-"), 'sbert.pkl'))