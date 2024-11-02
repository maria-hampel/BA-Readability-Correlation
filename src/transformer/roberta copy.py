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


# Function to get RoBERTa embeddings for a given text
def get_roberta_embeddings(text, model, tokenizer, max_length=512):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # The last hidden-state is the output of the model
    last_hidden_states = outputs.last_hidden_state

    # Extract the first token (CLS token) embedding
    roberta_embedding = last_hidden_states[0, 0, :]

    return roberta_embedding

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
        roberta_embedding = get_roberta_embeddings(text, model, tokenizer)
        
        # Store results
        results.append({
            'doc_id': doc_id,
            **{f'cls_{j}': roberta_embedding[j].numpy() for j in range(roberta_embedding.shape[0])}# Convert PyTorch tensor to NumPy array
        })

    # Create a DataFrame from the results
    embeddings_df = pd.DataFrame(results)
    print(embeddings_df.head())
    
    embeddings_df.to_pickle(path=os.path.join(datapath, dsname.replace("/", "-"), 'roberta.pkl.gzip'), compression='gzip')
    
 
    #embedding_matrix = np.concatenate(embeddings_df['roberta_embedding'].to_numpy(), axis=0)

    # # Flatten the embeddings into separate columns
    # embedding_columns = []
    # for i in range(embedding_matrix.shape[1]):
    #     embedding_columns.append(f'roberta_embedding_{i}')

    # # Create DataFrame with embedding matrix and column names
    # embeddings_df[embedding_columns] = pd.DataFrame(embedding_matrix, columns=embedding_columns)
    
    
    # embeddings_df = pd.DataFrame(results)
    
    # embeddings_df.to_pickle(path=os.path.join(datapath, dsname.replace("/", "-"), 'sbert.pkl'))