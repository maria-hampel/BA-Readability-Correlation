import torch
from transformers import AutoTokenizer, BertModel, BertTokenizer
import ir_datasets as ir
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

# Dataset and model configuration
dsname = 'beir/arguana'
datapath = os.path.join(os.getcwd(), 'data/transformer')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

def get_all_token_embeddings(text, max_length=512):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    # Pass through the model to get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Retrieve the last hidden state (all token embeddings)
    last_hidden_states = outputs.last_hidden_state
    
    # Convert embeddings to numpy format for easier storage
    embeddings = last_hidden_states.squeeze(0).numpy()  # (sequence_length, hidden_size)
    print(embeddings.shape())
    return embeddings

if __name__ == '__main__':
    dataset = ir.load(dsname)
    results = []
    
    # Iterate over documents
    for i, doc in tqdm(enumerate(dataset.docs_iter())):
        if i >= 10:  # Limit to first 100 documents for demonstration; remove this for full dataset
            break

        doc_id = doc.doc_id
        doc_text = doc.text
        
        # Get embeddings for all tokens in the document
        embeddings = get_all_token_embeddings(doc_text)
        
        # Store results: doc_id and the list of token embeddings
        results.append({
            'doc_id': doc_id,
            'token_embeddings': embeddings.tolist()  # Store embeddings as a list
        })

    # Create a DataFrame where each row contains the doc_id and list of token embeddings
    embeddings_df = pd.DataFrame(results)
    print(embeddings_df.head())

    # Save DataFrame as a compressed pickle file
    save_path = os.path.join(datapath, dsname.replace("/", "-"), 'bert_all_tokens.pkl.gzip')
    embeddings_df.to_pickle(path=save_path, compression='gzip')