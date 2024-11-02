
import torch
from transformers import BertTokenizer, BertModel
import ir_datasets as ir
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import multiprocessing

# Initialize global variables for tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

def get_cls_sep_embeddings(text, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)

    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    
    cls_embedding = last_hidden_states[0, 0, :]
    sep_embedding = last_hidden_states[0, -1, :]evaluatio

    return cls_embedding.numpy(), sep_embedding.numpy()

def process_document(doc):
    doc_id = doc.doc_id
    doc_text = doc.text

    cls_embedding, sep_embedding = get_cls_sep_embeddings(doc_text)
    combined_embedding = np.concatenate((cls_embedding, sep_embedding))

    # Create a dictionary for the result
    result = {
        'doc_id': doc_id,
        **{f'cls_{j}': cls_embedding[j] for j in range(cls_embedding.shape[0])},
        **{f'sep_{j}': sep_embedding[j] for j in range(sep_embedding.shape[0])}
    }
    
    return result

if __name__ == '__main__':
    dsname = 'beir/nfcorpus'
    datapath = os.path.join(os.getcwd(), 'data')
    
    dataset = ir.load(dsname)
    
    # Create a pool of workers equal to the number of cores available
    num_cores = multiprocessing.cpu_count()  # Or set to a fixed number e.g., 4
    pool = multiprocessing.Pool(processes=num_cores)
    
    # Process the documents in parallel
    results = list(tqdm(pool.imap(process_document, dataset.docs_iter()), total=dataset.docs_count()))
    
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()
    
    # Create a DataFrame from the results
    embeddings_df = pd.DataFrame(results)
    
    # Save the DataFrame as a pickle file
    save_path = os.path.join(datapath, dsname.replace("/", "-"), 'bert.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    embeddings_df.to_pickle(path=save_path)