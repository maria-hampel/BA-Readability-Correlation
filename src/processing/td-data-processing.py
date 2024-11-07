from pathlib import Path
import pandas as pd
import textstat as ts
import spacy
import textdescriptives as td
import matplotlib.pyplot as plt
import json
import gzip
from tqdm import tqdm
from numba import jit, njit, prange
import ir_datasets

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textdescriptives/all")

# DATASETS = [
#     "beir/arguana",
#     "beir/cqadupstack/mathematica",
#     "beir/nfcorpus",
#     "beir/scifact",
#     "nfcorpus",
# ]

dsname='clear/corpus'

def truncate_to_512_tokens(text):
    doc = nlp.make_doc(text)
    tokens = [token.text for token in doc[:512]]
    return ' '.join(tokens)

def truncate_to_1024_tokens(text):
    doc = nlp.make_doc(text)
    tokens = [token.text for token in doc[:1024]]
    return ' '.join(tokens)

def process_dataset(document_iter):
    result = pd.DataFrame([{'doc_id': i.doc_id} for i in tqdm(document_iter)])
    return result

def process_metrics(document_iter):
    truncated_texts = [doc.text for doc in tqdm(document_iter)]
    docs = nlp.pipe(truncated_texts, n_process=4)
    metrics = td.extract_df(docs, include_text=False)
    return metrics

if __name__ == "__main__":
    dataset = pd.read_json(path_or_buf='data/api/clear-corpus/all_batches.json', lines=True)

    dataset = dataset[['doc_id', 'text']]
    
    output_dir = "./data/spacy/all_tokens"
    output_file = Path(output_dir) / f'{dsname.replace("/", "-")}.jsonl.gz'
    processed_dataset = process_dataset(row for _, row in dataset.iterrows())
    processed_metrics = process_metrics(row for _, row in dataset.iterrows())
    processed_documents = pd.concat([processed_dataset, processed_metrics], axis=1)
    processed_documents.to_json(output_file, lines=True, orient='records')
    print(processed_documents.head())
