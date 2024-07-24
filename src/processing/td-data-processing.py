
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

DATASETS = [
    "beir/arguana",
    "beir/cqadupstack/mathematica",
    "beir/nfcorpus",
    "beir/scifact",
    "nfcorpus",
]

def process_dataset(document_iter):
    result = pd.DataFrame([{'docno': i.doc_id} for i in tqdm(document_iter)])
    return result

def process_metrics(document_iter):
    docs = nlp.pipe([doc.text for doc in tqdm(document_iter)], n_process=2)
    metrics = td.extract_df(docs, include_text=False)
    return metrics

if __name__ == "__main__":
    for dsname in DATASETS:
        dataset = ir_datasets.load(dsname)
        output_dir = "./data"
        output_file = Path(output_dir) / f'{dsname.replace("/", "-")}.jsonl.gz'
        processed_dataset = process_dataset(dataset.docs_iter())
        processed_metrics = process_metrics(dataset.docs_iter())
        processed_documents = pd.concat([processed_dataset, processed_metrics], axis=1)
        processed_documents.to_json(output_file, lines=True, orient='records')