
# from tira.third_party_integrations import ensure_pyterrier_is_loaded
# from tira.rest_api_client import Client
# ensure_pyterrier_is_loaded()
import torch
from transformers import AutoTokenizer, BertModel, BertTokenizer, BertForMaskedLM
import ir_datasets as ir
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def parse_tira_output(tiraname):
  # the software has the (at the moment automatically created) name hushed-vehicle by team tu-dresden-04
  ret = tira.get_run_output('ir-benchmarks/tu-dresden-04/spacy-document-features', dataset) + '/documents.jsonl.gz'
  return pd.read_json(ret, lines=True, dtype={'docno': str})

# Function to get CLS and SEP token embeddings for a given text
def get_cls_sep_embeddings(text, max_length=512):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # The last hidden-state is the output of the model
    last_hidden_states = outputs.last_hidden_state

    # Extract the CLS token embedding (first token)
    cls_embedding = last_hidden_states[0, 0, :]

    # Extract the SEP token embedding (last token)
    sep_embedding = last_hidden_states[0, -1, :]
    
    return cls_embedding, sep_embedding

if __name__ == '__main__':
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load pre-trained model
    model = BertModel.from_pretrained('bert-base-uncased')
    # Load Dataset
    dataset = ir.load('antique/test')
    # Prepare Tira 
    tiraname = 'antique-test-20230107-training'
    results = []
    # readability
    readability_df = pd.read_csv("data/tira/antique-test-20230107-training.csv")
    readability_df=readability_df.rename(columns={"docno":"doc_id"})
    #readability_df = readability_df[['doc_id','flesch_reading_ease', 'flesch_kincaid_grade', 'smog', 'gunning_fog', 'automated_readability_index', 'coleman_liau_index', 'lix', 'rix']]

    
    for i, doc in tqdm(enumerate(dataset.docs_iter())):
        if i >= 10000:  # Limit to first 100 documents for demonstration; remove this for full dataset
            break

        doc_id = doc.doc_id
        doc_text = doc.text
        
        cls_embedding, sep_embedding = get_cls_sep_embeddings(doc_text)
        combined_embedding = np.concatenate((cls_embedding, sep_embedding))
        
        # Store results
        results.append({
            'doc_id': doc_id,
            **{f'cls_{j}': cls_embedding[j] for j in range(cls_embedding.shape[0])},
            **{f'sep_{j}': sep_embedding[j] for j in range(sep_embedding.shape[0])}
        })
        #print(doc_id)

    # Create a DataFrame from the results
    
    embeddings_df = pd.DataFrame(results)
    merged_df = pd.merge(embeddings_df, readability_df, on='doc_id')
    
    #print(merged_df.head())
    #print(f"all doc ids: {merged_df[doc_id]}")
    correlation_matrix = merged_df.corr()
    readability_scores = [col for col in readability_df.columns if col != 'doc_id']
    correlations = correlation_matrix.loc[readability_scores, embeddings_df.columns[1:]]

    # Visualize the correlations in a heatmap
    plt.figure(figsize=(15, 10))
    sns.set_theme(font_scale=0.5)
    sns.heatmap(correlations, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap between BERT Embeddings and Readability Scores')
    plt.savefig('data/visualizations/bert/correlation_heatmap_embeddings.pdf')
    
    
    row_sums = correlations.abs().sum(axis=1)
    col_sums = correlations.abs().sum(axis=0)
    
    plt.figure(figsize=(15, 5))
    row_sums.plot(kind='bar', fontsize=10)
    plt.title('Sum of Absolute Values of Correlations for Each Readability Score')
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.savefig('data/visualizations/bert/row_sums_embeddings.pdf')
    plt.close()

    # Visualize the column sums in bar charts and save to directory
    plt.figure(figsize=(20, 5))
    col_sums.plot(kind='bar')
    plt.title('Sum of Absolute Values of Correlations for Each Embedding Dimension')
    plt.xticks(rotation=45, ha='right', fontsize=6)
    # plt.xticks(np.arange(0, len(col_sums), 4), col_sums.index[::4], rotation=45, ha='right', fontsize=8)
    plt.savefig('data/visualizations/bert/col_sums_embeddings.pdf')
    plt.close()