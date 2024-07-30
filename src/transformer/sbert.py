import ir_datasets as ir
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Function to get SBERT embeddings for a given text
def get_sbert_embeddings(text, model, max_length=512):
    # Encode the text
    embeddings = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
    return embeddings

if __name__ == '__main__':
    # Load Sentence-BERT model (example using 'paraphrase-MiniLM-L6-v2')
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Load Dataset (adjust dataset loading as per your requirement)
    dataset = ir.load('antique/test')

    # Read readability scores from CSV
    readability_df = pd.read_csv("data/tira/antique-test-20230107-training.csv")
    readability_df = readability_df.rename(columns={"docno":"doc_id"})
    # readability_df = readability_df[['doc_id', 'flesch_reading_ease', 'flesch_kincaid_grade', 'smog', 'gunning_fog', 
    #                                  'automated_readability_index', 'coleman_liau_index', 'lix', 'rix']]

    # Iterate through the dataset and process each document
    results = []
    for i, doc in tqdm(enumerate(dataset.docs_iter())):
        if i >= 10000:  # Limit to first 10000 documents for demonstration; remove this for full dataset
            break

        doc_id = doc.doc_id
        doc_text = doc.text
        
        # Get SBERT embeddings
        sbert_embeddings = get_sbert_embeddings(doc_text, model)
        
        # Store results
        results.append({
            'doc_id': doc_id,
            **{f'sbert_{j}': sbert_embeddings[j].item() for j in range(sbert_embeddings.shape[0])}
        })

    # Create a DataFrame from the results
    embeddings_df = pd.DataFrame(results)
    
    # Merge embeddings DataFrame with readability scores DataFrame on 'doc_id'
    merged_df = pd.merge(embeddings_df, readability_df, on='doc_id')
    
    # Compute correlation matrix
    correlation_matrix = merged_df.corr()
    
    # Extract correlations with readability scores
    readability_scores = [col for col in readability_df.columns if col != 'doc_id']
    correlations = correlation_matrix.loc[readability_scores, embeddings_df.columns[1:]]
    
    # Visualize the correlations in a heatmap
    plt.figure(figsize=(15, 10))
    sns.set_theme(font_scale=0.5)
    sns.heatmap(correlations, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap between SBERT Embeddings and Readability Scores')
    plt.savefig('data/visualizations/sbert/correlation_heatmap_embeddings.pdf')
    
    
    row_sums = correlations.abs().sum(axis=1)
    col_sums = correlations.abs().sum(axis=0)
    
    plt.figure(figsize=(15, 5))
    row_sums.plot(kind='bar')
    plt.title('Sum of Absolute Values of Correlations for Each Readability Score')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.savefig('data/visualizations/sbert/row_sums_embeddings.pdf')
    plt.close()

    # Visualize the column sums in bar charts and save to directory
    plt.figure(figsize=(15, 5))
    col_sums.plot(kind='bar')
    plt.title('Sum of Absolute Values of Correlations for Each Embedding Dimension')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.savefig('data/visualizations/sbert/col_sums_embeddings.pdf')
    plt.close()