import ir_datasets as ir
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model
import torch

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Function to get GPT-2 embeddings for a given text
def get_gpt2_embeddings(text, model, tokenizer, max_length=512):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # The last hidden-state is the output of the model
    last_hidden_states = outputs.last_hidden_state

    # Take the mean of the last hidden states (pooling strategy)
    mean_embedding = torch.mean(last_hidden_states, dim=1)
    
    # Take the embedding of the last token
    last_token_embedding = last_hidden_states[:, -1, :]

    return mean_embedding, last_token_embedding

if __name__ == '__main__':
    # Load Dataset (adjust dataset loading as per your requirement)
    dataset = ir.load('antique/test')

    # Read readability scores from CSV
    readability_df = pd.read_csv("data/tira/antique-test-20230107-training.csv")
    readability_df = readability_df.rename(columns={"docno":"doc_id"})
    # readability_df = readability_df[['doc_id', 'flesch_reading_ease', 'flesch_kincaid_grade', 'smog', 'gunning_fog', 'automated_readability_index', 'coleman_liau_index', 'lix', 'rix']]

    # Iterate through the dataset and process each document
    results = []
    for i, doc in tqdm(enumerate(dataset.docs_iter())):
        if i >= 10000:  # Limit to first 100 documents for demonstration; remove this for full dataset
            break

        doc_id = doc.doc_id
        doc_text = doc.text
        
        # Get GPT-2 embeddings for the entire document text
        mean_embedding, last_token_embedding = get_gpt2_embeddings(doc_text, model, tokenizer)
        
        # Store results
        results.append({
            'doc_id': doc_id,
            'mean_embedding': mean_embedding.numpy(),  # Convert PyTorch tensor to NumPy array
            'last_token_embedding': last_token_embedding.numpy()  # Convert PyTorch tensor to NumPy array
        })

    # Create a DataFrame from the results
    embeddings_df = pd.DataFrame(results)

    # Flatten the mean embeddings into separate columns
    mean_embedding_columns = [f'mean_embedding_{i}' for i in range(embeddings_df['mean_embedding'][0].shape[1])]
    mean_embedding_matrix = np.vstack(embeddings_df['mean_embedding'].to_numpy())
    embeddings_df[mean_embedding_columns] = pd.DataFrame(mean_embedding_matrix, columns=mean_embedding_columns)

    # Flatten the last token embeddings into separate columns
    last_token_embedding_columns = [f'last_token_embedding_{i}' for i in range(embeddings_df['last_token_embedding'][0].shape[1])]
    last_token_embedding_matrix = np.vstack(embeddings_df['last_token_embedding'].to_numpy())
    embeddings_df[last_token_embedding_columns] = pd.DataFrame(last_token_embedding_matrix, columns=last_token_embedding_columns)

    # Drop the original 'mean_embedding' and 'last_token_embedding' columns
    embeddings_df.drop(columns=['mean_embedding', 'last_token_embedding'], inplace=True)

    # Merge embeddings DataFrame with readability scores DataFrame on 'doc_id'
    merged_df = pd.merge(embeddings_df, readability_df, on='doc_id')

    # Compute correlation matrix
    correlation_matrix = merged_df.corr()

    # Extract correlations with readability scores
    readability_scores = [col for col in readability_df.columns if col != 'doc_id']
    correlations_mean = correlation_matrix.loc[readability_scores, mean_embedding_columns]
    correlations_last_token = correlation_matrix.loc[readability_scores, last_token_embedding_columns]

    plt.figure(figsize=(15, 10))
    sns.set_theme(font_scale=0.5)
    sns.heatmap(correlations_mean, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap between GPT-2 Mean Embeddings and Readability Scores')
    plt.savefig('data/visualizations/gpt_2/correlation_heatmap_mean_embeddings.pdf')
    plt.close()

    plt.figure(figsize=(15, 10))
    sns.set_theme(font_scale=0.5)
    sns.heatmap(correlations_last_token, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap between GPT-2 Last Token Embeddings and Readability Scores')
    plt.savefig('data/visualizations/gpt_2/correlation_heatmap_last_token_embeddings.pdf')
    plt.close()

    # Compute sum of absolute values of correlations for each row and each column
    row_sums_mean = correlations_mean.abs().sum(axis=1)
    col_sums_mean = correlations_mean.abs().sum(axis=0)

    row_sums_last_token = correlations_last_token.abs().sum(axis=1)
    col_sums_last_token = correlations_last_token.abs().sum(axis=0)

    # Visualize the row sums in bar charts and save to directory
    plt.figure(figsize=(15, 5))
    row_sums_mean.plot(kind='bar')
    plt.title('Sum of Absolute Values of Correlations for Each Readability Score (Mean Embeddings)')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.savefig('data/visualizations/gpt_2/row_sums_mean_embeddings.pdf')
    plt.close()

    plt.figure(figsize=(15, 5))
    row_sums_last_token.plot(kind='bar')
    plt.title('Sum of Absolute Values of Correlations for Each Readability Score (Last Token Embeddings)')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.savefig('data/visualizations/gpt_2/row_sums_last_token_embeddings.pdf')
    plt.close()

    # Visualize the column sums in bar charts and save to directory
    plt.figure(figsize=(15, 5))
    col_sums_mean.plot(kind='bar')
    plt.title('Sum of Absolute Values of Correlations for Each Embedding Dimension (Mean Embeddings)')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.savefig('data/visualizations/gpt_2/col_sums_mean_embeddings.pdf')
    plt.close()

    plt.figure(figsize=(15, 5))
    col_sums_last_token.plot(kind='bar')
    plt.title('Sum of Absolute Values of Correlations for Each Embedding Dimension (Last Token Embeddings)')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.savefig('data/visualizations/gpt_2/col_sums_last_token_embeddings.pdf')
    plt.close() 