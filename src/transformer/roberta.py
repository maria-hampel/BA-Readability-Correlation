import ir_datasets as ir
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch

# Load RoBERTa tokenizer and model
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
    roberta_embedding = last_hidden_states[:, 0, :]

    return roberta_embedding

if __name__ == '__main__':
    # Load Dataset (adjust dataset loading as per your requirement)
    dataset = ir.load('antique/test')

    # Read readability scores from CSV
    readability_df = pd.read_csv("data/antique-test-20230107-training.csv")
    readability_df = readability_df.rename(columns={"docno":"doc_id"})
    # readability_df = readability_df[['doc_id', 'flesch_reading_ease', 'flesch_kincaid_grade', 'smog', 'gunning_fog', 
    #                                  'automated_readability_index', 'coleman_liau_index', 'lix', 'rix']]

    # Iterate through the dataset and process each document
    results = []
    for i, doc in tqdm(enumerate(dataset.docs_iter())):
        if i >= 10000:  # Limit to first 100 documents for demonstration; remove this for full dataset
            break

        doc_id = doc.doc_id
        doc_text = doc.text
        
        # Get RoBERTa embeddings for the entire document text
        roberta_embedding = get_roberta_embeddings(doc_text, model, tokenizer)
        
        # Store results
        results.append({
            'doc_id': doc_id,
            'roberta_embedding': roberta_embedding.numpy()  # Convert PyTorch tensor to NumPy array
        })

    # Create a DataFrame from the results
    embeddings_df = pd.DataFrame(results)

    # Reshape embedding matrix to 2D array
    embedding_matrix = np.concatenate(embeddings_df['roberta_embedding'].to_numpy(), axis=0)

    # Flatten the embeddings into separate columns
    embedding_columns = []
    for i in range(embedding_matrix.shape[1]):
        embedding_columns.append(f'roberta_embedding_{i}')

    # Create DataFrame with embedding matrix and column names
    embeddings_df[embedding_columns] = pd.DataFrame(embedding_matrix, columns=embedding_columns)

    # Drop the original 'roberta_embedding' column
    embeddings_df.drop(columns=['roberta_embedding'], inplace=True)

    # Merge embeddings DataFrame with readability scores DataFrame on 'doc_id'
    merged_df = pd.merge(embeddings_df, readability_df, on='doc_id')

    # Compute correlation matrix
    correlation_matrix = merged_df.corr()

    # Extract correlations with readability scores
    readability_scores = [col for col in readability_df.columns if col != 'doc_id']
    correlations = correlation_matrix.loc[readability_scores, embedding_columns]

    # Visualize the correlations in a heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlations, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap between RoBERTa Embeddings and Readability Scores')
    plt.show()