import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os

score = "flesch_reading_ease"

cwd = os.getcwd()

class ReadabilityDataset(Dataset):
    def __init__(self, embedding_df) -> None:
        # Assuming each row in the DataFrame contains a list of word embeddings for a text.
        # The embeddings will now have shape (num_words, embedding_dim)
        self.embeddings = embedding_df.iloc[:, 1:-1].apply(lambda x: torch.tensor(x, dtype=torch.float32)).tolist()
        self.scores = embedding_df[score].astype('float32').values
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        embedding = self.embeddings[index]  # shape: (num_words, embedding_dim)
        score = torch.tensor(self.scores[index], dtype=torch.float32)
        return embedding, score

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x: shape (batch_size, seq_length, embedding_dim)
        _, (hn, _) = self.rnn(x)  # hn: shape (num_layers, batch_size, hidden_size)
        hn = hn[-1]  # Get the hidden state from the last layer
        output = self.fc(hn)  # Pass through a fully connected layer
        return output

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)    

dspath = os.path.join(cwd, 'data/torch/datasets/bert/' + score + '.pkl.gzip')
df = pd.read_pickle(filepath_or_buffer=dspath, compression='gzip')

print(df.head())
df = df.dropna(subset=[score])
dataset = ReadabilityDataset(df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: x)

input_size = dataset[0][0].shape[1]  # Size of word embedding dimension
hidden_size = 256  # You can adjust this

model = RNNModel(input_size=input_size, hidden_size=hidden_size)
model.to('cpu')
initialize_weights(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 100
train_losses = []
val_losses = []
train_mae = []
val_mae = []

for epoch in range(num_epochs):
    # Training step
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    for batch in train_dataloader:
        embeddings, readability_score = zip(*batch)
        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        readability_score = torch.tensor(readability_score).float()

        outputs = model(embeddings)
        loss = criterion(outputs, readability_score.unsqueeze(1))
        mae = torch.mean(torch.abs(outputs - readability_score.unsqueeze(1)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_mae += mae.item()

    avg_train_loss = running_loss / len(train_dataloader)
    avg_train_mae = running_mae / len(train_dataloader)
    train_losses.append(avg_train_loss)
    train_mae.append(avg_train_mae)

    # Validation step
    model.eval()
    val_loss = 0.0
    val_mae_sum = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            embeddings, readability_score = zip(*batch)
            embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
            readability_score = torch.tensor(readability_score).float()

            outputs = model(embeddings)
            loss = criterion(outputs, readability_score.unsqueeze(1))
            mae = torch.mean(torch.abs(outputs - readability_score.unsqueeze(1)))
            val_loss += loss.item()
            val_mae_sum += mae.item()

    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_mae = val_mae_sum / len(val_dataloader)
    val_losses.append(avg_val_loss)
    val_mae.append(avg_val_mae)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train MAE: {avg_train_mae:.4f}, Val MAE: {avg_val_mae:.4f}')
