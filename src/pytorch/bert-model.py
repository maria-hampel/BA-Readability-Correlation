import torch
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import gzip
import numpy as np

SCORES = ["flesch_reading_ease",
          "flesch_kincaid_grade",
          "smog",
          "gunning_fog",
          "automated_readability_index",
          "coleman_liau_index",
          "lix","rix"]

cwd = os.getcwd()

class ReadabilityDataset(Dataset):
    def __init__(self, embedding_df) -> None:
        self.embeddings = embedding_df.iloc[:, 1:-1].astype('float32').values
        self.scores = embedding_df[score].astype('float32').values
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        embedding = torch.tensor(self.embeddings[index], dtype=torch.float32)
        score = torch.tensor(self.scores[index], dtype=torch.float32)
        return embedding, score

class ANNModel(nn.Module):
    def __init__(self, input_size):
        super(ANNModel, self).__init__()
        self.dropout0 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_size, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        #self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.dropout0(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)    


for score in SCORES:
    dspath = os.path.join(cwd, 'data/torch/datasets/bert/'+score+'.pkl.gzip')
    df = pd.read_pickle(filepath_or_buffer=dspath, compression='gzip')

    print(df.head())
    df = df.dropna(subset=[score])
    dataset = ReadabilityDataset(df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_size = dataset[0][0].shape[0]  # Size of CLS embedding 
    model = ANNModel(input_size=input_size)
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)

    initialize_weights(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs = 1000
    train_losses = []
    val_losses = []
    train_mae = []
    val_mae = []

    for epoch in range(num_epochs):
        # Training step
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        for embeddings, readability_score in train_dataloader:
            embeddings, readability_score = embeddings.float(), readability_score.float()

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
            for embeddings, readability_score in val_dataloader:
                embeddings, readability_score = embeddings.float(), readability_score.float()
    
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

    # Losses
    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_mae, label='Training MAE')
    plt.plot(range(1, num_epochs+1), val_mae, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.title('MAE over Epochs')

    plt.tight_layout()


    visdir = os.path.join(cwd, 'data/torch/loss-vis-dropout/')
    os.makedirs(visdir, exist_ok=True)
    plt.savefig(visdir+score+'.pdf')

    modeldir = os.path.join(cwd, 'data/torch/models/' )
    os.makedirs(modeldir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cwd, 'data/torch/models/'+score+'_bert_ann_model.pth'))
