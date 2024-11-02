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
          "lix","rix", "FRE.sc",
          "FKGL.sc",
          "ARI.sc",
          "SMOG.sc",
          "CAREC.sc",
          "CAREC_M.sc",
          "CML2RI.sc"]

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

def calculate_accuracy(predictions, targets, tolerance=0.1):
    absolute_error = torch.abs(predictions - targets)
    allowable_error = tolerance * torch.abs(targets)
    accurate_predictions = torch.le(absolute_error, allowable_error).float()
    accuracy = accurate_predictions.mean() * 100
    return accuracy

def r2_score(predictions, targets):
    ss_res = torch.sum((predictions - targets) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    return 1 - ss_res / ss_tot

for score in SCORES:
    dspath = os.path.join(cwd, 'data/torch/datasets/jina/'+score+'.pkl.gzip')
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


    num_epochs = 10
    train_losses = []
    val_losses = []
    train_mae = []
    val_mae = []
    train_r2_scores = []
    val_r2_scores = []
    train_accuracies = []
    val_accuracies = []     

    for epoch in range(num_epochs):
        # Training step
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        running_r2 = 0.0  
        running_accuracy = 0.0
        for embeddings, readability_score in train_dataloader:
            embeddings, readability_score = embeddings.float(), readability_score.float()

            outputs = model(embeddings)
            loss = criterion(outputs, readability_score.unsqueeze(1))
            mae = torch.mean(torch.abs(outputs - readability_score.unsqueeze(1)))
            r2 = r2_score(outputs, readability_score.unsqueeze(1))  
            accuracy = calculate_accuracy(outputs, readability_score.unsqueeze(1))  


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mae += mae.item()
            running_r2 += r2.item()  
            running_accuracy += accuracy.item()  


        avg_train_loss = running_loss / len(train_dataloader)
        avg_train_mae = running_mae / len(train_dataloader)
        avg_train_r2 = running_r2 / len(train_dataloader)  
        avg_train_accuracy = running_accuracy / len(train_dataloader)
        train_losses.append(avg_train_loss)
        train_mae.append(avg_train_mae)
        train_r2_scores.append(avg_train_r2)  
        train_accuracies.append(avg_train_accuracy)

        # Validation step
        model.eval()
        val_loss = 0.0
        val_mae_sum = 0.0
        val_r2_sum = 0.0  
        val_accuracy_sum = 0.0
        with torch.no_grad():
            for embeddings, readability_score in val_dataloader:
                embeddings, readability_score = embeddings.float(), readability_score.float()

                outputs = model(embeddings)
                loss = criterion(outputs, readability_score.unsqueeze(1))
                mae = torch.mean(torch.abs(outputs - readability_score.unsqueeze(1)))
                r2 = r2_score(outputs, readability_score.unsqueeze(1))  
                accuracy = calculate_accuracy(outputs, readability_score.unsqueeze(1))  


                val_loss += loss.item()
                val_mae_sum += mae.item()
                val_r2_sum += r2.item() 
                val_accuracy_sum += accuracy.item()

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_mae = val_mae_sum / len(val_dataloader)
        avg_val_r2 = val_r2_sum / len(val_dataloader) 
        avg_val_accuracy = val_accuracy_sum / len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_mae.append(avg_val_mae)
        val_r2_scores.append(avg_val_r2) 
        val_accuracies.append(avg_val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train MAE: {avg_train_mae:.4f}, Val MAE: {avg_val_mae:.4f}, Train R²: {avg_train_r2:.4f}, Val R²: {avg_val_r2:.4f}, Train Accuracy: {avg_train_accuracy:.2f}%, Val Accuracy: {avg_val_accuracy:.2f}%')
        
    # Plotting the loss
    plt.figure(figsize=(18,12))
    plt.subplot(2, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    # Plotting the accuracy (MAE)
    plt.subplot(2, 2, 2)
    plt.plot(range(1, num_epochs+1), train_mae, label='Training MAE')
    plt.plot(range(1, num_epochs+1), val_mae, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.title('MAE over Epochs')
    
    plt.subplot(2, 2, 3)
    plt.plot(range(1, num_epochs+1), train_r2_scores, label='Training R²')
    plt.plot(range(1, num_epochs+1), val_r2_scores, label='Validation R²')
    plt.xlabel('Epochs')
    plt.ylabel('R² Score')
    plt.legend()
    plt.title('R² Score over Epochs')
    
    # Plotting the accuracy as a percentage
    plt.subplot(2, 2, 4)  
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.tight_layout()


    visdir = os.path.join(cwd, 'data/torch/loss-vis-dropout/jina/')
    os.makedirs(visdir, exist_ok=True)
    plt.savefig(visdir+score+'.pdf')

    modeldir = os.path.join(cwd, 'data/torch/models/jina' )
    os.makedirs(modeldir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cwd, 'data/torch/models/jina/'+score+'_ann_model.pth'))
