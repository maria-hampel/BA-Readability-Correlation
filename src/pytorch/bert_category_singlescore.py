import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
from sklearn.metrics import confusion_matrix

score = "flesch_reading_ease"
cwd = os.getcwd()

def score_to_tensor(score: float):
    boundaries = [-float('inf'), 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]

    category_index = []
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= score < boundaries[i + 1]:
            category_index.append(1)
        else:
            category_index.append(0)

    return torch.tensor(category_index, dtype=torch.float32)

def score_to_category(score: float):
    boundaries = [-float('inf'), 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= score < boundaries[i + 1]:
            return i
    return -1  

class ReadabilityDataset(Dataset):
    def __init__(self, embedding_df) -> None:
        self.embeddings = embedding_df.iloc[:, 1:-1].astype('float32').values
        self.scores = embedding_df[score].astype('float32').values
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        embedding = torch.tensor(self.embeddings[index], dtype=torch.float32)
        score_tensor = score_to_tensor(self.scores[index])
        category = score_to_category(self.scores[index])
        return embedding, score_tensor, category

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
        self.fc5 = nn.Linear(64, 12)  # 12 categories

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
model.to('cpu')
initialize_weights(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for embeddings, readability_category, _ in train_dataloader:
        embeddings = embeddings.float()
        target = torch.argmax(readability_category, dim=1)
        
        outputs = model(embeddings)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    all_true_categories = []
    all_predicted_categories = []

    with torch.no_grad():
        for embeddings, readability_category, true_category in val_dataloader:
            embeddings = embeddings.float()
            target = torch.argmax(readability_category, dim=1)
            outputs = model(embeddings)

            predicted_category = torch.argmax(outputs, dim=1)
            all_true_categories.extend(true_category.numpy())
            all_predicted_categories.extend(predicted_category.numpy())

            loss = criterion(outputs, target)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Confusion Matrix
conf_matrix = confusion_matrix(all_true_categories, all_predicted_categories)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=range(12), yticklabels=range(12))
plt.xlabel('Predicted Category')
plt.ylabel('True Category')
plt.title('Confusion Matrix Heatmap')
plt.show()