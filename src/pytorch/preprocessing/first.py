import torch
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

def get_cls_embedding(text, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)

    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the CLS token's embedding (first token)
    cls_embedding = outputs.last_hidden_state[0, 0, :]
    return cls_embedding.numpy()


class ReadabilityDataset(Dataset):
    def __init__(self, texts, scores):
        self.texts = texts
        self.scores = scores
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained('bert-base-cased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        score = self.scores[idx]
        
        # Get the CLS embedding
        cls_embedding = get_cls_embedding(text)
        
        # Convert to tensor
        cls_embedding_tensor = torch.tensor(cls_embedding, dtype=torch.float32)
        score_tensor = torch.tensor(score, dtype=torch.float32)
        
        return cls_embedding_tensor, score_tensor