from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    'jinaai/jina-reranker-v2-base-multilingual',
    torch_dtype="auto",
    trust_remote_code=True,
)

model.to('cpu') # or 'cpu' if no GPU is available
model.eval()



# construct sentence pairs
sentence_pairs = [[query, doc] for doc in documents]

scores = model.compute_score(sentence_pairs, max_length=1024)

print(scores)