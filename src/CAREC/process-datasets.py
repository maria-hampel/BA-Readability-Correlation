import os
import re
import json
import requests
import ir_datasets
import time 
import pandas as pd
from datetime import datetime, timedelta

with open("api-key.txt", "r") as file:
    API_KEY=file.read()
API_URL = "https://nlp.gsu.edu/ping_api"
SELECTION = ["ARI", "CAREC", "CAREC_M", "CML2RI", "FRE", "FKGL", "NDC", "SMOG", "SentenceBert"]
DAILY_LIMIT = 1000

cwd = os.getcwd()

dsname = "beir/arguana"
dataset = ir_datasets.load(dsname)

outputpath = os.path.join(cwd, "data/api", dsname.replace("/", "-"))
os.makedirs(outputpath, exist_ok=True)

state_file = os.path.join(outputpath, "state.json")

def load_state():
    if os.path.exists(state_file):
        with open(state_file, "r") as file:
            state = json.load(file)
    else:
        state = {
            "batch_num": 1,
            "processed_docs": 0,
            "last_request_date": None,
            "last_doc_id": None
        }
    return state

def save_state(state):
    with open(state_file, "w") as file:
        json.dump(state, file)

def make_api_request(text):
    payload = {
        "APIkey": API_KEY,
        "selection": SELECTION,
        "text": text
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

def sanitize_text(text):
    return text.replace("\n", "")

def write_df(batch_df, batch_num):
    output_file = os.path.join(outputpath, f"batch_{batch_num}.csv")
    batch_df.to_csv(output_file, index=False)
    print(f"Batch {batch_num} saved to {output_file}")

if __name__ == '__main__':
    state = load_state()
    batch_num = state["batch_num"]
    processed_docs = state["processed_docs"]
    last_request_date = state["last_request_date"]
    last_doc_id = state["last_doc_id"]
    found_lastdoc=False
    if last_request_date:
        last_request_date = datetime.strptime(last_request_date, '%Y-%m-%d').date()
    else:
        last_request_date = datetime.now().date()

    current_date = datetime.now().date()

    # Reset processed_docs if the day has changed
    if current_date > last_request_date:
        processed_docs = 0
        state["processed_docs"] = 0
        state["last_request_date"] = str(current_date)
        save_state(state)

    batch_df = pd.DataFrame()

    for doc in dataset.docs_iter():
        doc_id, text = doc.doc_id, doc.text

        if last_doc_id and not found_lastdoc:
            if doc_id != last_doc_id:
                continue  
            found_lastdoc=True
            # Skip already processed documents

        if processed_docs >= DAILY_LIMIT:
            state["last_doc_id"] = doc_id
            print("Daily API request limit reached. Resuming tomorrow.")
            break

        sanitized_text = sanitize_text(text)
        response = make_api_request(sanitized_text)
        if response:
            response_data = {"doc_id": doc_id, "text": sanitized_text}
            response_data.update(response)
            response_df = pd.DataFrame([response_data])
            batch_df = pd.concat([batch_df, response_df], ignore_index=True)
            processed_docs += 1

        state["processed_docs"] = processed_docs
        state["last_doc_id"] = doc_id
        save_state(state)

    if not batch_df.empty:
        write_df(batch_df, batch_num)
        state["batch_num"] += 1
        save_state(state)