import re
import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# Embeddings
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

# -------------------------------
# Text cleaning
# -------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# -------------------------------
# Parse HTML ticket
# -------------------------------
def parse_html_ticket(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            content = soup.get_text().lower()

        ticket_number_tag = soup.find('h1')
        ticket_number = ticket_number_tag.text.split(' ')[-1].strip() if ticket_number_tag else None

        if "database" in content:
            category = "database crash"
        elif "software" in content or "application" in content:
            category = "software crash"
        elif "server" in content or "power supply" in content:
            category = "server crash"
        elif any(k in content for k in ["system", "cpu", "memory", "blue screen", "thermal"]):
            category = "system crash"
        elif any(k in content for k in ["network", "connection", "nic", "ethernet"]):
            category = "network issue"
        else:
            category = "other issue"
            
        ticket_id = f"Ticket #{ticket_number}" if ticket_number else None
        return (ticket_id, category, content)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return (None, None, None)

# -------------------------------
# Create DataFrame from HTMLs
# -------------------------------
def create_tickets_dataframe(directory_path):
    html_files = [f for f in os.listdir(directory_path) if f.endswith('.html')]
    
    if not html_files:
        print(f"No HTML files found in the directory: {directory_path}")
        return pd.DataFrame(columns=['X', 'Y', 'content'])
        
    data = []
    for file_name in html_files:
        file_path = os.path.join(directory_path, file_name)
        ticket_id, category, content = parse_html_ticket(file_path)
        
        if ticket_id and category and content:
            data.append({"X": ticket_id, "Y": category, "content": content})
            
    df = pd.DataFrame(data)
    print(f"Successfully processed {len(df)} HTML files into a DataFrame.")
    return df

# -------------------------------
# Create embeddings
# -------------------------------
def create_embeddings(df, method):
    if df.empty:
        return None, None, "DataFrame is empty, cannot create embeddings."
    
    texts = df['content'].tolist()

    if method == "tfidf":
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(texts)
        return embeddings, vectorizer, "TF-IDF embeddings created."

    elif method == "count":
        vectorizer = CountVectorizer()
        embeddings = vectorizer.fit_transform(texts)
        return embeddings, vectorizer, "CountVectorizer embeddings created."

    elif method == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        embeddings = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_embedding[0])
        return pd.DataFrame(embeddings), tokenizer, "BERT embeddings created."

    elif method == "roberta":
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModel.from_pretrained("roberta-base")
        model.eval()
        embeddings = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_embedding[0])
        return pd.DataFrame(embeddings), tokenizer, "RoBERTa embeddings created."

    elif method == "sbert":
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, convert_to_numpy=True)
        return pd.DataFrame(embeddings), model, "Sentence-BERT embeddings created."

    else:
        return None, None, f"Unknown embedding method: {method}"

# -------------------------------
# Train XGBoost
# -------------------------------
def train_xgboost(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n--- XGBoost Model Results ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, le

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    html_files_directory = 'C:/Users/Administrator/Downloads/webpages out/embeddings'
    
    tickets_df = create_tickets_dataframe(html_files_directory)

    if not tickets_df.empty:
        methods = ["tfidf", "count", "bert", "roberta", "sbert"]  # embedding methods
        accuracy_dict = {}  # store accuracy for each method
        
        for method in methods:
            print(f"\n>>> Using {method.upper()} Embeddings <<<")
            embeddings, vectorizer, message = create_embeddings(tickets_df, method)
            print(f"--- {message} ---")
            
            if embeddings is not None:
                print(f"Shape of {method.upper()} embeddings: {embeddings.shape}")
                model, label_encoder = train_xgboost(embeddings, tickets_df['Y'])
                
                # Evaluate accuracy on full dataset
                le = label_encoder
                y_encoded = le.transform(tickets_df['Y'])
                y_pred = model.predict(embeddings)
                acc = accuracy_score(y_encoded, y_pred)
                accuracy_dict[method.upper()] = acc

        # -------------------------------
        # Print Accuracy Comparison Table
        # -------------------------------
        print("\n=== Embedding Model Accuracy Comparison ===")
        for emb, acc in accuracy_dict.items():
            print(f"{emb:<10}: {acc:.4f}")
        
        # Identify the best model
        best_model = max(accuracy_dict, key=accuracy_dict.get)
        print(f"\n✅ Best Embedding Model: {best_model} with Accuracy {accuracy_dict[best_model]:.4f}")
