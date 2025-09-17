"""
BBC News Text Classification and KD-Tree Recommendations
Training script for text classification model and KD-Tree index
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import cKDTree
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("NLTK data downloaded successfully")
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")
        print("The application will still work with basic text processing")

download_nltk_data()

class TextPreprocessor:
    """Text preprocessing pipeline for BBC News data"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.nltk_available = True
        except Exception as e:
            print(f"Warning: NLTK not fully available: {e}")
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            self.lemmatizer = None
            self.nltk_available = False
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        if self.nltk_available:
            tokens = nltk.word_tokenize(text)
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
        else:
            # Fallback to simple tokenization
            tokens = text.split()
            tokens = [token for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess(self, texts):
        """Apply full preprocessing pipeline"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        processed_texts = [self.tokenize_and_lemmatize(text) for text in cleaned_texts]
        return processed_texts

class BBCNewsClassifier:
    """BBC News Text Classification and KD-Tree Recommendations System"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
        self.label_encoder = LabelEncoder()
        self.kdtree = None
        self.tfidf_matrix = None
        self.df_processed = None
        
    def load_data(self, file_path):
        """Load and preprocess the BBC News dataset"""
        print("Loading BBC News dataset...")
        
        # Load data with tab separator
        df = pd.read_csv(file_path, sep='\t')
        
        print(f"Dataset shape: {df.shape}")
        print(f"Categories: {df['category'].value_counts().to_dict()}")
        
        # Preprocess text
        print("Preprocessing text...")
        df['processed_text'] = self.preprocessor.preprocess(df['content'])
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 10]
        
        self.df_processed = df
        print(f"Processed dataset shape: {df.shape}")
        
        return df
    
    def train_classifier(self, df):
        """Train the text classification model"""
        print("Training text classification model...")
        
        # Prepare features and labels
        X = df['processed_text']
        y = df['category']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Vectorize text
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        y_pred = self.classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Classification Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        return accuracy
    
    def build_kdtree(self, df):
        """Build KD-Tree for similarity recommendations"""
        print("Building KD-Tree for recommendations...")
        
        # Vectorize all texts
        self.tfidf_matrix = self.vectorizer.transform(df['processed_text'])
        
        # Build KD-Tree
        self.kdtree = cKDTree(self.tfidf_matrix.toarray())
        
        print(f"KD-Tree built with {self.tfidf_matrix.shape[0]} documents")
        
    def predict_category(self, text):
        """Predict category for new text"""
        # Preprocess text
        processed_text = self.preprocessor.preprocess([text])[0]
        
        # Vectorize
        text_vector = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.classifier.predict(text_vector)[0]
        probability = self.classifier.predict_proba(text_vector)[0]
        
        category = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probability[prediction]
        
        return category, confidence
    
    def find_similar_articles(self, text, k=10):
        """Find similar articles using KD-Tree"""
        # Preprocess text
        processed_text = self.preprocessor.preprocess([text])[0]
        
        # Vectorize
        text_vector = self.vectorizer.transform([processed_text])
        
        # Query KD-Tree
        distances, indices = self.kdtree.query(text_vector.toarray(), k=k+1)
        dists = distances[0]
        # Convert Euclidean distance to cosine similarity.
        # With l2-normalized TF-IDF, cos_sim = 1 - (dist^2)/2 in [0, 1].
        cosine_similarities = 1 - (dists ** 2) / 2.0
        
        # Get similar articles (exclude the query itself if it's in the dataset)
        similar_articles = []
        for i, (sim, idx) in enumerate(zip(cosine_similarities, indices[0])):
            if i == 0 and dists[i] < 1e-6:  # Skip if it's the same article
                continue
            
            article = self.df_processed.iloc[idx]
            similar_articles.append({
                'title': article['title'],
                'category': article['category'],
                'content': article['content'][:200] + '...',
                'similarity_score': float(max(0.0, min(1.0, sim)))
            })
            
            if len(similar_articles) >= k:
                break
        
        return similar_articles
    
    def save_models(self):
        """Save trained models"""
        print("Saving models...")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save models
        joblib.dump(self.classifier, 'models/classifier.pkl')
        joblib.dump(self.vectorizer, 'models/vectorizer.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        joblib.dump(self.kdtree, 'models/kdtree.pkl')
        joblib.dump(self.tfidf_matrix, 'models/tfidf_matrix.pkl')
        joblib.dump(self.df_processed, 'models/processed_data.pkl')
        
        print("Models saved successfully!")
    
    def load_models(self):
        """Load pre-trained models"""
        print("Loading pre-trained models...")
        
        self.classifier = joblib.load('models/classifier.pkl')
        self.vectorizer = joblib.load('models/vectorizer.pkl')
        self.label_encoder = joblib.load('models/label_encoder.pkl')
        self.kdtree = joblib.load('models/kdtree.pkl')
        self.tfidf_matrix = joblib.load('models/tfidf_matrix.pkl')
        self.df_processed = joblib.load('models/processed_data.pkl')
        
        print("Models loaded successfully!")

def main():
    """Main training function"""
    print("=" * 60)
    print("BBC News Text Classification and KD-Tree Recommendations")
    print("=" * 60)
    
    # Initialize classifier
    classifier = BBCNewsClassifier()
    
    # Load and preprocess data
    df = classifier.load_data('data/bbc-news-data.csv')
    
    # Train classification model
    accuracy = classifier.train_classifier(df)
    
    # Build KD-Tree for recommendations
    classifier.build_kdtree(df)
    
    # Save models
    classifier.save_models()
    
    # Test the system
    print("\n" + "=" * 60)
    print("Testing the system...")
    print("=" * 60)
    
    # Test with a sample text
    test_text = """
    Apple Inc. reported record quarterly revenue of $123.9 billion for the first quarter of 2024, 
    driven by strong iPhone sales and services growth. The company's stock price surged following 
    the announcement, with investors optimistic about the company's future prospects in the 
    competitive technology market.
    """
    
    # Predict category
    category, confidence = classifier.predict_category(test_text)
    print(f"Predicted Category: {category}")
    print(f"Confidence: {confidence:.4f}")
    
    # Find similar articles
    similar_articles = classifier.find_similar_articles(test_text, k=5)
    print(f"\nTop 5 Similar Articles:")
    for i, article in enumerate(similar_articles, 1):
        print(f"{i}. {article['title']} ({article['category']}) - Similarity: {article['similarity_score']:.4f}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
