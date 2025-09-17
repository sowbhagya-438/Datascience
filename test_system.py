"""
Quick test script for BBC News Classification System
"""

import pandas as pd
from train_model import BBCNewsClassifier
import time

def test_system():
    """Test the classification system with a small sample"""
    print("Testing BBC News Classification System...")
    
    # Initialize classifier
    classifier = BBCNewsClassifier()
    
    # Load a small sample of data for testing
    print("Loading sample data...")
    df = pd.read_csv('data/bbc-news-data.csv', sep='\t')
    
    # Take a balanced sample for quick testing
    df_sample = df.groupby('category').head(20)  # Use 20 articles from each category
    
    print(f"Sample dataset shape: {df_sample.shape}")
    print(f"Categories: {df_sample['category'].value_counts().to_dict()}")
    
    # Preprocess text
    print("Preprocessing text...")
    start_time = time.time()
    df_sample['processed_text'] = classifier.preprocessor.preprocess(df_sample['content'])
    preprocessing_time = time.time() - start_time
    print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
    
    # Remove empty texts
    df_sample = df_sample[df_sample['processed_text'].str.len() > 10]
    print(f"Processed sample shape: {df_sample.shape}")
    
    # Train classifier on sample
    print("Training classifier on sample...")
    start_time = time.time()
    accuracy = classifier.train_classifier(df_sample)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Set the processed dataframe for similarity search
    classifier.df_processed = df_sample
    
    # Build KD-Tree
    print("Building KD-Tree...")
    start_time = time.time()
    classifier.build_kdtree(df_sample)
    kdtree_time = time.time() - start_time
    print(f"KD-Tree built in {kdtree_time:.2f} seconds")
    
    # Test classification
    print("\nTesting classification...")
    test_text = "Apple Inc. reported record quarterly revenue driven by strong iPhone sales and services growth."
    
    start_time = time.time()
    category, confidence = classifier.predict_category(test_text)
    prediction_time = time.time() - start_time
    
    print(f"Test text: {test_text}")
    print(f"Predicted Category: {category}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Prediction time: {prediction_time:.4f} seconds")
    
    # Test similarity search
    print("\nTesting similarity search...")
    start_time = time.time()
    similar_articles = classifier.find_similar_articles(test_text, k=5)
    search_time = time.time() - start_time
    
    print(f"Found {len(similar_articles)} similar articles:")
    for i, article in enumerate(similar_articles, 1):
        print(f"{i}. {article['title']} ({article['category']}) - Similarity: {article['similarity_score']:.4f}")
    
    print(f"Similarity search time: {search_time:.4f} seconds")
    
    print("\n✅ System test completed successfully!")
    print(f"Total processing time: {preprocessing_time + training_time + kdtree_time:.2f} seconds")

if __name__ == "__main__":
    test_system()
