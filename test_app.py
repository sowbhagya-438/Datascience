"""
Test script for the BBC News Classification System
"""

from simple_train import SimpleBBCClassifier

def test_classification():
    """Test the classification system"""
    print("Testing BBC News Classification System...")
    
    # Load models
    classifier = SimpleBBCClassifier()
    classifier.load_models()
    
    # Test texts
    test_texts = [
        "Apple Inc. reported record quarterly revenue driven by strong iPhone sales and services growth.",
        "New AI breakthrough in machine learning promises to revolutionize data processing capabilities.",
        "Government announces new policy changes affecting healthcare and education sectors.",
        "Championship game ends in dramatic overtime victory with record-breaking attendance.",
        "Hollywood blockbuster breaks box office records with stunning visual effects."
    ]
    
    expected_categories = ["business", "tech", "politics", "sport", "entertainment"]
    
    print("\nTesting Classification:")
    print("=" * 60)
    
    for i, (text, expected) in enumerate(zip(test_texts, expected_categories)):
        category, confidence = classifier.predict_category(text)
        print(f"\nTest {i+1}:")
        print(f"Text: {text[:50]}...")
        print(f"Expected: {expected}")
        print(f"Predicted: {category}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Correct: {'✅' if category.lower() == expected else '❌'}")
    
    # Test similarity search
    print("\n\nTesting Similarity Search:")
    print("=" * 60)
    
    query_text = "Apple Inc. reported record quarterly revenue driven by strong iPhone sales and services growth."
    similar_articles = classifier.find_similar_articles(query_text, k=5)
    
    print(f"\nQuery: {query_text}")
    print(f"\nTop 5 Similar Articles:")
    for i, article in enumerate(similar_articles, 1):
        print(f"{i}. {article['title']} ({article['category']}) - Similarity: {article['similarity_score']:.4f}")
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    test_classification()
