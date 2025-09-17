# BBC News Text Classification and KD-Tree Recommendations

A comprehensive text classification system that predicts news article categories and provides similar article recommendations using TF-IDF vectorization and KD-Tree indexing.

## 🎯 Features

- **Text Classification**: Predict news article categories (business, tech, politics, sport, entertainment)
- **Similarity Search**: Find top 10 similar articles using KD-Tree
- **Interactive Web App**: Streamlit-based user interface
- **Real-time Analysis**: Process any text input instantly
- **Performance Metrics**: Comprehensive evaluation and visualization

## 📊 Dataset

The BBC News dataset contains ~2,225 news articles across 5 categories:
- **Business**: 510 articles
- **Tech**: 401 articles  
- **Politics**: 417 articles
- **Sport**: 511 articles
- **Entertainment**: 386 articles

## 🏗️ Project Structure

```
bbc_classification_kdtree/
├── data/
│   └── bbc-news-data.csv          # BBC News dataset
├── models/
│   ├── classifier.pkl             # Trained Logistic Regression model
│   ├── vectorizer.pkl            # TF-IDF vectorizer
│   ├── label_encoder.pkl         # Label encoder
│   ├── kdtree.pkl                # KD-Tree index
│   ├── tfidf_matrix.pkl          # TF-IDF matrix
│   └── processed_data.pkl         # Processed dataset
├── app.py                         # Streamlit web application
├── train_model.py                 # Model training script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the BBC News dataset
- Train a Logistic Regression classifier
- Build a KD-Tree index for similarity search
- Save all models to the `models/` directory

### 3. Run the Web Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 🔧 Technical Implementation

### Text Preprocessing Pipeline

1. **Text Cleaning**:
   - Convert to lowercase
   - Remove URLs, emails, numbers
   - Remove punctuation and extra whitespace

2. **Tokenization & Lemmatization**:
   - Tokenize text using NLTK
   - Remove stopwords
   - Lemmatize words using WordNetLemmatizer

### Feature Engineering

- **TF-IDF Vectorization**:
  - Maximum features: 10,000
  - N-gram range: (1, 2)
  - Minimum document frequency: 2
  - Maximum document frequency: 95%

### Model Training

- **Algorithm**: Logistic Regression
- **Train/Test Split**: 80/20
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Cross-validation**: Stratified split

### KD-Tree Implementation

- **Library**: scipy.spatial.cKDTree
- **Purpose**: Fast similarity search
- **Query Method**: Euclidean distance-based nearest neighbors
- **Output**: Top-K similar articles with similarity scores

## 📱 Web Application Features

### 1. Text Classification Tab
- Input text area for article classification
- Sample text options for quick testing
- Real-time category prediction with confidence scores
- Probability distribution visualization
- Category-specific color coding

### 2. Similar Articles Tab
- Query text input for similarity search
- Configurable number of recommendations (5-20)
- Minimum similarity score filtering
- Detailed article cards with metadata
- Similarity score distribution analysis

### 3. Model Performance Tab
- Dataset statistics and visualizations
- Category distribution charts
- Text length analysis by category
- Sample articles from each category
- Model performance metrics

## 🎨 User Interface

- **Modern Design**: Clean, responsive layout
- **Interactive Visualizations**: Plotly charts and graphs
- **Color-coded Categories**: Visual category identification
- **Real-time Feedback**: Instant results and loading indicators
- **Mobile-friendly**: Responsive design for all devices

## 📈 Performance Metrics

The model achieves high performance on the BBC News dataset:

- **Accuracy**: ~95%+ on test set
- **Precision/Recall**: Balanced across all categories
- **F1-Score**: High performance for multi-class classification
- **Similarity Search**: Sub-second response times

## 🔍 Usage Examples

### Text Classification

```python
from train_model import BBCNewsClassifier

# Load trained model
classifier = BBCNewsClassifier()
classifier.load_models()

# Classify new text
text = "Apple reports record quarterly revenue driven by iPhone sales"
category, confidence = classifier.predict_category(text)
print(f"Category: {category}, Confidence: {confidence:.2%}")
```

### Similar Articles Search

```python
# Find similar articles
similar_articles = classifier.find_similar_articles(text, k=10)

for article in similar_articles:
    print(f"Title: {article['title']}")
    print(f"Category: {article['category']}")
    print(f"Similarity: {article['similarity_score']:.3f}")
```

## 🛠️ Customization

### Model Parameters

You can customize the model by modifying parameters in `train_model.py`:

```python
# TF-IDF parameters
vectorizer = TfidfVectorizer(
    max_features=10000,    # Adjust vocabulary size
    ngram_range=(1, 2),    # Change n-gram range
    min_df=2,              # Minimum document frequency
    max_df=0.95            # Maximum document frequency
)

# Classifier parameters
classifier = LogisticRegression(
    random_state=42,
    max_iter=1000,         # Maximum iterations
    C=1.0                  # Regularization strength
)
```

### Web App Customization

Modify `app.py` to:
- Add new visualization types
- Change color schemes
- Add additional features
- Customize the layout

## 📚 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **nltk**: Natural language processing
- **streamlit**: Web application framework
- **plotly**: Interactive visualizations
- **joblib**: Model serialization
- **scipy**: Scientific computing (KD-Tree)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- BBC News for providing the dataset
- scikit-learn team for the machine learning tools
- Streamlit team for the web framework
- NLTK team for natural language processing tools

## 📞 Support

For questions or issues, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

---

**Happy Text Classification! 🎉**
