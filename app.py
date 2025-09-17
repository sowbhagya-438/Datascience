"""
BBC News Text Classification and KD-Tree Recommendations
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from train_model import BBCNewsClassifier
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="BBC News Text Classification & Recommendations",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .article-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
    }
    .business { background-color: #e74c3c; }
    .tech { background-color: #3498db; }
    .politics { background-color: #2ecc71; }
    .sport { background-color: #f39c12; }
    .entertainment { background-color: #9b59b6; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load pre-trained models with caching"""
    try:
        classifier = BBCNewsClassifier()
        classifier.load_models()
        return classifier
    except FileNotFoundError:
        st.error("Models not found! Please run train_model.py first.")
        return None

def get_category_color(category):
    """Get color for category badge"""
    colors = {
        'business': 'business',
        'tech': 'tech', 
        'politics': 'politics',
        'sport': 'sport',
        'entertainment': 'entertainment'
    }
    return colors.get(category.lower(), 'business')

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📰 BBC News Text Classification & Recommendations</h1>', 
                unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        classifier = load_models()
    
    if classifier is None:
        st.stop()
    
    # Sidebar
    st.sidebar.markdown("## 🎯 About This App")
    st.sidebar.markdown("""
    This application demonstrates:
    - **Text Classification**: Predict news article categories
    - **Similarity Search**: Find similar articles using KD-Tree
    - **Real-time Analysis**: Process any text input instantly
    """)
    
    st.sidebar.markdown("## 📊 Dataset Statistics")
    if classifier.df_processed is not None:
        category_counts = classifier.df_processed['category'].value_counts()
        
        # Create pie chart
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Article Distribution by Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=300)
        st.sidebar.plotly_chart(fig, use_container_width=True)
        
        # Display counts
        for category, count in category_counts.items():
            st.sidebar.metric(f"{category.title()} Articles", count)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Text Classification", "📚 Similar Articles", "📈 Model Performance"])
    
    with tab1:
        st.markdown('<h2 class="section-header">Text Classification</h2>', 
                   unsafe_allow_html=True)
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Enter your news article text:")
            input_text = st.text_area(
                "Article Text",
                height=200,
                placeholder="Paste your news article text here...",
                help="Enter any news article text to classify its category"
            )
        
        with col2:
            st.markdown("### Sample Texts:")
            sample_texts = {
                "Business": "Apple Inc. reported record quarterly revenue driven by strong iPhone sales and services growth.",
                "Technology": "New AI breakthrough in machine learning promises to revolutionize data processing capabilities.",
                "Politics": "Government announces new policy changes affecting healthcare and education sectors.",
                "Sports": "Championship game ends in dramatic overtime victory with record-breaking attendance.",
                "Entertainment": "Hollywood blockbuster breaks box office records with stunning visual effects."
            }
            
            selected_sample = st.selectbox("Choose a sample:", list(sample_texts.keys()))
            if st.button("Use Sample"):
                input_text = sample_texts[selected_sample]
                st.rerun()
        
        # Classification
        if st.button("🔍 Classify Text", type="primary"):
            if input_text.strip():
                with st.spinner("Classifying text..."):
                    category, confidence = classifier.predict_category(input_text)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Predicted Category", category.title())
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Confidence", f"{confidence:.2%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        color_class = get_category_color(category)
                        st.markdown(f'<span class="category-badge {color_class}">{category.title()}</span>', 
                                   unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Confidence visualization
                    st.markdown("### Confidence Distribution")
                    categories = classifier.label_encoder.classes_
                    probabilities = classifier.classifier.predict_proba(
                        classifier.vectorizer.transform([classifier.preprocessor.preprocess([input_text])[0]])
                    )[0]
                    
                    prob_df = pd.DataFrame({
                        'Category': categories,
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=True)
                    
                    fig = px.bar(
                        prob_df, 
                        x='Probability', 
                        y='Category',
                        orientation='h',
                        title="Classification Probabilities",
                        color='Probability',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                st.warning("Please enter some text to classify.")
    
    with tab2:
        st.markdown('<h2 class="section-header">Similar Articles Recommendation</h2>', 
                   unsafe_allow_html=True)
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Enter text to find similar articles:")
            query_text = st.text_area(
                "Query Text",
                height=150,
                placeholder="Enter text to find similar articles...",
                key="query_text"
            )
        
        with col2:
            st.markdown("### Search Parameters:")
            num_recommendations = st.slider(
                "Number of Recommendations", 
                min_value=5, 
                max_value=10, 
                value=10
            )
            
            min_similarity = st.slider(
                "Minimum Similarity Score", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.3,
                step=0.1
            )
        
        # Find similar articles
        if st.button("🔍 Find Similar Articles", type="primary"):
            if query_text.strip():
                with st.spinner("Finding similar articles..."):
                    # Fetch up to 10 neighbors and display exactly the selected count (max 10)
                    top_n = int(min(num_recommendations, 10))
                    similar_articles = classifier.find_similar_articles(query_text, k=10)
                    
                    # Top-N closest articles (ignoring threshold)
                    top_articles = similar_articles[:top_n]
                    # Compute normalized weights that sum to 1
                    sim_vals = np.array([a.get('similarity_score', 0.0) for a in top_articles], dtype=float)
                    # Avoid negative/NaN; clip to [0,1]
                    sim_vals = np.clip(sim_vals, 0.0, 1.0)
                    sim_sum = float(sim_vals.sum()) if float(sim_vals.sum()) > 0 else 1.0
                    weights = sim_vals / sim_sum

                    st.markdown(f"### Top {top_n} Closest Articles (ignoring threshold)")
                    for i, article in enumerate(top_articles, 1):
                        with st.container():
                            st.markdown('<div class="article-card">', unsafe_allow_html=True)
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**{i}. {article['title']}**")
                                st.markdown(f"*{article['content']}*")
                            with col2:
                                color_class = get_category_color(article['category'])
                                st.markdown(f'<span class="category-badge {color_class}">{article["category"].title()}</span>', 
                                           unsafe_allow_html=True)
                                # Clamp similarity to [0, 1] for display stability
                                display_sim = max(0.0, min(1.0, float(article.get('similarity_score', 0.0))))
                                st.metric("Similarity", f"{display_sim:.3f}")
                                st.metric("Weight", f"{weights[i-1]:.3f}")
                            st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown("---")

                    # Filter by minimum similarity (user-selected) and cap to N (<=10)
                    filtered_articles = [article for article in similar_articles 
                                       if article['similarity_score'] >= min_similarity][:top_n]
                    
                    if filtered_articles:
                        st.markdown(f"### Top {len(filtered_articles)} Filtered Articles (respecting threshold)")

                        # Normalized weights for the filtered block
                        f_sim_vals = np.array([a.get('similarity_score', 0.0) for a in filtered_articles], dtype=float)
                        f_sim_vals = np.clip(f_sim_vals, 0.0, 1.0)
                        f_sum = float(f_sim_vals.sum()) if float(f_sim_vals.sum()) > 0 else 1.0
                        f_weights = f_sim_vals / f_sum
                        
                        for i, article in enumerate(filtered_articles, 1):
                            with st.container():
                                st.markdown('<div class="article-card">', unsafe_allow_html=True)
                                
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"**{i}. {article['title']}**")
                                    st.markdown(f"*{article['content']}*")
                                
                                with col2:
                                    color_class = get_category_color(article['category'])
                                    st.markdown(f'<span class="category-badge {color_class}">{article["category"].title()}</span>', 
                                               unsafe_allow_html=True)
                                    display_sim = max(0.0, min(1.0, float(article.get('similarity_score', 0.0))))
                                    st.metric("Similarity", f"{display_sim:.3f}")
                                    st.metric("Weight", f"{f_weights[i-1]:.3f}")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Similarity distribution
                        st.markdown("### Similarity Score Distribution")
                        similarity_scores = [max(0.0, min(1.0, float(article['similarity_score']))) for article in filtered_articles]
                        
                        fig = px.histogram(
                            x=similarity_scores,
                            nbins=10,
                            title="Distribution of Similarity Scores",
                            labels={'x': 'Similarity Score', 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.warning(f"No articles found with similarity score >= {min_similarity}")
                        
            else:
                st.warning("Please enter some text to find similar articles.")
    
    with tab3:
        st.markdown('<h2 class="section-header">Model Performance & Analysis</h2>', 
                   unsafe_allow_html=True)
        
        # Model information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Articles", len(classifier.df_processed))
        
        with col2:
            st.metric("Categories", len(classifier.label_encoder.classes_))
        
        with col3:
            st.metric("Features", classifier.vectorizer.max_features)
        
        # Category distribution
        st.markdown("### Article Distribution by Category")
        category_counts = classifier.df_processed['category'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Articles per Category",
                color=category_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_title="Category", yaxis_title="Number of Articles")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Category Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Text length analysis
        st.markdown("### Text Length Analysis")
        classifier.df_processed['text_length'] = classifier.df_processed['content'].str.len()
        
        fig = px.box(
            classifier.df_processed,
            x='category',
            y='text_length',
            title="Article Length Distribution by Category",
            color='category'
        )
        fig.update_layout(xaxis_title="Category", yaxis_title="Text Length (characters)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample articles
        st.markdown("### Sample Articles from Each Category")
        for category in classifier.df_processed['category'].unique():
            st.markdown(f"#### {category.title()}")
            sample_article = classifier.df_processed[
                classifier.df_processed['category'] == category
            ].iloc[0]
            
            st.markdown(f"**Title:** {sample_article['title']}")
            st.markdown(f"**Content:** {sample_article['content'][:300]}...")
            st.markdown("---")

if __name__ == "__main__":
    main()
