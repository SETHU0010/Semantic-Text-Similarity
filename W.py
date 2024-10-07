import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
import nltk

# Download NLTK data (Stopwords)
nltk.download('stopwords')

# Load the pre-trained model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Upload the Excel file
st.title("Semantic Text Similarity (STS) Test with Additional Metrics")

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Function to calculate similarity metrics
def calculate_similarities(row):
    sentence1 = preprocess_text(row['Sentence1'])
    sentence2 = preprocess_text(row['Sentence2'])
    
    # Semantic Similarity (STS) with SentenceTransformer
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    semantic_similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

    # Jaccard Similarity
    set1 = set(sentence1.split())
    set2 = set(sentence2.split())
    jaccard_similarity = len(set1.intersection(set2)) / len(set1.union(set2))

    # Cosine Similarity (TF-IDF)
    vectorizer = TfidfVectorizer().fit_transform([sentence1, sentence2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)[0, 1]
    
    return pd.Series([semantic_similarity, jaccard_similarity, cosine_sim])

# Function to categorize deviation based on percentage
def categorize_deviation(percentage):
    if percentage >= 85:
        return "Matched"
    elif 70 <= percentage < 85:
        return "Need Review"
    elif 50 <= percentage < 70:
        return "Moderate Review"
    elif 25 <= percentage < 50:
        return "Significant Review"
    else:
        return "Not Matched"

# Upload Excel file
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)
    
    # Rename columns based on your file structure
    df.columns = ['Sentence1', 'Sentence2']
    
    # Calculate all similarity metrics
    df[['Semantic Similarity', 'Jaccard Similarity', 'Cosine Similarity']] = df.apply(calculate_similarities, axis=1)
    
    # Calculate the mean of Jaccard Similarity and Cosine Similarity
    df['Mean Similarity'] = df[['Jaccard Similarity', 'Cosine Similarity']].mean(axis=1)
    
    # Convert Mean Similarity to percentage
    df['Mean Similarity (%)'] = df['Mean Similarity'] * 100
    
    # Categorize deviation based on the Mean Similarity percentage
    df['Deviation'] = df['Mean Similarity (%)'].apply(categorize_deviation)
    
    # Format Mean Similarity as a percentage with two decimal places (optional)
    df['Mean Similarity (%)'] = df['Mean Similarity (%)'].apply(lambda x: f'{x:.2f}%')
    
    # Display the dataframe with similarity scores
    st.write(df)
    
    # Download the result as a CSV file
    st.download_button(
        label="Download Result as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='semantic_similarity_results_with_deviation.csv',
        mime='text/csv'
    )
