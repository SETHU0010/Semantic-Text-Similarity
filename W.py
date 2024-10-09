import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
import nltk
from io import BytesIO  # Import BytesIO for in-memory file handling

# Download NLTK data (Stopwords)
nltk.download('stopwords')

# Load the pre-trained model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Function to calculate similarity metrics
def calculate_similarities(sentence1, sentence2):
    sentence1 = preprocess_text(sentence1)
    sentence2 = preprocess_text(sentence2)
    
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
    
    return semantic_similarity, jaccard_similarity, cosine_sim

# Function to categorize deviation based on Semantic Similarity percentage
def categorize_semantic_similarity(percentage):
    # Adjusting thresholds to reflect more logical segmentation
    if percentage >= 86:
        return "Matched"
    elif 70 <= percentage < 85.99:
        return "Need Review"
    elif 50 <= percentage < 70:
        return "Moderate Review"
    elif 25 <= percentage < 50:
        return "Significant Review"
    else:
        return "Not Matched"

# Main function to define app layout
def main():
    st.title("Text Similarity Analysis and Categorization")
    
    # Main layout with two columns
    col1, col2 = st.columns(2)

    with col1:
        st.header("Navigation")
        options = [
            "Home", "Upload Data", "Manual Input"
        ]
        choice = st.radio("Go to", options)
    
    if choice == "Home":
        # Home Page Content
        st.markdown("""
        <h2 style='font-size:28px;'>Semantic Similarity</h2>
        <p style='font-size:16px;'>Measures how similar two sentences are in meaning using models (e.g., Sentence Transformers).</p>

        <h2 style='font-size:28px;'>Jaccard Similarity</h2>
        <p style='font-size:16px;'>Compares two sets by dividing the size of their intersection by the size of their union.</p>
        
        <p><strong>Formula:</strong>  
        Jaccard = \\(\\frac{|A \cap B|}{|A \cup B|}\\)</p>

        <h2 style='font-size:28px;'>Cosine Similarity</h2>
        <p style='font-size:16px;'>Measures the cosine of the angle between two vectors in a multi-dimensional space.</p>

        <p><strong>Formula:</strong>  
        Cosine = \\(\\frac{A \cdot B}{||A|| \, ||B||}\\)</p>

        <h2 style='font-size:28px;'>Mean Similarity</h2>
        <p style='font-size:16px;'>Average of Semantic, Jaccard, and Cosine Similarity scores.</p>

        <p><strong>Formula:</strong>  
        Mean Similarity = \\(\\frac{\text{Semantic} + \text{Jaccard} + \text{Cosine}}{3}\\)</p>

        <h2 style='font-size:28px;'>Percentage Metrics</h2>
        <ul style='font-size:16px;'>
        <li><strong>Mean Similarity (%):</strong> Mean similarity expressed as a percentage.</li>
        <li><strong>Semantic Similarity (%):</strong> Semantic similarity score as a percentage.</li>
        </ul>

        <h2 style='font-size:28px;'>Semantic Deviation Categories</h2>
        <ul style='font-size:16px;'>
        <li><strong>Matched:</strong> 86% to 100%</li>
        <li><strong>Need Review:</strong> 70% to 85.99%</li>
        <li><strong>Moderate Review:</strong> 50% to 69.99%</li>
        <li><strong>Significant Review:</strong> 25% to 49.99%</li>
        <li><strong>Not Matched:</strong> 0% to 24.99%</li>
        </ul>
        """, unsafe_allow_html=True)
    
    elif choice == "Upload Data":
        # Upload Excel file
        uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

        if uploaded_file:
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
            
            # Rename columns based on your file structure
            df.columns = ['Sentence1', 'Sentence2']
            
            # Calculate all similarity metrics
            similarities = df.apply(lambda row: calculate_similarities(row['Sentence1'], row['Sentence2']), axis=1)
            df[['Semantic Similarity', 'Jaccard Similarity', 'Cosine Similarity']] = pd.DataFrame(similarities.tolist(), index=df.index)
            
            # Convert Semantic Similarity to percentage
            df['Semantic Similarity (%)'] = df['Semantic Similarity'] * 100
            
            # Categorize deviation based on the Semantic Similarity percentage
            df['Semantic Deviation'] = df['Semantic Similarity (%)'].apply(categorize_semantic_similarity)
            
            # Calculate the mean of Semantic Similarity, Jaccard Similarity, and Cosine Similarity
            df['Mean Similarity'] = df[['Semantic Similarity', 'Jaccard Similarity', 'Cosine Similarity']].mean(axis=1)
            
            # Convert Mean Similarity to percentage
            df['Mean Similarity (%)'] = df['Mean Similarity'] * 100
            
            # Format Mean Similarity and Semantic Similarity as percentages with two decimal places
            df['Mean Similarity (%)'] = df['Mean Similarity (%)'].apply(lambda x: f'{x:.2f}%')
            df['Semantic Similarity (%)'] = df['Semantic Similarity (%)'].apply(lambda x: f'{x:.2f}%')

            # Reorder columns
            df = df[['Sentence1', 'Sentence2', 'Semantic Similarity', 'Jaccard Similarity', 
                     'Cosine Similarity', 'Mean Similarity', 'Mean Similarity (%)', 
                     'Semantic Similarity (%)', 'Semantic Deviation']]
            
            # Display the dataframe with similarity scores and deviation
            st.subheader("Similarity Results:")
            st.write(df)
            
            # Save the DataFrame to an in-memory buffer
            output = BytesIO()
            df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)  # Rewind the buffer

            # Download the result as an Excel file
            st.download_button(
                label="Download Result as Excel",
                data=output,
                file_name='semantic_similarity_results_with_deviation.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            st.warning("Please upload an Excel file to proceed.")
    
    elif choice == "Manual Input":
        st.subheader("Manual Input for Sentence Similarity")
        sentence1 = st.text_input("Enter the first sentence:")
        sentence2 = st.text_input("Enter the second sentence:")
        
        if st.button("Calculate Similarity"):
            if sentence1 and sentence2:
                # Ensure the input isn't just whitespace
                if sentence1.strip() and sentence2.strip():
                    # Calculate similarity scores
                    semantic_similarity, jaccard_similarity, cosine_similarity_score = calculate_similarities(sentence1, sentence2)

                    # Convert to percentage
                    semantic_similarity_pct = semantic_similarity * 100
                    mean_similarity = (semantic_similarity + jaccard_similarity + cosine_similarity_score) / 3
                    mean_similarity_pct = mean_similarity * 100

                    # Categorize based on Semantic Similarity percentage
                    deviation_category = categorize_semantic_similarity(semantic_similarity_pct)

                    # Display results
                    st.write(f"**Semantic Similarity:** {semantic_similarity_pct:.2f}%")
                    st.write(f"**Jaccard Similarity:** {jaccard_similarity:.2f}")
                    st.write(f"**Cosine Similarity:** {cosine_similarity_score:.2f}")
                    st.write(f"**Mean Similarity:** {mean_similarity_pct:.2f}%")
                    st.write(f"**Semantic Deviation Category:** {deviation_category}")
                else:
                    st.warning("Please enter valid sentences (non-whitespace).")
            else:
                st.warning("Please enter both sentences to calculate similarity.")

if __name__ == "__main__":
    main()
