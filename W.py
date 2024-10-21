import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import time

# Changes
import logging

# Set up logging for error tracking
logging.basicConfig(level=logging.ERROR)

# Function to categorize semantic deviation based on similarity percentage
def categorize_semantic_deviation(similarity_percentage):
    if similarity_percentage >= 70:
        return "Matched"
    elif similarity_percentage >= 65:
        return "Moderate Review"
    elif similarity_percentage >= 50:
        return "Need Review"
    elif similarity_percentage >= 30:
        return "Significant Review"
    else:
        return "Not Matched"

# Function to create a downloadable Excel file
def create_download_link(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Similarity Results')
    output.seek(0)
    return output

# Batch processing for large datasets
def batch_process(df, batch_size=100):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]

# Function to calculate similarity using TF-IDF and cosine similarity
def calculate_similarity(sentence1, sentence2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity_percentage = similarity_score[0][0] * 100
    return similarity_score[0][0], similarity_percentage

# Main function to define app layout
def main():
    st.title("Semantic Text Similarity (STS) Test")

    # Use Tabs for smoother navigation
    tab1, tab2, tab3 = st.tabs(["Home", "Upload Data", "Manual Input"])

    with tab1:
        st.markdown("""
        <h2 style='font-size:28px;'>Semantic Similarity (Using TF-IDF)</h2>
        <p style='font-size:16px;'>Measures how similar two sentences are in meaning using TF-IDF and cosine similarity.</p>
        <ul style='font-size:16px;'>
            <li><strong>Matched:</strong> 70% to 100%</li>
            <li><strong>Moderate Review:</strong> 65% to 69.9%</li>
            <li><strong>Need Review:</strong> 50% to 64.9%</li>
            <li><strong>Significant Review:</strong> 30% to 49.9%</li>
            <li><strong>Not Matched:</strong> Below 30%</li>
        </ul>
        """, unsafe_allow_html=True)

    with tab2:
        uploaded_file = st.file_uploader("Upload an Excel file with two columns", type=["xlsx"])

        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                if df.shape[1] < 2:
                    st.error("The uploaded file must contain at least two columns.")
                    return

                st.write("Uploaded Data:")
                st.dataframe(df)

                sentence1_col = df.columns[0]
                sentence2_col = df.columns[1]

                if st.button("Calculate Similarity", key="upload_button"):
                    results = []
                    progress_bar = st.progress(0)

                    # Process rows in batches
                    total_batches = len(df) / 100
                    for i, batch in enumerate(batch_process(df)):
                        for _, row in batch.iterrows():
                            sentence1 = row[sentence1_col]
                            sentence2 = row[sentence2_col]

                            if pd.isna(sentence1) or pd.isna(sentence2):
                                similarity_percentage = 0
                            else:
                                _, similarity_percentage = calculate_similarity(sentence1, sentence2)

                            results.append({
                                "Sentence 1": sentence1,
                                "Sentence 2": sentence2,
                                "Similarity Percentage": round(similarity_percentage, 2),
                                "Semantic Deviation": categorize_semantic_deviation(similarity_percentage)
                            })

                        progress_bar.progress(min((i + 1) / total_batches, 1.0))

                    results_df = pd.DataFrame(results)
                    st.write("Similarity Results:")
                    st.dataframe(results_df)

                    excel_data = create_download_link(results_df)
                    st.download_button(
                        label="Download Results as Excel",
                        data=excel_data,
                        file_name="similarity_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            except Exception as e:
                logging.error(f"Error processing file: {e}")
                st.error(f"Error processing file: {e}")

    with tab3:
        sentence1 = st.text_area("Enter the first sentence:")
        sentence2 = st.text_area("Enter the second sentence:")

        if st.button("Calculate Similarity", key="manual_button"):
            if sentence1 and sentence2:
                try:
                    similarity_score, similarity_percentage = calculate_similarity(sentence1, sentence2)

                    st.write(f"**Similarity Score:** {similarity_score:.4f}")
                    st.write(f"**Similarity Percentage:** {similarity_percentage:.2f}%")
                    st.write(f"**Semantic Deviation:** {categorize_semantic_deviation(similarity_percentage)}")

                    result_data = [{
                        "Sentence 1": sentence1,
                        "Sentence 2": sentence2,
                        "Similarity Score": round(similarity_score, 4),
                        "Similarity Percentage": round(similarity_percentage, 2),
                        "Semantic Deviation": categorize_semantic_deviation(similarity_percentage)
                    }]
                    results_df = pd.DataFrame(result_data)

                    excel_data = create_download_link(results_df)
                    st.download_button(
                        label="Download Result as Excel",
                        data=excel_data,
                        file_name="manual_similarity_result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                except Exception as e:
                    logging.error(f"Error calculating similarity: {e}")
                    st.error(f"Error calculating similarity: {e}")
            else:
                st.error("Please enter both sentences.")

    st.markdown("---")
    st.write("### About this App")
    st.write("This app uses TF-IDF and cosine similarity to calculate the semantic similarity between two sentences.")

if __name__ == "__main__":
    main()
