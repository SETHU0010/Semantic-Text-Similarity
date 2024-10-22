import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from io import BytesIO
import time
import logging

# Set up logging for error tracking
logging.basicConfig(level=logging.ERROR)

# Load pre-trained SBERT model (optimized for semantic similarity)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to categorize semantic deviation based on similarity percentage
def categorize_semantic_deviation(similarity_percentage):
    if similarity_percentage >= 80:
        return "Matched"
    elif similarity_percentage >= 70:
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

# Function to calculate similarity using SBERT and cosine similarity
def calculate_similarity_bert(sentence1, sentence2):
    embeddings = model.encode([sentence1, sentence2])
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    similarity_percentage = similarity_score * 100
    return similarity_score, similarity_percentage

# Main function to define app layout
def main():
    st.title("Semantic Text Similarity (STS) Test with SBERT")

    # Use Tabs for smoother navigation
    tab1, tab2, tab3 = st.tabs(["Home", "Upload Data", "Manual Input"])

    with tab1:
        st.markdown("""
        <h2 style='font-size:28px;'>Semantic Similarity (Using SBERT)</h2>
        <p style='font-size:16px;'>Measures how similar two sentences are in meaning using SBERT (Sentence-BERT) and cosine similarity.</p>
        <ul style='font-size:16px;'>
            <li><strong>Matched:</strong> 80% to 100%</li>
            <li><strong>Moderate Review:</strong> 65% to 79.9%</li>
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
                                _, similarity_percentage = calculate_similarity_bert(sentence1, sentence2)

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
                        file_name="similarity_results_sbert.xlsx",
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
                    similarity_score, similarity_percentage = calculate_similarity_bert(sentence1, sentence2)

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
                        file_name="manual_similarity_result_sbert.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                except Exception as e:
                    logging.error(f"Error calculating similarity: {e}")
                    st.error(f"Error calculating similarity: {e}")
            else:
                st.error("Please enter both sentences.")

    st.markdown("---")
    st.write("### About this App")
    st.write("This app uses SBERT (Sentence-BERT) and cosine similarity to calculate the semantic similarity between two sentences.")

if __name__ == "__main__":
    main()
