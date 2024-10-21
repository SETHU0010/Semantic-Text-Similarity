# Semantic Text Similarity (STS) Test

This is a **Streamlit** web application that measures the **semantic similarity** between two sentences using **TF-IDF** and **cosine similarity**. The app provides two options for input: uploading an Excel file with sentence pairs or entering the sentences manually.

## 🚀 Features

- **Semantic Similarity Measurement**: Calculate similarity between two sentences using TF-IDF and cosine similarity.
- **Similarity Categories**: 
  - Matched: 70% to 100%
  - Moderate Review: 65% to 69.9%
  - Need Review: 50% to 64.9%
  - Significant Review: 30% to 49.9%
  - Not Matched: Below 30%
- **Batch Processing**: Handles large datasets in batches for efficiency.
- **Downloadable Results**: Download the results as an Excel file.
- **Manual Input**: Option to input sentences manually for one-on-one comparison.
- **Error Handling**: Includes logging for error tracking and smooth user experience.

## 🛠️ How it Works

The application uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical features and **cosine similarity** to measure the similarity between those features.

### Categories of Semantic Deviation
- **Matched**: 70% and above
- **Moderate Review**: 65% to 69.9%
- **Need Review**: 50% to 64.9%
- **Significant Review**: 30% to 49.9%
- **Not Matched**: Below 30%

## 📄 How to Use

### 1. Upload an Excel File
- Upload an Excel file with two columns containing the sentence pairs.
- Click on "Calculate Similarity" to get the similarity score and the semantic deviation for each pair.

### 2. Manual Input
- Enter two sentences manually.
- Click on "Calculate Similarity" to get the similarity score and category.

### 3. Download Results
- After the calculation, you can download the results as an Excel file.
