# 🔍 Semantic Text Similarity (STS) Test with Deviation Categories

This project is a **Streamlit** app that calculates **semantic similarity** between two sentences using various similarity metrics: **Semantic Similarity** (via `SentenceTransformer`), **Jaccard Similarity**, and **Cosine Similarity**. It also assigns a **Deviation Category** to each comparison based on predefined thresholds.

## ✨ Features
- 📁 **Upload Excel files** containing two columns of sentences for similarity calculation.
- 📈 Computes **Semantic Similarity**, **Jaccard Similarity**, and **Cosine Similarity** for each sentence pair.
- 📊 Provides **Mean Similarity** scores and categorizes results into **Deviation Categories** such as Matched, Need Review, etc.
- 💾 Option to **download results** as an Excel file.

## ⚙️ Installation
1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/SETHU0010/semantic-text-similarity-app.git
    ```

2. Navigate to the project directory:
    ```bash
    cd semantic-text-similarity-app
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage
To run the Streamlit app locally:

1. Execute the following command in the terminal:
    ```bash
    streamlit run app.py
    ```

2. Open your browser and go to the address:
    ```bash
    http://localhost:8501
    ```

3. Upload an Excel file containing two columns: `Sentence1` and `Sentence2` to compare.

4. The app will calculate the similarity scores and display the results. You can download the results as an Excel file.

## 📊 Similarity Metrics
The app uses the following metrics to calculate the similarity between two sentences:

1. **Semantic Similarity** (via `SentenceTransformer`): 
   - Utilizes a pre-trained transformer model to assess semantic meaning similarity between two sentences.

2. **Jaccard Similarity**: 
   - Measures similarity by comparing the sets of words in the sentences, calculated as:
     \[
     \text{Jaccard Similarity} = \frac{|A \cap B|}{|A \cup B|}
     \]
     Where A and B represent the sets of words from the two sentences.

3. **Cosine Similarity** (TF-IDF): 
   - Represents the sentences as vectors in a multi-dimensional space and computes the cosine of the angle between them.

## 📏 Deviation Categories
Based on the calculated **Mean Similarity** percentage, the app categorizes sentence comparisons into the following deviation categories:

| 🎯 Range (Percentage)  | 📌 Category              |
|------------------------|--------------------------|
| ✅ 85% to 100%          | Matched                  |
| 🟡 70% to 84.99%        | Need Review              |
| 🟠 50% to 69.99%        | Moderate Review          |
| 🔴 25% to 49.99%        | Significant Review       |
| ❌ 0% to 24.99%         | Not Matched              |

## 🧮 Algorithm
The app follows these steps to calculate similarity and categorize sentences:

1. **Input**: Read sentence pairs from the uploaded Excel file.
2. **Preprocessing**: Clean and preprocess sentences (lowercasing, removing punctuation, and stopwords).
3. **Similarity Calculation**:
   - Compute **Semantic Similarity** using a Sentence Transformer.
   - Calculate **Jaccard Similarity** based on word sets.
   - Measure **Cosine Similarity** using TF-IDF vectors.
4. **Mean Similarity**: Calculate the mean of the similarity scores.
5. **Deviation Categorization**: Assign a deviation category based on the mean similarity percentage.
6. **Output**: Display results and provide an option to download them as an Excel file.
