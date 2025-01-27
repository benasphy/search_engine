import re
import string
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import requests

# Download NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# Initialize lemmatizer and stopwords
lemmer = WordNetLemmatizer()
stopwords_list = stopwords.words("english")
custom_stopwords = stopwords_list + [
    "things", "that's", "something", "take", "don't", "may", "want", "you're",
    "set", "might", "says", "including", "lot", "much", "said", "know", "good",
    "step", "often", "going", "thing", "things", "think", "back", "actually",
    "better", "look", "find", "right", "example", "verb", "verbs"
]

# Helper function to preprocess text
def preprocess_text(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\s{2,}", " ", text)  # Remove multiple spaces
    return " ".join([lemmer.lemmatize(word) for word in text.split() if word not in custom_stopwords])

# Function to extract text from uploaded files
def extract_text_from_file(file):
    if file.name.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        return " ".join([page.extract_text() for page in pdf_reader.pages])
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return " ".join([paragraph.text for paragraph in doc.paragraphs])
    return None

# Function to scrape text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([para.get_text() for para in paragraphs])
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

# Streamlit UI setup
st.title("Enhanced Search System: Files and Websites")
st.write("Upload files (PDFs or DOCX) or provide URLs to search for similar content.")

# Input section: files and URLs
uploaded_files = st.file_uploader("Upload Files (PDF/DOCX)", accept_multiple_files=True)
website_links = st.text_area("Enter website links (one per line)")

# Process uploaded documents and URLs
documents = []
titles = []

if uploaded_files:
    for file in uploaded_files:
        extracted_text = extract_text_from_file(file)
        if extracted_text:
            documents.append(preprocess_text(extracted_text))
            titles.append(file.name)

if website_links:
    for link in website_links.strip().split("\n"):
        extracted_text = extract_text_from_url(link)
        if extracted_text:
            documents.append(preprocess_text(extracted_text))
            titles.append(link)

# Ensure documents are available
if documents:
    # TF-IDF Vectorizer setup
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),  # Use unigrams and bigrams
        max_features=10000,
        stop_words=custom_stopwords
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Search query input
    query = st.text_input("Enter your search query:")
    num_results = st.slider("Number of results to display:", min_value=1, max_value=10, value=5)

    if query:
        # Preprocess and vectorize the query
        query_vector = vectorizer.transform([preprocess_text(query)]).toarray().reshape(-1)

        # Calculate cosine similarity
        similarity_scores = {}
        for i, doc_vector in enumerate(tfidf_matrix.toarray()):
            score = np.dot(doc_vector, query_vector) / (np.linalg.norm(doc_vector) * np.linalg.norm(query_vector))
            similarity_scores[i] = score

        # Rank results
        ranked_results = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:num_results]

        # Display results
        st.subheader("Search Results")
        for rank, (index, score) in enumerate(ranked_results):
            if score > 0:
                st.write(f"### Rank {rank + 1}")
                st.write(f"**Title:** {titles[index]}")
                st.write(f"**Similarity Score:** {score:.2f}")
                st.write(f"**Content Snippet:** {documents[index][:500]}...")
                st.write("-" * 50)
else:
    st.info("Please upload files or provide URLs to search.")
