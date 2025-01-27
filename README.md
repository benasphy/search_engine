# **Dynamic Document Similarity Search**

A powerful Streamlit application that allows users to search for content within uploaded files (PDF/DOCX) and websites. The app leverages TF-IDF and cosine similarity to provide highly relevant search results based on user queries. 

---

## **Features**

- **File Upload Support**: Upload multiple PDF and DOCX files for text extraction and search.
- **Website Integration**: Input URLs to scrape and include text content from websites.
- **Text Preprocessing**: Automatically cleans and preprocesses text (e.g., removes stopwords, punctuation, and irrelevant content).
- **Search Engine**: Uses TF-IDF vectorization and cosine similarity to rank documents based on relevance.
- **Customizable Results**: Adjust the number of search results displayed with a slider.
- **Streamlit UI**: Clean and interactive user interface for seamless user experience.

---

## **Technologies Used**

- **Python Libraries**:
  - `Streamlit` for the web interface.
  - `NLTK` for text tokenization and stopword removal.
  - `Scikit-learn` for TF-IDF vectorization and cosine similarity.
  - `PyPDF2` for extracting text from PDF files.
  - `python-docx` for extracting text from Word documents.
  - `BeautifulSoup` for scraping text from websites.
  - `Requests` for handling web requests.

---

## **Installation**

Follow these steps to run the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/benasphy/search_engine.git
   cd search_engine
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Resources**:
   Open a Python shell and run:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

## **How to Use**

1. **Upload Files**: Upload one or more PDF or DOCX files.
2. **Enter Website Links**: Provide URLs (one per line) to fetch and analyze text from websites.
3. **Search Query**: Enter your search query in the input box.
4. **View Results**: Adjust the number of results displayed and explore the ranked content snippets.

---


## **Example Use Cases**

- **Academic Research**: Quickly search through research papers or articles to find relevant content.
- **Business**: Analyze reports and documents for specific insights.
- **Web Scraping**: Search and compare content scraped from multiple websites.

---

## **File Structure**

```plaintext
.
├── app.py                  # Main Streamlit app
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation


---

## **Future Improvements**

- Add support for additional file types (e.g., TXT, CSV).
- Implement advanced search features like fuzzy matching.
- Enable keyword-based filters for more targeted search results.
- Optimize web scraping for faster processing.

---

## **Contributing**

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add a new feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

---

