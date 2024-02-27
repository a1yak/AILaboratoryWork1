import pandas as pd
import nltk
import PySimpleGUI as sg
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import ctypes

# Set window size to occupy full screen
user32 = ctypes.windll.user32
screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('job_description.csv')

# Drop duplicates
df = df.drop_duplicates()

# Text preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    if isinstance(text, str):
        tokens = word_tokenize(text.lower())  # Tokenization and convert to lowercase
        tokens = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        tokens = [stemmer.stem(word) for word in tokens]  # Stemming
        return ' '.join(tokens)
    else:
        return ''

df['processed_text'] = df['Description'].apply(preprocess_text)

# Define PySimpleGUI layout
layout = [
    [sg.Text('Enter search query:'), sg.InputText(key='-SEARCH-'), sg.Button('Search')],
    [sg.Text('Click the button to perform clustering')],
    [sg.Button('Cluster')],
    [sg.Multiline('', size=(120, 40), key='-OUTPUT-')]  # Multiline element to display results
]

# Create the window
window = sg.Window('Document Clustering', layout, size=(screen_width, screen_height))

# Event loop
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break
    elif event == 'Cluster':
        # Calculate TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=5)  # You can adjust the number of clusters as needed
        kmeans.fit(tfidf_matrix)

        # Assign documents to clusters
        document_clusters = kmeans.predict(tfidf_matrix)

        # Create categories based on clusters
        categories = {f"Category {i+1}": [] for i in range(len(set(document_clusters)))}

        # Assign documents to categories
        for document_idx, cluster_idx in enumerate(document_clusters):
            categories[f"Category {cluster_idx+1}"].append(document_idx)

        # Calculate top keywords for each document
        feature_names = tfidf_vectorizer.get_feature_names_out()
        category_keywords = {category: set() for category in categories}
        encountered_keywords = set()

        for category, document_indices in categories.items():
            # Calculate top keywords for documents in this category
            top_keywords_counter = Counter()
            for doc_idx in document_indices:
                doc = tfidf_matrix[doc_idx]
                top_indices = doc.indices[doc.data.argsort()[-5:]]  # Get indices of top 5 TF-IDF values
                top_keywords = [feature_names[idx] for idx in top_indices]
                top_keywords_counter.update(top_keywords)

            # Assign top keywords to this category, ensuring maximum 5 and uniqueness
            for keyword, count in top_keywords_counter.items():
                if count == 1 and len(category_keywords[category]) < 5 and keyword not in encountered_keywords:
                    category_keywords[category].add(keyword)
                    encountered_keywords.add(keyword)

        # Display results in the PySimpleGUI window
        output_text = ''
        for category, keywords in category_keywords.items():
            output_text += f"{category}: {', '.join(keywords)}\n" + '-'*40 + '\n'

            output_text += "Sample Documents:\n" + '-'*40 + '\n'
            # Display up to 5 documents for each category
            for idx, doc_idx in enumerate(categories[category][:5], 1):  # Limit to the first 5 documents
                document = df.loc[doc_idx, 'Description']
                if pd.notna(document):
                    output_text += f"{idx}) {document}\n\n"

        window['-OUTPUT-'].update(output_text)

    elif event == 'Search':
        search_query = values['-SEARCH-']
        if search_query:
            # Preprocess the search query
            search_query_processed = preprocess_text(search_query)

            # Convert search query to TF-IDF vector
            search_vector = tfidf_vectorizer.transform([search_query_processed])

            # Calculate cosine similarity between search query and documents
            similarities = cosine_similarity(search_vector, tfidf_matrix).flatten()

            # Sort documents by similarity
            sorted_indices = similarities.argsort()[::-1]

            # Display top 5 most similar documents
            output_text = ''
            for idx, doc_idx in enumerate(sorted_indices[:5], 1):
                document = df.loc[doc_idx, 'Description']
                similarity_score = similarities[doc_idx]
                if pd.notna(document):
                    output_text += f"{idx}) {document}\nSimilarity: {similarity_score:.2f}\n\n"
            window['-OUTPUT-'].update(output_text)

# Close the window
window.close()
