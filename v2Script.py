import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

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

# Print the count of documents in each category along with the list of documents
for category, document_indices in categories.items():
    documents = df.loc[document_indices, 'Description'].tolist()
    print(f"{category} ({len(documents)} documents): {documents}")
