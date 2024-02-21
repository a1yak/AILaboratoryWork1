import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


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

# Keyword Identification
# Get feature names (terms) from TF-IDF vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Find top keywords (terms) for each document
top_keywords_per_document = []
for i in range(tfidf_matrix.shape[0]):  # Iterate over each document
    doc = tfidf_matrix[i]
    top_indices = doc.indices[doc.data.argsort()[-5:]]  # Get indices of top 5 TF-IDF scores
    top_keywords = [feature_names[idx] for idx in top_indices]  # Get corresponding keywords
    top_keywords_per_document.append(top_keywords)

# Categorization using KMeans clustering
num_clusters = 5  # Define the number of clusters/groups
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
federation_data = df[['Description']]
kmeans.fit(tfidf_matrix)

# Assign cluster labels to dataset elements
df['cluster_label'] = kmeans.labels_

# Search Functionality
def search_by_federation(federation):
    # Fill NaN values with an empty string
    df['Description'].fillna('', inplace=True)
    # Search for documents containing the specified federation
    matching_documents = df[df['Description'].str.contains(federation, case=False)]
    return matching_documents

# Example usage of search function
search_results = search_by_federation('support')
print(search_results)