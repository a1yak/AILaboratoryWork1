import pandas as pd
import nltk
import PySimpleGUI as sg
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

# Define PySimpleGUI layout
layout = [
    [sg.Text('Click the button to perform clustering')],
    [sg.Button('Cluster')],
    [sg.Multiline('', size=(60, 20), key='-OUTPUT-')],  # Multiline element to display results
    [sg.Text('Enter search query:'), sg.InputText(key='-SEARCH-'), sg.Button('Search')]
]

# Create the window
window = sg.Window('Document Clustering', layout)

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

        # Display results in the PySimpleGUI window
        output_text = ''
        for category, document_indices in categories.items():
            output_text += f"{category}\n" + '-'*40 + '\n'
            documents = df.loc[document_indices, 'Description'].tolist()
            output_text += '\n'.join(str(doc) if pd.notna(doc) else '' for doc in documents) + '\n\n'
        window['-OUTPUT-'].update(output_text)

    elif event == 'Search':
     search_query = values['-SEARCH-']
    if search_query:
        # Drop rows with NA / NaN values in the 'Description' column
        df.dropna(subset=['Description'], inplace=True)
        
        # Perform search on non-null 'Description' values
        search_results = df[df['Description'].str.contains(search_query, case=False)]
        
        # Update the output text
        output_text = ''
        for idx, row in search_results.iterrows():
            output_text += f"Search Results for '{search_query}':\n" + '-'*40 + '\n'
            output_text += row['Description'] + '\n\n'
        window['-OUTPUT-'].update(output_text)

# Close the window
window.close()
