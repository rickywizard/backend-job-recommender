from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests

app = Flask(__name__)
CORS(app, resources={r'/predict': {'origins': 'http://127.0.0.1:3000', 'methods': ['POST']}})

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def load_and_process_data(filepath):
    jobs_df = pd.read_csv(filepath)
    jobs_df['Description'] = jobs_df['Description'].fillna('')
    jobs_df['Description'] = jobs_df['Description'].apply(preprocess_text)
    return jobs_df

jobs_df = load_and_process_data("./JobsDataset.csv")

@app.route('/', methods=['GET'])
def home():
    return "Jobs dataset is loaded and processed."

@app.route('/get-recommendation', methods=['POST'])
def get_recommendation():
    data = request.get_json()
    skills = data.get('skills', '')
    comparison_jobs = data.get('comparisonJobs', [])
    
    input_text = f"{skills}"
    input_text = preprocess_text(input_text)
    
    corpus = jobs_df['Description'].tolist()
    corpus.append(input_text)
    
    vectorizer = TfidfVectorizer(
        min_df=3,
        max_features=None,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 3),
        stop_words='english'
    )
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    jobs_df['Similarity'] = cosine_sim[0]
    recommended_jobs = jobs_df.sort_values(by='Similarity', ascending=False).head(10)

    job_titles = recommended_jobs['Job Title'].tolist()
    
    # Fetch the new corpus from the external API
    response = requests.get('http://localhost:3000/api/position')
    if response.status_code == 200:
        new_corpus = response.json()
    else:
        return jsonify({'error': 'Failed to fetch new corpus from external API'}), 500
    
    new_corpus_processed = [preprocess_text(job) for job in new_corpus]
    new_corpus_processed.append(preprocess_text(", ".join(job_titles)))
    
    tfidf_matrix_new = vectorizer.fit_transform(new_corpus_processed)
    cosine_sim_new = cosine_similarity(tfidf_matrix_new[-1], tfidf_matrix_new[:-1])
    
    similarity_scores = list(zip(new_corpus, cosine_sim_new[0]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_3_jobs = [job for job, score in similarity_scores[:3]]
    
    return jsonify(top_3_jobs)

if __name__ == '__main__':
    app.run(debug=True)
