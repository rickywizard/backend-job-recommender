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
    jobs_df = pd.read_csv(filepath, sep=';', encoding='cp1252')
    jobs_df['Requirement'] = jobs_df['Requirement'].fillna('')
    jobs_df['Description'] = jobs_df['Description'].fillna('')
    jobs_df['Combined'] = jobs_df['Requirement'] + ' ' + jobs_df['Description']
    jobs_df['Combined'] = jobs_df['Combined'].apply(preprocess_text)
    jobs_df = jobs_df.drop_duplicates(subset=['Combined'])
    return jobs_df

jobs_df = load_and_process_data('./AllJob.csv')

@app.route('/', methods=['GET'])
def home():
    return 'Jobs dataset is loaded and processed.'

@app.route('/get-recommendation', methods=['POST'])
def get_recommendation():
    data = request.get_json()
    skills = data.get('skills', [])
    projects = data.get('projectDesc', [])
    experiences = data.get('experiences', [])
    
    input_text = ' '.join(skills + projects + experiences)
    input_text = preprocess_text(input_text)
    
    corpus = jobs_df['Combined'].tolist()
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
    
    # Get the top 5 unique recommendations
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    seen_descriptions = set()
    top_jobs = []
    
    for i, _ in sim_scores:
        job_description = jobs_df.iloc[i]['Combined']
        if job_description not in seen_descriptions:
            top_jobs.append(i)
            seen_descriptions.add(job_description)
        if len(top_jobs) >= 5:
            break
    
    recommended_jobs = jobs_df.iloc[top_jobs]
    recommended_jobs_indices = recommended_jobs.index.tolist()
    
    # Get all jobs excluding the recommended ones
    all_jobs = jobs_df.drop(recommended_jobs_indices)
    
    # Combine recommended jobs at the top with the rest of the jobs
    combined_jobs = pd.concat([recommended_jobs, all_jobs])
    
    # Drop unwanted columns
    combined_jobs = combined_jobs.drop(columns=['Unnamed: 7', 'Combined'], errors='ignore')
    
    # Convert to dictionary
    all_jobs_dict = combined_jobs.to_dict(orient='records')
    
    return jsonify(all_jobs_dict)

if __name__ == '__main__':
    app.run(debug=True)