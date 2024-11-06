from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_and_process_data, preprocess_text
import nltk

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*', 'methods': ['POST']}})

nltk.download('punkt')
nltk.download('stopwords')

jobs_df = load_and_process_data('./dataset/AllJobCSV.csv')

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
    
    recommended_jobs = jobs_df.iloc[top_jobs][['Company', 'Position']]
    
    # Convert to dictionary
    top_jobs_dict = recommended_jobs.to_dict(orient='records')
    
    return jsonify(top_jobs_dict)

if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0', port=5000)