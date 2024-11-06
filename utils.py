import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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