import pandas as pd
import numpy as np
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk


nltk.download('punkt') # for tokenization
nltk.download('wordnet') # for lemmatization
nltk.download('stopwords') # for stopword removal

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()

# load, concatenate review files, and merge with product info
def load_and_merge_data(data_folder='data'):    
    review_files = [
        os.path.join(data_folder, f) 
        for f in os.listdir(data_folder) 
        if f.startswith('reviews_') and f.endswith('.csv')
    ]
    product_file = os.path.join(data_folder, 'product_info.csv')

    print(f"Loading {len(review_files)} review files...")
    list_of_dfs = []
    
    # combine all review csv files
    for file in review_files:
        try:
            df = pd.read_csv(file)
            list_of_dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
            
    if not list_of_dfs:
        raise FileNotFoundError("No review files were successfully loaded.")
        
    df_reviews_combined = pd.concat(list_of_dfs, ignore_index=True)
    
    # load product metadata
    print("Loading Product Metadata...")
    df_products = pd.read_csv(product_file)

    # join review text/rating with product details
    df_master = pd.merge(
        df_reviews_combined, 
        df_products[['product_id', 'brand_name', 'price_usd', 'ingredients', 'primary_category']],
        on='product_id', 
        how='left' 
    )
    print(f"Data Merged. Total rows: {len(df_master):,}")
    return df_master

# cleaning & filtering functions: removes punctuation, and standardizes case
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# tokenization, stop-word removal, lemmatization
def lemmatize_and_remove_stopwords(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
        
    # try NLTK tokenization
    try:
        word_list = nltk.word_tokenize(text)
    except LookupError:
        word_list = text.split() 
    
    # lemmatization and stop-word removal
    lemmatized_output = [
        lemmatizer.lemmatize(w) 
        for w in word_list 
        if w not in stop_words and len(w) > 1
    ]
    
    return " ".join(lemmatized_output)

# use VADER to assign scores - ranges from -1 to +1
def get_sentiment_score(text):
    return analyzer.polarity_scores(text)['compound']

def run_phase_1():
    df_master = load_and_merge_data()
    
    print("\n--- Starting Phase 1: NLP Cleaning & Sentiment Analysis ---")

    # filter to negative reviews (rating <= 3 stars)
    df_phase1_output = df_master[df_master['rating'] <= 3].copy()
    print(f"Filtered to {len(df_phase1_output):,} negative reviews (rating <= 3 stars).")
    
    # handle missing text/product ID
    original_count = len(df_phase1_output)
    df_phase1_output.dropna(subset=['review_text', 'product_id'], inplace=True)
    print(f"Removed {original_count - len(df_phase1_output)} rows with missing critical data.")

    # cleaning steps
    df_phase1_output['clean_review'] = df_phase1_output['review_text'].apply(clean_text)
    
    # lemmatization and stop-word removal
    df_phase1_output['final_text'] = df_phase1_output['clean_review'].apply(lemmatize_and_remove_stopwords)
    print("Text cleaning and lemmatization applied.")
    
    # filter out rows where 'final_text' is empty after cleaning
    df_phase1_output = df_phase1_output[df_phase1_output['final_text'].str.len() > 0]

    # sentiment scoring
    df_phase1_output['sentiment_compound'] = df_phase1_output['clean_review'].apply(get_sentiment_score)
    
    print(f"Phase 1 Complete. Final dataset size: {len(df_phase1_output):,}")
    
    # save the output file
    output_path = 'phase1_cleaned_data.csv'
    df_phase1_output.to_csv(output_path, index=False)
    print(f"\nCleaned and scored data saved to '{output_path}'. Phase 1 is complete.")
    return df_phase1_output

if __name__ == '__main__':
    run_phase_1()