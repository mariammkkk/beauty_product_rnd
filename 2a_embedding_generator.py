import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import CountVectorizer
import os

# loads, samples, and returns cleaned data from Phase 1
def load_phase1_data(filepath='phase1_cleaned_data.csv'):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Loaded ALL {len(df):,} reviews.")

        sample_size = int(len(df) * 0.25)
        df_sampled = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        print(f"Sampling for stability: Using {len(df_sampled):,} reviews (25% sample).")
        
        return df_sampled
        
    except FileNotFoundError:
        print(f"Error: Phase 1 output file not found at {filepath}.")
        return None

if __name__ == '__main__':
    
    # paths
    embeddings_path = 'embeddings.npy'
    data_output_path = 'phase2_data_for_clustering.csv'

    # check if embeddings already exist
    if os.path.exists(embeddings_path) and os.path.exists(data_output_path):
        print("Embeddings already exist. Skipping the generation step. Ready for clustering.")
        exit()
    df = load_phase1_data()
    
    if df is not None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating embeddings (this will take a WHILEE)...")
        
        documents = df['final_text'].tolist()
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedding_model.encode(documents, show_progress_bar=True)
        np.save(embeddings_path, embeddings)
        df[['product_id', 'review_text', 'sentiment_compound', 'final_text']].to_csv(data_output_path, index=False)
    
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Embeddings complete. Ready for clustering.")