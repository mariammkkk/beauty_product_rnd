import pandas as pd
import numpy as np
import umap
from sklearn.cluster import KMeans 
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
import os
import shutil

if __name__ == '__main__':
    
    # load data
    try:
        df = pd.read_csv('phase2_data_for_clustering.csv', low_memory=False)
        embeddings = np.load('embeddings.npy')
    except FileNotFoundError:
        print("Error: Required files ('embeddings.npy' or 'phase2_data_for_clustering.csv') not found.")
        exit()

    documents = df['final_text'].tolist()
    
    print("\n--- Starting Phase 2: Topic Clustering ---")
    
    # umamp - dimensionality reduction
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 1/3: Reducing Dimensionality (UMAP)...")
    umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings)

    # k-means clustering
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 2/3: Clustering Embeddings (K-Means)...")
    
    k_clusters = 50 
    kmeans_model = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto', verbose=0)
    
    df['Topic_ID'] = kmeans_model.fit_predict(reduced_embeddings) 

    # bertopic - topic modeling
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 3/3: Integrating Models and Extracting Topic Representation...")
    
    # initialize bertopic
    topic_model = BERTopic(
        umap_model=umap_model,
        cluster_model=kmeans_model,
        vectorizer_model=CountVectorizer(stop_words='english'),
        language="english", 
        calculate_probabilities=False, 
        verbose=True,
    )

    # fit and transform documents
    df['Topic_ID'], _ = topic_model.fit_transform(documents, embeddings)
    topic_info = topic_model.get_topic_info() 
      
    # print results
    print("\nTopic Clustering Complete.")
    print("Top Topic Summary:")
    print(topic_model.get_topic_info().head(10))

    # save to output file
    if os.path.exists("bertopic_model"):
        shutil.rmtree("bertopic_model")
    topic_model.save("bertopic_model", serialization="safetensors")
    
    # merge topic_id back to original df
    documents_for_ctfidf = pd.DataFrame({
        "Document": documents,
        "Topic": df['Topic_ID']
    })
    
    # calculate frequency of documents per topic
    topic_sizes = df.groupby('Topic_ID').size().to_dict()
    topic_model.topic_sizes_ = topic_sizes
    
    # calculate c-TF-IDF and topic representations
    topic_model.c_tf_idf_ = topic_model._c_tf_idf(documents_for_ctfidf)
    topic_model.topic_representations_ = topic_model._extract_topics(documents_for_ctfidf)
    
    # generate topic labels
    topic_model.topic_labels_ = topic_model.generate_topic_labels(
        topic_model.topic_representations_
    )
    
    topic_info = topic_model.get_topic_info() 
      
    # print results
    print("\nTopic Clustering Complete.")
    print("Top Topic Summary:")
    print(topic_model.get_topic_info().head(10))

    # save model
    if os.path.exists("bertopic_model"):
        shutil.rmtree("bertopic_model")
    topic_model.save("bertopic_model", serialization="safetensors") 
    
    # merge the topic id back onto the full Phase 1 df cols
    df_output = df[['product_id', 'sentiment_compound', 'Topic_ID']].merge(
        pd.read_csv('phase1_cleaned_data.csv', low_memory=False)[['product_id', 'brand_name', 'price_usd', 'ingredients']], 
        on='product_id', how='left'
    )

    # cleanup
    df_output['Topic_ID'] = df['Topic_ID']
    df_output['sentiment_compound'] = df['sentiment_compound']

    # save final output file
    output_path = 'phase2_topic_tagged_data.csv'
    df_output.to_csv(output_path, index=False)
    print(f"\nFinal topic-tagged data saved to '{output_path}'. Phase 2b complete.")