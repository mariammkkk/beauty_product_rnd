import pandas as pd
import numpy as np
import random
from bertopic import BERTopic

# load BERTopic model
def load_bertopic_model(model_path="bertopic_model"):
    try:
        model = BERTopic.load(model_path)
        print("BERTopic model loaded successfully for topic name retrieval.")
        return model
    except Exception as e:
        print(f"Error loading BERTopic model: {e}")
        return None

# in millions USD based on R&D complexity
def simulate_cost_of_fix(topic_name):
    if any(keyword in topic_name.lower() for keyword in ['texture', 'irritation', 'scent', 'formula', 'longevity', 'skin', 'chemical']):
        return round(random.uniform(3.0, 7.0), 2) 
    elif any(keyword in topic_name.lower() for keyword in ['packaging', 'leakage', 'applicator', 'breakage', 'bottle']):
        return round(random.uniform(1.5, 3.5), 2)
    elif any(keyword in topic_name.lower() for keyword in ['price', 'cost', 'money', 'value', 'availability']):
        return round(random.uniform(0.5, 2.0), 2)
    else:
        return round(random.uniform(1.0, 3.0), 2)


# priority ranking algorithm for each complaint topic
def run_phase_3(filepath='phase2_topic_tagged_data.csv'):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Phase 2 output file not found at {filepath}.")
        return None

    topic_model = load_bertopic_model()
    if topic_model is None:
        print("Cannot proceed with accurate cost calculation. Exiting.")
        return None

    print("\n--- Starting Phase 3: Priority Ranking ---")
    
    # filter out unassigned topics
    df_topics = df[df['Topic_ID'] != -1].copy()
    
    # calculate frequency and average sentiment per topic
    topic_metrics = df_topics.groupby('Topic_ID').agg(
        Frequency=('Topic_ID', 'size'), 
        Avg_Sentiment=('sentiment_compound', 'mean')
    ).reset_index()

    # map topic names
    topic_info = topic_model.get_topic_info()
    topic_name_map = topic_info.set_index('Topic').to_dict()['Name']

    topic_metrics['Topic_Name'] = topic_metrics['Topic_ID'].map(topic_name_map)
    
    topic_metrics['Topic_Name'] = topic_metrics['Topic_Name'].apply(
        lambda x: " ".join(str(x).split('_')[1:])
    )

    topic_metrics['Cost_of_Fix_M'] = topic_metrics['Topic_Name'].apply(simulate_cost_of_fix) 
    
    # calculate priority score
    topic_metrics['Priority_Score'] = (
            (topic_metrics['Frequency'] * topic_metrics['Avg_Sentiment'].clip(upper=0).abs()) / 
            topic_metrics['Cost_of_Fix_M']
        )
    
    # rank topics by priority score
    df_ranking = topic_metrics.sort_values(by='Priority_Score', ascending=False)
    df_ranking['Priority_Score'] = df_ranking['Priority_Score'].round(2)
    
    # top 5 topics
    print("\nPriority Ranking Complete. Top 5 Topics:")
    print(df_ranking[['Topic_Name', 'Frequency', 'Avg_Sentiment', 'Cost_of_Fix_M', 'Priority_Score']].head())
    
    output_path = 'phase3_priority_ranking.csv'
    df_ranking.to_csv(output_path, index=False)
    print(f"\nPriority ranking saved to '{output_path}'. Phase 3 complete.")
    return df_ranking

if __name__ == '__main__':
    run_phase_3()