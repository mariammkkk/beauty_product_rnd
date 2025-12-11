import pandas as pd
import numpy as np
import random

# create action type mapping based on topic keywords
def assign_action_type(topic_name):
    topic = topic_name.lower()
    
    if any(keyword in topic for keyword in ['irritation', 'scent', 'texture', 'longevity', 'formula', 'chemical']):
        return 'R&D: Formulation Review'
    elif any(keyword in topic for keyword in ['packaging', 'leakage', 'applicator', 'breakage', 'bottle']):
        return 'Packaging & Supplier Review'
    elif any(keyword in topic for keyword in ['smudging', 'color', 'accuracy', 'consistency', 'wear']):
        return 'Quality Control/QA Review'
    elif any(keyword in topic for keyword in ['price', 'cost', 'value', 'sale', 'availability']):
        return 'Marketing/Pricing Strategy'
    elif any(keyword in topic for keyword in ['shipping', 'delivery', 'logistics', 'tracking']):
        return 'Logistics/Operational Fix'
    else:
        return 'General Review'

# simulate projected gain based on topic frequency - more frequent topics have a higher potential revenue gain from fixing
def simulate_projected_gain(topic_frequency):
    scaling_factor = 0.005
    base_gain = topic_frequency * scaling_factor
    noise = random.uniform(0.5, 1.5) 
    projected_gain = max(1.0, base_gain * noise)
    
    return round(projected_gain, 2)

def run_phase_4(filepath='phase3_priority_ranking.csv'):

    # load phase 3 output    
    try:
        df_ranking = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Phase 3 output file not found at {filepath}.")
        return None

    print("\n--- Starting Phase 4: Financial Modeling & Action Mapping ---")

    # financial modeling 
    df_ranking['Projected_Gain_M'] = df_ranking['Frequency'].apply(simulate_projected_gain)
    
    # net impact = projected gain - cost of fix
    df_ranking['Net_Impact_M'] = df_ranking['Projected_Gain_M'] - df_ranking['Cost_of_Fix_M']

    # action mapping
    df_ranking['Recommended_Action_Type'] = df_ranking['Topic_Name'].apply(assign_action_type)
    
    # final output sorted by priority score
    df_final = df_ranking[[
        'Topic_ID', 
        'Topic_Name', 
        'Frequency', 
        'Avg_Sentiment',
        'Cost_of_Fix_M', 
        'Projected_Gain_M', 
        'Net_Impact_M',
        'Priority_Score', 
        'Recommended_Action_Type'
    ]].sort_values(by='Priority_Score', ascending=False)
    
    # save final dashboard data
    output_path = 'phase4_final_dashboard_data.csv'
    df_final.to_csv(output_path, index=False)
    print(f"Phase 4 Complete. Final dashboard data saved to '{output_path}'.")
    
    return df_final

if __name__ == '__main__':
    run_phase_4()