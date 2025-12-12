import pandas as pd
import os
import time
# 🚨 CHANGE 1: Import AuthenticationError to diagnose key issues
from openai import OpenAI, AuthenticationError 
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

# --- 1. Configuration and Data Loading ---

if not API_KEY:
    print("FATAL ERROR: OPENAI_API_KEY not found in environment. Check your .env file and ensure it is valid.")
    client = None
else:
    try:
        print(f"API Key loaded (starts with: {API_KEY[:4]}... )")
        # Client initialization is still correct:
        client = OpenAI(api_key=API_KEY) 
        print("OpenAI Client Initialized Successfully.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        client = None


INPUT_FILE = 'phase4_final_dashboard_data.csv'

# --- Load Data ---
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded data with {len(df)} rows and {len(df['Topic_Name'].unique())} unique topics.")
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found. Ensure Phase 4 scripts have run successfully.")
    df = pd.DataFrame()


# --- 2. Generation Function (Updated for better error diagnosis) ---

def generate_business_summary(topic_name: str) -> str:
    """
    Calls the OpenAI API to rephrase a technical topic name into a clear business summary.
    """
    
    if client is None:
        return f"[API ERROR: Client not initialized] {topic_name}"

    system_prompt = (
        "You are an expert market analyst and R&D communicator. "
        "Your task is to rephrase a technical, keyword-based topic name (e.g., 'break broke wanted love') "
        "from a customer review analysis into a clear, non-jargon, actionable business complaint summary. "
        "The output must be a single, concise sentence."
    )
    
    user_prompt = f"Rephrase the following topic name into a concise business summary: '{topic_name}'"
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3, 
            max_tokens=60
        )
        summary = response.choices[0].message.content.strip()
        return summary
        
    # 🚨 CHANGE 2: Catch AuthenticationError specifically for a clearer message
    except AuthenticationError as e:
        print(f"FATAL AUTH ERROR: API Key rejected for topic '{topic_name}'. Check key validity and billing. {e}")
        return f"[FATAL AUTH ERROR] {topic_name}"
    
    except Exception as e:
        print(f"API Error (Non-Auth/Quota) for topic '{topic_name}': {e}")
        return f"[API QUOTA/NETWORK ERROR] {topic_name}"


# --- 3. Processing and Saving ---

if not df.empty:
    
    unique_topics = df['Topic_Name'].unique()
    topic_summary_map = {}
    
    print(f"\n--- Starting Summary Generation for {len(unique_topics)} Unique Topics ---")
    start_time = datetime.now()
    
    for i, topic in enumerate(unique_topics):
        print(f"[{i+1}/{len(unique_topics)}] Processing: {topic}...")
        
        time.sleep(0.2) 
        
        summary = generate_business_summary(topic)
        topic_summary_map[topic] = summary
        
    end_time = datetime.now()
    print(f"\nGeneration Complete in {(end_time - start_time).total_seconds():.2f} seconds.")

    # --- Merge New Column Back to DataFrame ---
    df['Business_Summary'] = df['Topic_Name'].map(topic_summary_map)

    # --- Inspect Results ---
    print("\n--- Sample of Topic Transformations (First 5) ---")
    print(df[['Topic_Name', 'Business_Summary', 'Priority_Score']].head(5).to_markdown(index=False))
    
    # --- Final Save ---
    df.to_csv(INPUT_FILE, index=False)
    print(f"\nSUCCESS: Updated data saved back to {INPUT_FILE}.")
    print("If successful, you can now run the Streamlit dashboard (`dashboard_app.py`).")

else:
    print("\nProcessing skipped because the DataFrame is empty.")