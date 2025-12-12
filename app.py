import pandas as pd
import streamlit as st
import numpy as np


st.set_page_config(layout="wide", page_title="Customer Feedback Strategy Dashboard")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("phase4_final_dashboard_data.csv")
    except FileNotFoundError:
        st.error("Error: 'phase4_final_dashboard_data.csv' not found. Please ensure the file is in the same directory.")
        st.stop()
        
    df['Value_to_Cost_Ratio'] = df.apply(
        lambda row: row['Projected_Gain_M'] / row['Cost_of_Fix_M'] 
        if row['Cost_of_Fix_M'] != 0 else np.inf, axis=1
    )
    return df

df = load_data()

# calculate global metrics
total_frequency = df['Frequency'].sum()
total_cost = df['Cost_of_Fix_M'].sum()
total_gain = df['Projected_Gain_M'].sum()
total_net_impact = df['Net_Impact_M'].sum()

global_roi = total_gain / total_cost if total_cost != 0 else np.inf

# sentiment summary
def get_sentiment_summary(data):
    NEGATIVE_THRESHOLD = -0.10
    POSITIVE_THRESHOLD = 0.10
    
    negative_topics = data[data['Avg_Sentiment'] <= NEGATIVE_THRESHOLD]
    positive_topics = data[data['Avg_Sentiment'] >= POSITIVE_THRESHOLD]
    neutral_topics = data[(data['Avg_Sentiment'] > NEGATIVE_THRESHOLD) & (data['Avg_Sentiment'] < POSITIVE_THRESHOLD)]
    
    return {
        'negative_count': len(negative_topics),
        'neutral_count': len(neutral_topics),
        'positive_count': len(positive_topics),
        'total_topics': len(data),
        'top_negative': negative_topics.sort_values(by='Avg_Sentiment', ascending=True).head(5)
    }

# generate executive summary narrative
def generate_executive_summary_narrative(total_net_impact, global_roi, df):
    top_roi_topic = df.sort_values(by='Value_to_Cost_Ratio', ascending=False).iloc[0]
    top_threat = df.sort_values(by='Avg_Sentiment', ascending=True).iloc[0]

    summary = f"""
    The comprehensive analysis of customer feedback reveals a total untapped financial opportunity of **\${total_net_impact:,.2f} million** across all identified topics. The overall investment strategy maintains a healthy **{global_roi:,.2f}:1 Global ROI**.
    
    Key strategic focus areas must balance absolute value with brand risk mitigation:
    
    * **Most Efficient Investment (High ROI):** The topic '{top_roi_topic['Topic_Name']}' offers an exceptional **{top_roi_topic['Value_to_Cost_Ratio']:,.2f}:1 ROI** and a substantial net impact of \${top_roi_topic['Net_Impact_M']:,.2f}M. This should be prioritized for immediate action due to efficiency.
    * **Most Urgent Brand Threat (Sentiment Risk):** Addressing '{top_threat['Topic_Name']}' is critical for immediate brand health, as it currently carries the most severe negative sentiment ({top_threat['Avg_Sentiment']:+.3f}).
    """
    return summary

# streamlit app sections
def display_welcome_section(summary_narrative):
    st.title("💡 Customer Feedback: Strategy Formulation Dashboard")
    st.markdown("## Welcome to Your Strategic Roadmap!")
    st.markdown(summary_narrative)
    st.markdown("---")


def display_kpi_header(total_net_impact, total_cost, global_roi):
    st.header("1. Strategic Investment Potential: Key Performance Indicators")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Net Impact (Overall Opportunity)",
            value=f"${total_net_impact:,.2f}M",
            delta="Maximum Potential Profit across all solutions"
        )
    
    with col2:
        st.metric(
            label="Total Estimated Cost of Fix",
            value=f"${total_cost:,.2f}M",
            delta="Total Required Investment"
        )
    
    with col3:
        roi_value = "∞" if np.isinf(global_roi) else f"{global_roi:,.2f}:1"
        st.metric(
            label="Global Value-to-Cost Ratio (ROI)",
            value=roi_value,
            delta="Value Generated Per $1 Spent (Overall)",
            delta_color="normal"
        )
    st.markdown("---")


def display_key_insights(data):
    st.header("2. Key Insights: Efficiency (ROI) and Urgency (Sentiment)")

    st.subheader("2.A. High ROI Opportunities (Investment Efficiency)")
    st.markdown("These topics offer the **highest Value-to-Cost Ratio**, representing "
                "projects where a small investment yields a massive projected return.")
    
    finite_roi_data = data[~np.isinf(data['Value_to_Cost_Ratio'])].copy()
    top_5_roi = finite_roi_data.sort_values(by='Value_to_Cost_Ratio', ascending=False).head(5)
    
    top_5_roi['ROI (Value:Cost)'] = top_5_roi['Value_to_Cost_Ratio'].apply(lambda x: f"{x:,.2f}:1")
    top_5_roi['Net Impact (M)'] = top_5_roi['Net_Impact_M'].apply(lambda x: f"${x:,.2f}M")
    top_5_roi.rename(columns={'Topic_Name': 'Topic Driver', 'Recommended_Action_Type': 'Action'}, inplace=True)

    st.dataframe(top_5_roi[['Topic Driver', 'ROI (Value:Cost)', 'Net Impact (M)', 'Action']], 
                 use_container_width=True, hide_index=True)

    sentiment_data = get_sentiment_summary(data)
    
    st.subheader("2.B. Customer Loyalty & Risk Insights (Brand Urgency)")
    st.markdown(f"**Overall Sentiment Health Check:** Out of {sentiment_data['total_topics']} topics, "
                f"**{sentiment_data['negative_count']}** are Negative ($\le -0.10$), and **{sentiment_data['positive_count']}** are Positive ($\ge +0.10$).")

    st.markdown("**Top 5 Brand Threats (Most Negative Sentiment):** Addressing these is crucial for immediate risk mitigation.")
    
    top_negative = sentiment_data['top_negative'].copy()
    top_negative['Avg Sentiment'] = top_negative['Avg_Sentiment'].apply(lambda x: f"{x:+.3f}")
    top_negative['Net Impact (M)'] = top_negative['Net_Impact_M'].apply(lambda x: f"${x:,.2f}M")
    
    st.dataframe(top_negative[['Topic_Name', 'Avg Sentiment', 'Net Impact (M)']], 
                 use_container_width=True, hide_index=True)

    st.markdown("---")


def display_roadmap_table(data):
    st.header("3. 🎯 Full Actionable Roadmap: All Topics")
    st.markdown(
        """
        The table below provides **all scores and actions** for every topic, ranked by **Priority Score**. 
        Use the filters in the sidebar to segment the data and find your next high-impact project.
        """
    )
    
    st.sidebar.header("Data Segmentation Filters")
    
    # filter by action type
    action_options = ['All Action Types'] + sorted(data['Recommended_Action_Type'].unique().tolist())
    selected_action = st.sidebar.selectbox(
        '1. Filter by Recommended Action Type:',
        options=action_options,
        index=0
    )
    
    # filter sliders
    min_net_impact_val = data['Net_Impact_M'].min()
    max_net_impact_val = data['Net_Impact_M'].max()
    net_impact_threshold = st.sidebar.slider(
        '2. Minimum Net Impact (M):',
        min_value=float(min_net_impact_val),
        max_value=float(max_net_impact_val),
        value=float(min_net_impact_val),
        step=0.5,
        format='%.2f'
    )
    
    min_sentiment = data['Avg_Sentiment'].min()
    max_sentiment = data['Avg_Sentiment'].max()
    sentiment_threshold = st.sidebar.slider(
        '3. Max. Average Sentiment (Filter out too positive topics):',
        min_value=float(min_sentiment),
        max_value=float(max_sentiment),
        value=float(max_sentiment), 
        step=0.01
    )
    
    min_freq = data['Frequency'].min()
    max_freq = data['Frequency'].max()
    frequency_threshold = st.sidebar.slider(
        '4. Minimum Topic Frequency (Volume):',
        min_value=int(min_freq),
        max_value=int(max_freq),
        value=int(min_freq),
        step=100
    )
    
    filtered_data = data.copy()
    
    # filter 1: action type
    if selected_action != 'All Action Types':
        filtered_data = filtered_data[filtered_data['Recommended_Action_Type'] == selected_action]
        
    # filter 2: net impact
    filtered_data = filtered_data[filtered_data['Net_Impact_M'] >= net_impact_threshold]
    
    # filter 3: sentiment 
    filtered_data = filtered_data[filtered_data['Avg_Sentiment'] <= sentiment_threshold]
    
    # filter 4: frequency
    filtered_data = filtered_data[filtered_data['Frequency'] >= frequency_threshold]
    
    data_for_display = filtered_data.sort_values(by='Priority_Score', ascending=False)
    
    data_for_display['Frequency'] = data_for_display['Frequency'].apply(lambda x: f"{x:,.0f}")
    data_for_display['Avg_Sentiment'] = data_for_display['Avg_Sentiment'].apply(lambda x: f"{x:+.3f}")
    data_for_display['Cost_of_Fix_M'] = data_for_display['Cost_of_Fix_M'].apply(lambda x: f"${x:,.2f}M")
    data_for_display['Projected_Gain_M'] = data_for_display['Projected_Gain_M'].apply(lambda x: f"${x:,.2f}M")
    data_for_display['Net_Impact_M'] = data_for_display['Net_Impact_M'].apply(lambda x: f"${x:,.2f}M")
    data_for_display['Value_to_Cost_Ratio'] = data_for_display['Value_to_Cost_Ratio'].apply(lambda x: "∞" if np.isinf(x) else f"{x:,.2f}:1")
    
    # rename columns for readability
    data_for_display.rename(columns={
        'Topic_Name': 'Topic Driver',
        'Avg_Sentiment': 'Avg Sentiment',
        'Priority_Score': 'Priority Score',
        'Cost_of_Fix_M': 'Cost (M)',
        'Projected_Gain_M': 'Projected Gain (M)',
        'Net_Impact_M': 'Net Impact (M)',
        'Value_to_Cost_Ratio': 'ROI (Value:Cost)',
        'Recommended_Action_Type': 'Recommended Action',
        'Business_Summary': 'Business Summary'
    }, inplace=True)

    final_columns = [
        'Topic Driver', 'Frequency', 'Avg Sentiment', 'Priority Score', 
        'Net Impact (M)', 'ROI (Value:Cost)', 'Cost (M)', 'Projected Gain (M)',
        'Recommended Action', 'Business Summary'
    ]
    
    st.dataframe(data_for_display[final_columns], use_container_width=True)
    
    st.sidebar.markdown(f"---")
    st.sidebar.info(f"Showing **{len(data_for_display)}** out of {len(data)} total topics based on current filters.")


def display_logic_audit():
    st.header("4. 🔬 Calculation & Logic Audit: Transparency in Findings")
    st.markdown(
        """
        This section provides the mathematical rationale behind the strategic recommendations, ensuring stakeholders can fully audit the data.
        """
    )

    st.subheader("A. Core Financial Metrics")
    st.markdown("""
    The financial value of each topic is derived from the following formulas:
    """)
    
    col_metric, col_formula = st.columns([1, 2])
    
    with col_metric:
        st.markdown("**Net Impact (M)**")
        st.markdown("**Value-to-Cost Ratio (ROI)**")
    
    with col_formula:
        st.latex(r"""
        \text{Net Impact} = \text{Projected\_Gain\_M} - \text{Cost\_of\_Fix\_M}
        """)
        st.latex(r"""
        \text{ROI} = \frac{\text{Projected\_Gain\_M}}{\text{Cost\_of\_Fix\_M}}
        """)
    
    st.subheader("B. Prioritization Logic (Priority Score)")
    st.markdown(
        """
        The **Priority Score** is the final metric used to rank topics. It is a weighted function that ensures investment is directed toward issues that are both **high-value (Net Impact)** and **urgent (negative Sentiment/high Frequency).**
        """
    )
    st.latex(r"""
        \text{Priority Score} \approx \text{Net Impact} \times (1 - \text{Avg Sentiment}) + \text{Frequency Factor}
    """)
    st.markdown("""
    ---
    ***Audit Insight:*** The highest ROI topics should be prioritized as they promise the most efficient return on investment. The filters allow users to balance ROI efficiency against absolute Net Impact value.
    """)

def main():
    summary_narrative = generate_executive_summary_narrative(total_net_impact, global_roi, df)
    display_welcome_section(summary_narrative)
    display_kpi_header(total_net_impact, total_cost, global_roi)
    display_key_insights(df)
    display_roadmap_table(df)
    display_logic_audit()

if __name__ == "__main__":
    main()