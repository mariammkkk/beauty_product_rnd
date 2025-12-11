import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

@st.cache_data
def load_data(filepath='phase4_final_dashboard_data.csv'):
    try:
        df = pd.read_csv(filepath)

        numeric_cols = ['Frequency', 'Avg_Sentiment', 'Cost_of_Fix_M', 'Projected_Gain_M', 'Net_Impact_M', 'Priority_Score']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        total_low_rated_reviews = df['Frequency'].sum()
        high_priority_topics = len(df[df['Priority_Score'] >= df['Priority_Score'].quantile(0.8)]) # Top 20%
        avg_negative_sentiment = df['Avg_Sentiment'].abs().mean()
        
        return df, total_low_rated_reviews, high_priority_topics, avg_negative_sentiment
    except FileNotFoundError:
        st.error("Data file 'phase4_final_dashboard_data.csv' not found. Please run the data_processor, topic_modeling, and priority_ranking scripts first.")

        return pd.DataFrame(), 0, 0, 0.0

df_final, total_low_rated_reviews, high_priority_topics, avg_negative_sentiment = load_data()

st.set_page_config(
    page_title="R&D Insight: Product Prioritization Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #FAF9F6;
    }
    .stMetric {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1 {
        color: #2C2C2C;
        font-weight: 300;
        letter-spacing: 1px;
    }
    h2, h3 {
        color: #4A4A4A;
        font-weight: 300;
    }
    .stDataFrame {
        background-color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background-color: #F5F5F3;
    }
    </style>
    """, unsafe_allow_html=True)

# header
st.markdown("""
    <div style="text-align: center; padding: 2rem 0; border-bottom: 1px solid #E8E8E8; margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0.5rem;">✨ R&D Insight: Product Prioritization Dashboard</h1>
        <p style="color: #8B8B8B; font-size: 0.9rem; margin-top: 0;">Customer Review Text Analysis | Phase 2-4 Integration</p>
    </div>
    """, unsafe_allow_html=True)

# sidebar filters
st.sidebar.header("🔍 Filters")
st.sidebar.markdown("---")

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="color: #8B8B8B; font-size: 0.85rem; padding: 1rem;">
        <p><strong>Data Source:</strong> Sephora Review Analysis (Kaggle)</p>
        <p><strong>Last Analysis Run:</strong> {}</p>
    </div>
    """.format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)

if df_final.empty:
    st.warning("Dashboard data is empty. Please run the full ETL pipeline (data_processor.py -> topic_modeling.py -> priority_ranking.py -> financial_modeling.py) to generate the input file.")
else:

    st.markdown("### Key Performance Indicators")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    
    total_projected_revenue_retained = df_final['Projected_Gain_M'].sum()
    
    with col1:
        st.metric(
            label="Total Negative Review Volume (Rating $\le3$)",
            value=f"{total_low_rated_reviews:,}",
        )
    
    with col2:
        st.metric(
            label="High-Priority Topics Identified (Top 20% Rank)",
            value=high_priority_topics,
        )
    
    with col3:
        st.metric(
            label="Average Negative Sentiment Intensity (0 to -1)",
            value=f"-{avg_negative_sentiment:.2f}",
            delta_color="off"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- R&D Prioritization Chart ---
    st.markdown("### R&D Prioritization Analysis")
    st.markdown("*Top Topics Ranked by Cost-Adjusted Priority Score*")
    st.markdown("---")
    
    # horizontal bar chart
    df_prioritization = df_final.sort_values(by='Priority_Score', ascending=True).tail(10)
    
    fig_prioritization = px.bar(
        df_prioritization,
        x='Priority_Score',
        y='Topic_Name',
        orientation='h',
        color='Priority_Score',
        color_continuous_scale=['#E8E8E8', '#D4A574', '#B8956A'],
        text='Priority_Score',
        labels={'Priority_Score': 'Priority Score', 'Topic_Name': 'Complaint Theme (Topic Modeling Output)'}
    )
    
    fig_prioritization.update_traces(
        texttemplate='%{text:.1f}',
        textposition='outside',
        marker_line_color='#FFFFFF',
        marker_line_width=1
    )
    
    fig_prioritization.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FAF9F6',
        font=dict(color='#4A4A4A', size=11),
        height=500,
        showlegend=False,
        xaxis=dict(
            title_font=dict(size=12, color='#4A4A4A'),
            gridcolor='#F0F0F0',
            range=[0, df_prioritization['Priority_Score'].max() * 1.2]
        ),
        yaxis=dict(
            title_font=dict(size=12, color='#4A4A4A'),
            categoryorder='total ascending'
        ),
        margin=dict(l=150, r=50, t=20, b=50)
    )
    
    st.plotly_chart(fig_prioritization, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Financial Impact Modeling ---
    st.markdown("### Financial Impact Modeling")
    st.markdown("*Business Justification (Cost vs. Gain) for Top Topics*")
    st.markdown("---")
    
    col_fin1, col_fin2 = st.columns([1, 2])
    
    with col_fin1:
        st.markdown("#### Total Projected Annual Revenue Retained")
        st.markdown(f"""
            <div style="background-color: #FFFFFF; padding: 2rem; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <h1 style="color: #B8956A; margin: 0; font-size: 3rem;">${total_projected_revenue_retained:.1f}M</h1>
                <p style="color: #8B8B8B; margin-top: 0.5rem;">Annual Projection (Estimated from LTV Retention)</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_fin2:
        st.markdown("#### Cost vs. Projected Gain Analysis")
        
        df_financial = df_final.head(8).sort_values(by='Priority_Score', ascending=False)
        topics_for_financial = df_financial['Topic_Name'].tolist()
        cost_data = df_financial['Cost_of_Fix_M'].tolist()
        gain_data = df_financial['Projected_Gain_M'].tolist()
        net_impact = df_financial['Net_Impact_M'].tolist()
        
        fig_financial = go.Figure()
        
        fig_financial.add_trace(go.Bar(
            name='Cost of Fix (Input)',
            x=topics_for_financial,
            y=[-c for c in cost_data],
            marker_color='#E8E8E8',
            text=[f'${c:.1f}M' for c in cost_data],
            textposition='outside'
        ))
        
        fig_financial.add_trace(go.Bar(
            name='Projected Gain (Output)',
            x=topics_for_financial,
            y=gain_data,
            marker_color='#B8956A',
            text=[f'${g:.1f}M' for g in gain_data],
            textposition='outside'
        ))
        
        fig_financial.add_trace(go.Scatter(
            name='Net Impact (ROI)',
            x=topics_for_financial,
            y=net_impact,
            mode='lines+markers',
            line=dict(color='#4A4A4A', width=2),
            marker=dict(size=8, color='#4A4A4A'),
            text=[f'${n:.1f}M' for n in net_impact],
            textposition='top center'
        ))
        
        fig_financial.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FAF9F6',
            font=dict(color='#4A4A4A', size=10),
            height=400,
            barmode='group',
            xaxis=dict(
                title='Topic',
                title_font=dict(size=11, color='#4A4A4A'),
                gridcolor='#F0F0F0',
                tickangle=-45
            ),
            yaxis=dict(
                title='Amount ($M)',
                title_font=dict(size=11, color='#4A4A4A'),
                gridcolor='#F0F0F0'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=50, b=100)
        )
        
        st.plotly_chart(fig_financial, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Prescriptive Action Mapping ---
    st.markdown("### Prescriptive Action Mapping")
    st.markdown("*Recommended Actions by Topic (Linked to Responsible Departments)*")
    st.markdown("---")
    
    df_actions = df_final[['Topic_Name', 'Recommended_Action_Type', 'Net_Impact_M']].copy()
    
    st.dataframe(
        df_actions.head(10),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Topic_Name": st.column_config.TextColumn(
                "Complaint Theme",
                width="medium"
            ),
            "Recommended_Action_Type": st.column_config.TextColumn(
                "Prescriptive Action (Department)",
                width="large"
            ),
            "Net_Impact_M": st.column_config.NumberColumn(
                "Net ROI ($M)",
                format="$%.1fM",
                width="small"
            )
        }
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #8B8B8B; font-size: 0.85rem; padding: 2rem 0;">
            <p>R&D Insight Dashboard | Generated from Customer Review Text Analysis</p>
            <p>Data visualization powered by Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
