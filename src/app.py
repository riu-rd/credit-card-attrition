import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Credit Card Attrition Dashboard",
    page_icon="💳",
    layout="wide"
)

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('src/datasets/final_cc_attrition.csv')
    return df

@st.cache_resource
def load_model():
    model = joblib.load('src/models/lightgbm_best.pkl')
    return model

# Main app
def main():
    # Title
    st.title("💳 Credit Card Customer Attrition Analysis")
    st.markdown("---")
    
    # Load data and model
    df = load_data()
    model = load_model()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Overview", "Data Explorer", "Individual Prediction"])
    
    if page == "Overview":
        show_overview(df)
    elif page == "Data Explorer":
        show_explorer(df)
    else:
        show_prediction(df, model)

def show_overview(df):
    st.header("📊 Overview")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    total_customers = len(df)
    avg_income = df['Income'].mean()
    avg_credit = df['CreditLimit'].mean()
    
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Avg Income", f"₱{avg_income:,.0f}")
    col3.metric("Avg Credit Limit", f"₱{avg_credit:,.0f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attrition by Tenure Group")
        tenure_map = {0: 'New (0-3y)', 1: 'Growing (3-6y)', 
                     2: 'Established (6-12y)', 3: 'Loyal (12+y)'}
        tenure_stats = df.groupby('TenureGroup_ordinal')['AttritionFlag'].mean() * 100
        tenure_stats.index = tenure_stats.index.map(tenure_map)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(range(len(tenure_stats)), tenure_stats.values)
        ax.set_xticks(range(len(tenure_stats)))
        ax.set_xticklabels(tenure_stats.index, rotation=45, ha='right')
        ax.set_ylabel('Attrition Rate (%)')
        ax.set_ylim(0, max(tenure_stats.values) * 1.2)
        
        for i, v in enumerate(tenure_stats.values):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Attrition by Card Type")
        card_map = {0: 'Silver', 1: 'Gold', 2: 'Platinum', 3: 'Black'}
        card_stats = df.groupby('CardType_ordinal')['AttritionFlag'].mean() * 100
        card_stats.index = card_stats.index.map(card_map)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(range(len(card_stats)), card_stats.values,
                      color=['#C0C0C0', '#FFD700', '#E5E4E2', '#333333'])
        ax.set_xticks(range(len(card_stats)))
        ax.set_xticklabels(card_stats.index)
        ax.set_ylabel('Attrition Rate (%)')
        ax.set_ylim(0, max(card_stats.values) * 1.2)
        
        for i, v in enumerate(card_stats.values):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Summary stats table
    st.markdown("---")
    st.subheader("Summary Statistics")
    
    summary_cols = ['Age', 'Income', 'CreditLimit', 'TotalTransactions', 'TotalSpend', 'Tenure']
    summary_df = df[summary_cols].describe().round(2)
    st.dataframe(summary_df)

def show_explorer(df):
    st.header("🔍 Data Explorer")
    
    # Feature selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        feature = st.selectbox(
            "Select Feature to Analyze",
            ['Age', 'Income', 'CreditLimit', 'TotalSpend', 'Tenure', 'TotalTransactions']
        )
    
    # Distribution plot
    st.subheader(f"Distribution of {feature}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Overall distribution
    ax1.hist(df[feature], bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel(feature)
    ax1.set_ylabel('Count')
    ax1.set_title(f'{feature} Distribution')
    ax1.grid(True, alpha=0.3)
    
    # By attrition status
    churned = df[df['AttritionFlag'] == 1][feature]
    retained = df[df['AttritionFlag'] == 0][feature]
    
    ax2.hist([retained, churned], bins=20, label=['Retained', 'Churned'], 
             color=['green', 'red'], alpha=0.6, edgecolor='black')
    ax2.set_xlabel(feature)
    ax2.set_ylabel('Count')
    ax2.set_title(f'{feature} by Attrition Status')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Statistics by attrition
    st.markdown("---")
    st.subheader("Statistics by Attrition Status")
    
    stats_df = df.groupby('AttritionFlag')[feature].describe().round(2)
    stats_df.index = ['Retained', 'Churned']
    st.dataframe(stats_df)

def show_prediction(df, model):
    st.header("🎯 Individual Customer Prediction")
    st.markdown("Select a customer from the dataset to predict their attrition risk.")
    
    # Customer selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Sample selection
        st.subheader("Select Customer")
        
        # Random sample button
        if st.button("🎲 Random Customer"):
            st.session_state.selected_idx = np.random.randint(0, len(df))
        
        # Or manual selection
        selected_idx = st.number_input(
            "Or enter row number:",
            min_value=0,
            max_value=len(df)-1,
            value=st.session_state.get('selected_idx', 0),
            step=1
        )
        st.session_state.selected_idx = selected_idx
    
    with col2:
        # Get selected customer
        customer = df.iloc[selected_idx]
        customer_id = customer['CustomerID']
        
        st.subheader(f"Customer Details - {customer_id}")
        
        # Display key features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Age", f"{int(customer['Age'])} years")
            st.metric("Tenure", f"{int(customer['Tenure'])} years")
        
        with col2:
            st.metric("Income", f"₱{customer['Income']:,.0f}")
            st.metric("Credit Limit", f"₱{customer['CreditLimit']:,.0f}")
        
        with col3:
            st.metric("Total Spend", f"₱{customer['TotalSpend']:,.0f}")
            st.metric("Transactions", f"{int(customer['TotalTransactions'])}")
    
    st.markdown("---")
    
    # Prediction
    st.subheader("Prediction Results")
    
    # Prepare features for prediction - use exact same 60 features as training
    training_features = ['Age', 'Income', 'CreditLimit', 'TotalTransactions', 'TotalSpend', 'Tenure',
                        'PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5', 'PCA_6', 'PCA_7', 'PCA_8', 
                        'PCA_9', 'PCA_10', 'PCA_11', 'PCA_12', 'PCA_13', 'PCA_14', 'PCA_15', 
                        'PCA_16', 'PCA_17', 'PCA_18', 'PCA_19', 'PCA_20', 'PCA_21', 'PCA_22', 
                        'PCA_23', 'PCA_24', 'PCA_25', 'PCA_26', 'PCA_27', 'PCA_28', 'PCA_29', 
                        'PCA_30', 'PCA_31', 'PCA_32', 'PCA_33', 'PCA_34', 'PCA_35', 'PCA_36', 
                        'PCA_37', 'PCA_38', 'PCA_39', 'PCA_40', 'PCA_41', 'PCA_42', 'PCA_43', 
                        'PCA_44', 'PCA_45', 'SpendingRate', 'MonthlySpendingRate', 
                        'Log_TransactionFrequency', 'Gender_encoded', 'MaritalStatus_encoded', 
                        'EducationLevel_encoded', 'CardType_ordinal', 'TenureGroup_ordinal', 
                        'CreditTier_ordinal']
    X_pred = customer[training_features].values.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(X_pred)[0]
    prediction_proba = model.predict_proba(X_pred)[0]
    
    # Get actual label
    actual = int(customer['AttritionFlag'])
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Actual Status", "Churned" if actual == 1 else "Retained")
    
    with col2:
        st.metric("Predicted Status", "Churned" if prediction == 1 else "Retained")
    
    with col3:
        is_correct = prediction == actual
        accuracy_text = "✅ Correct" if is_correct else "❌ Incorrect"
        st.metric("Prediction", accuracy_text)
    
    # Probability chart
    st.markdown("---")
    st.subheader("Prediction Confidence")
    
    fig, ax = plt.subplots(figsize=(8, 3))
    
    labels = ['Retain', 'Churn']
    probabilities = prediction_proba
    colors = ['green', 'red']
    
    bars = ax.barh(labels, probabilities, color=colors, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    
    for bar, prob in zip(bars, probabilities):
        ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1%}', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature importance note
    st.info("💡 **Key Factors**: The model primarily considers Income, Credit Limit, and Total Spend when making predictions.")

if __name__ == "__main__":
    main()