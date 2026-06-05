import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# --- 1. Load Artifacts ---
@st.cache_resource
def load_artifacts():
    model = joblib.load(r'C:\saas customer churn prediction\xgb_churn_model.pkl')
    preprocessor = joblib.load(r'C:\saas customer churn prediction\preprocessor.pkl')
    medians = joblib.load(r'C:\saas customer churn prediction\train_medians.pkl')
    modes = joblib.load(r'C:\saas customer churn prediction\train_modes.pkl')
    return model, preprocessor, medians, modes

model, preprocessor, train_medians, train_modes = load_artifacts()

# --- 2. App UI Layout ---
st.set_page_config(page_title="SaaS Churn Predictor", layout="wide")
st.title("📉 SaaS Customer Churn Predictor & Explainer")
st.markdown("Enter customer details to predict churn probability and identify actionable retention strategies.")

# --- 3. Sidebar Inputs ---
st.sidebar.header("Customer Profile")

# We only ask for the top business drivers identified in our EDA/SHAP
contract_type = st.sidebar.selectbox("Contract Type", ['Month-to-Month', 'One Year', 'Two Year'])
last_login_days_ago = st.sidebar.slider("Days Since Last Login", 0, 100, 15)
satisfaction_score = st.sidebar.slider("Satisfaction Score (1-10)", 1, 10, 5)
number_of_support_tickets = st.sidebar.number_input("Number of Support Tickets", 0, 50, 2)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 20.0, 200.0, 70.0)
tenure_months = st.sidebar.number_input("Tenure (Months)", 1, 72, 12)

# --- 4. Prediction Logic ---
if st.sidebar.button("Predict Churn Risk"):
    # Create a baseline dataframe using training medians/modes to prevent missing column errors
    input_data = pd.DataFrame([train_modes])
    
    # Overwrite with user inputs
    input_data['contract_type'] = contract_type
    input_data['last_login_days_ago'] = last_login_days_ago
    input_data['satisfaction_score'] = satisfaction_score
    input_data['number_of_support_tickets'] = number_of_support_tickets
    input_data['monthly_charges'] = monthly_charges
    input_data['tenure_months'] = tenure_months
    
    # Feature Engineering (Must match training exactly)
    input_data['engagement_score'] = 10 * 6 / 60 # Default placeholder. Simplified for app
    input_data['support_intensity'] = number_of_support_tickets / (tenure_months + 1)
    input_data['value_per_month'] = monthly_charges # Simplified for app
    input_data['contract_risk'] = 1 if contract_type == 'Month-to-Month' else 0

    # Preprocess and Predict
    processed_input = preprocessor.transform(input_data)
    churn_prob = model.predict_proba(processed_input)[0][1]
    
    # --- 5. Display Results ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(label="Predicted Churn Probability", value=f"{churn_prob*100:.1f}%", 
                  delta=f"{'High Risk' if churn_prob > 0.5 else 'Safe'}", delta_color="inverse")
        
        if churn_prob > 0.5:
            st.error("🚨 **Action Required:** High risk of churn detected.")
        else:
            st.success("✅ **Status:** Customer is stable.")

    with col2:
        st.subheader("🔍 Why is this customer churning? (Model Explainability)")
        
        # Run SHAP for this single prediction
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(processed_input)[0]
        
        # Get feature names from the preprocessor
        feature_names = (
            preprocessor.named_transformers_['num'].feature_names_in_.tolist() + 
            list(preprocessor.named_transformers_['cat'].get_feature_names_out(['gender', 'region', 'income_level', 'subscription_type', 'usage_frequency', 'payment_method', 'contract_type', 'promotional_response', 'discount_used']))
        )
        
        # Find top 3 drivers
        shap_impact = pd.DataFrame({'Feature': feature_names, 'Impact': shap_vals})
        shap_impact['Abs_Impact'] = shap_impact['Impact'].abs()
        top_drivers = shap_impact.sort_values('Abs_Impact', ascending=False).head(3)
        
        # Print plain English explanations
        for _, row in top_drivers.iterrows():
            feat = row['Feature']
            impact = row['Impact']
            direction = "⬆️ **Increases**" if impact > 0 else "⬇️ **Decreases**"
            
            # Pull the real value the user inputted
            if feat in input_data.columns:
                real_val = input_data[feat].iloc[0]
            else:
                real_val = "N/A"
                
            readable_feat = feat.replace('_', ' ').title()
            st.markdown(f"- **{readable_feat}** (Value: `{real_val}`): {direction} churn risk.")
            
        if churn_prob >= 0.5:
            st.info("💡 **Recommendation:** Based on these factors, customer service follow-up recommended.")