import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
from catboost import CatBoostClassifier # Make sure catboost is installed
import matplotlib.pyplot as plt

# Set page config for better layout
st.set_page_config(layout="wide")

# --- 1. Load Model and Metadata ---
@st.cache_resource # Cache the model loading for better performance
def load_model_and_metadata():
    try:
        with open('best_catboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('categorical_cols.pkl', 'rb') as f:
            categorical_cols = pickle.load(f)
        return model, feature_names, categorical_cols
    except FileNotFoundError:
        st.error("Model or metadata files not found. Please ensure 'best_catboost_model.pkl', 'feature_names.pkl', and 'categorical_cols.pkl' are in the same directory.")
        st.stop()

model, feature_names, categorical_cols = load_model_and_metadata()

# --- 2. Preprocessing Function ---
def preprocess_input(user_input_dict, feature_names, categorical_cols):
    # Create a DataFrame from user input
    input_df = pd.DataFrame([user_input_dict])

    # Convert any detected float64 categorical columns to object for consistent one-hot encoding
    for col in categorical_cols:
        if col in input_df.columns and pd.api.types.is_numeric_dtype(input_df[col]):
            input_df[col] = input_df[col].astype(str)

    # Apply One-Hot Encoding to categorical features
    # Ensure consistent encoding with the model's training data
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True, dtype=int)

    # Align columns with the model's feature names
    # This handles cases where user input might miss a category or have an unseen category
    aligned_input = input_encoded.reindex(columns=feature_names, fill_value=0)

    return aligned_input

# --- 3. Streamlit UI ---
st.title("Infectious Complications Prediction App")
st.write("Enter patient data to predict the likelihood of infectious complications after surgery.")

# Create two columns for layout: one for input, one for results
input_col, output_col = st.columns([1, 1])

with input_col:
    st.header("Patient Data Input")
    user_input_data = {}
    
    # Dynamically create input fields for each feature
    # Group input fields by type for better user experience
    
    numeric_features = []
    categorical_features_for_input = [] # Use a new list to store categorical features for input creation

    # Separate features based on whether they were OHE original features or numeric
    for feature in feature_names:
        # Skip OHE columns for direct input creation
        is_ohe = False
        for cat_col in categorical_cols:
            if feature.startswith(f'{cat_col}_'):
                is_ohe = True
                break
        if not is_ohe and feature not in categorical_cols:
            numeric_features.append(feature)
        elif feature in categorical_cols:
            categorical_features_for_input.append(feature)
            
    # Sort features alphabetically for consistent UI
    numeric_features.sort()
    categorical_features_for_input.sort()

    # Collect user input for numeric features
    st.subheader("Numeric Features")
    for feature in numeric_features:
        # Attempt to infer type and default value. For simplicity, all numeric treated as floats initially.
        # More robust type inference and range limits could be added here.
        default_value = 0.0 # Default to 0, user can change
        if feature in ['Возраст', 'Вес', 'Рост', 'ИМТ', 'Длительность операции, мин', 'Длит. госпитализ, дни',
                       'Длительность от поступления до операции, дни', 'Длительность после операции до выписки, дни']:
            user_input_data[feature] = st.number_input(f"Enter {feature}", value=float(default_value), format="%.2f", key=feature)
        else: # Generic float input for other numeric features
            user_input_data[feature] = st.number_input(f"Enter {feature}", value=float(default_value), format="%.2f", key=feature)

    # Collect user input for categorical features
    st.subheader("Categorical Features")
    for feature in categorical_features_for_input:
        # For categorical features, get unique values from original df if available
        # For now, we assume simple string input or a selection list if we had pre-defined options
        # For example, if we knew 'Профильное отделение' had specific options:
        # options = df_original['Профильное отделение'].unique().tolist()
        # user_input_data[feature] = st.selectbox(f"Select {feature}", options, key=feature)
        # If we expect the encoded output to be integer placeholders like 999.0 for 'Возбудитель', 'Биоматериал'
        if feature in ['Возбудитель', 'Биоматериал']:
            # For these, 999.0 implies 'None' or 'Not Applicable', otherwise actual values
            options = ['999.0'] + [str(i) for i in range(10)] # Example options, need to be real unique values
            user_input_data[feature] = st.selectbox(f"Select {feature}", options, index=0, key=feature)
        else:
            # For other string categorical features, use text input
            user_input_data[feature] = st.text_input(f"Enter {feature}", key=feature)

    predict_button = st.button("Predict Complication Risk")

with output_col:
    st.header("Prediction Result")
    if predict_button:
        if not user_input_data: # Basic check if any input was provided
            st.warning("Please enter patient data.")
        else:
            # Ensure all feature_names from the model are present in user_input_data
            # For missing numeric features in user_input_data, fill with 0 or a sensible default
            for col in feature_names:
                if col not in user_input_data:
                    # Heuristic: if it's a generated OHE column, it should be 0 unless a specific category is active
                    # If it's an original numeric column and not in input, fill with a default (e.g., 0)
                    is_ohe_feature = False
                    for cat_col in categorical_cols:
                        if col.startswith(f'{cat_col}_'):
                            is_ohe_feature = True
                            break
                    if not is_ohe_feature: # If it's an original numeric feature
                        user_input_data[col] = 0.0 # Default value for missing numeric input
                    # OHE features will be handled by reindex in preprocess_input


            # Preprocess user input
            processed_input = preprocess_input(user_input_data, feature_names, categorical_cols)
            
            # Make prediction
            prediction_proba = model.predict_proba(processed_input)[:, 1]
            
            st.subheader(f"Predicted Probability of Complications: {prediction_proba[0]:.4f}")
            
            if prediction_proba[0] > 0.5: # Example threshold
                st.error("High Risk of Infectious Complications!")
            else:
                st.success("Low Risk of Infectious Complications.")
            
            # --- SHAP Interpretation ---
            st.subheader("SHAP Interpretation")
            # Ensure SHAP JS is initialized for plots (though Streamlit handles some of this automatically)
            shap.initjs()
            
            # Create a TreeExplainer for the CatBoost model
            # The model is the 'catboost' step within the pipeline
            try:
                catboost_only_model = model.named_steps['catboost']
                explainer = shap.TreeExplainer(catboost_only_model)
                shap_values = explainer.shap_values(processed_input)

                # Use the feature names for plotting
                # If the model is a pipeline, the explainer needs the actual CatBoost model
                # and shap_values will be for the positive class (index 1)
                st.write("Feature contributions to the prediction (Force Plot):")
                # Use matplotlib=True and show=False to capture the plot and display it with st.pyplot
                shap.force_plot(explainer.expected_value[1], shap_values[1][0], processed_input.iloc[0], matplotlib=True, show=False)
                st.pyplot(bbox_inches='tight') # Display matplotlib plot in Streamlit
                plt.clf() # Clear the current figure to prevent it from being displayed again accidentally
            except Exception as e:
                st.warning(f"Could not generate SHAP plot: {e}")
                st.info("SHAP plots might require specific data formats or model types. If the model is a pipeline, ensure the explainer targets the CatBoost step.")

# Note: To run this app, save the code as `app.py` and execute `streamlit run app.py` in your terminal.
# Make sure all .pkl files are in the same directory as app.py
