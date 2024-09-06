import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define available models
available_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Ridge Regression":  Ridge(),
    "Lasso Regression": Lasso(),
    "ElasticNet Regression": ElasticNet()
}

# Function to preprocess the data
def preprocess_data(df):
    # Removing any duplicates
    df = df.drop_duplicates()
    
    # Handling outliers using z-score (removing rows where any column has a z-score > 3)
    df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) < 3).all(axis=1)]
    
    # Renaming columns for better readibility (optional)
    #df.columns = [col.replace('_', ' ').title() for col in df.columns]
    
    # Converving categorical variables into numerical
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Normalizing numerical features
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    
    return df

# Function for model training and performance Calculation
def train_evaluate_model(df, model, target_variable):
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    performance = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2 Score": r2_score(y_test, predictions)
    }
    return performance, model, X_train, y_train

def display_results_and_plots(model, X, y, target_variable):
    # Feature importance (for models that support it)
    if hasattr(model, 'coef_'):
        importances = model.coef_
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = None

    if importances is not None:
        # For ElasticNet and Lasso, filter out zero coefficients
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        })

        # Filter to show only non-zero coefficients (important for Lasso and ElasticNet)
        feature_importance_df = feature_importance_df[feature_importance_df['Importance'] != 0]
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        if not feature_importance_df.empty:
            # Plot top feature importance
            st.subheader("Top Feature Importance")
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), orient='h')
            st.pyplot(plt)

            # Plot regression plots for top features in a 3-column layout
            top_features = feature_importance_df['Feature'].head(10)
            st.subheader("Regression Plots for Top Features")
            
            for i, feature in enumerate(top_features):
                if i % 3 == 0:
                    cols = st.columns(3)
                with cols[i % 3]:
                    st.write(f"**{feature} vs {target_variable}**")
                    fig, ax = plt.subplots()
                    sns.regplot(x=X[feature], y=y, ax=ax)
                    st.pyplot(fig)
        else:
            st.write("No significant features identified by this model (all coefficients are zero).")
    else:
        st.write("This model does not provide feature importance.")

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Set the Streamlit page configuration
st.set_page_config(page_title="RegressLy - Regression Analysis App", layout="wide")

# Load the CSS file
load_css("style.css")

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

st.sidebar.title("RegressLy")

# Render different sidebar options based on whether analysis is complete
if st.session_state.analysis_complete:
    page = st.sidebar.radio("Go to", ["Home", "Regression Analysis", "Dashboard"], index=2)
else:
    page = st.sidebar.radio("Go to", ["Home", "Regression Analysis"], index=0)

st.session_state.page = page

# Home Page
if page == "Home":
    st.markdown("<h1 style='padding-top: 20px;'>RegressLy</h1>", unsafe_allow_html=True)

    st.write("""
    **Regressly** is your go-to application for conducting comprehensive regression analysis with ease. Whether you're a data scientist, analyst, or someone new to the world of data, **Regressly** simplifies the process of uncovering relationships between variables, predicting outcomes, and gaining insights from your datasets.
    
    **Key Features:**
    - **Automated Data Preprocessing**: Upload your dataset, and let **Regressly** handle the heavy lifting, including cleaning, scaling, and encoding.
    - **Intuitive Regression Analysis**: Perform detailed regression analysis with just a few clicks, and understand the key drivers behind your data.
    - **Interactive Visualizations**: Explore your results through dynamic, easy-to-understand visualizations that make your findings clear and actionable.
    - **Insights Generation**: Receive automatically generated insights to support decision-making and communicate your results effectively.
    
    Get started with **Regressly** today and make your data work for you!
    """)
    st.write("- ###### Use the left menu to navigate to the Regression analysis to start")

# Regression Analysis Page
elif page == "Regression Analysis":
    st.title("🔍 Regression Analysis")
    st.write("Using the side menu upload your dataset and start the analysis process...")

    # Sidebar for user inputs
    st.sidebar.header("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("CSV format only", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Overview")
        st.write(df.head())

        # Target variable selection
        st.write("### Select the target variable")
        target_variable = st.selectbox("Columns", df.columns)

        # Check if the target variable is continuous before proceeding
        if pd.api.types.is_numeric_dtype(df[target_variable]):
            # Model selection
            st.write("### Select the regression model")
            selected_model_name = st.selectbox("Models", available_models.keys())

            # Proceeding with the process
            if st.button("Proceed with Regression"):
                # Data preprocessing
                df = preprocess_data(df)

                # Model training and evaluation
                model = available_models[selected_model_name]
                performance, trained_model, X_train, y_train = train_evaluate_model(df, model, target_variable)

                # Store results in session state for the dashboard page
                st.session_state.performance = performance
                st.session_state.trained_model = trained_model
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.target_variable = target_variable
                st.session_state.selected_model_name = selected_model_name

                # Mark analysis as complete
                st.session_state.analysis_complete = True

                # Inform the user that regression has been performed
                st.success("Regression analysis completed.")
                #st.success("Visualize the results by selecting :red[**Dashboard**] on the left menu.")
                if st.button("View Dashboard"):
                    # Redirect to the Dashboard page
                    st.session_state.page = "Dashboard"
                    st.experimental_rerun()
        else: 
            st.error("Regression analysis can only be performed on continuous, quantitative target variables. Please select an appropriate target variable.")

# Regression Dashboard Page
elif page == "Dashboard":
    # Ensure the page starts at the top
    st.write("")
    st.write("")
    #st.write("")
    
    st.title("Regression Analysis Dashboard")
    st.write("Find the detailed regression results and visualizations displayed below.")

    # Retrieve the initial session state values
    performance = st.session_state.get('performance', None)
    X_train = st.session_state.get('X_train', None)
    y_train = st.session_state.get('y_train', None)
    target_variable = st.session_state.get('target_variable', None)
    selected_model_name = st.session_state.get('selected_model_name', None)
    trained_model = st.session_state.get('trained_model', None)  # Ensure trained_model is retrieved

    # Add a model selector on the dashboard page
    st.sidebar.write("#### Select a regression model to view results")
    selected_model_name_dashboard = st.sidebar.radio("Models", available_models.keys(), index=list(available_models.keys()).index(selected_model_name))

    if X_train is not None and y_train is not None:
        # If the selected model is different from the one used in the initial analysis, re-train and evaluate
        if selected_model_name_dashboard != selected_model_name or trained_model is None:
            model_dashboard = available_models[selected_model_name_dashboard]
            performance_dashboard, trained_model_dashboard, _, _ = train_evaluate_model(pd.concat([X_train, y_train], axis=1), model_dashboard, target_variable)

            # Update the session state with the new model's results
            st.session_state.performance = performance_dashboard
            st.session_state.trained_model = trained_model_dashboard
            st.session_state.selected_model_name = selected_model_name_dashboard
            performance = performance_dashboard
            trained_model = trained_model_dashboard

        #st.markdown(f"Model Selected: {selected_model_name_dashboard}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{performance['MAE']:.2f}")
        col2.metric("MSE", f"{performance['MSE']:.2f}")
        col3.metric("RMSE", f"{performance['RMSE']:.2f}")
        col4.metric("R2 Score", f"{performance['R2 Score']:.2f}")

        display_results_and_plots(trained_model, X_train, y_train, target_variable)

    else:
        st.write("No analysis has been conducted yet. Please go to the Regression Analysis page and complete the process.")
# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created by Jean-Emmanuel Kouadio | © 2024")