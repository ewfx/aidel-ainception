import streamlit as st
import pandas as pd
import requests
import json

# Set page configuration
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Custom CSS to hide sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("Transaction Risk Scoring System")

uploaded_file = st.file_uploader("Upload a CSV or Text file", type=["csv", "txt"])


if uploaded_file is not None:
    st.write("Processing uploaded file...")
    file_type = uploaded_file.type
    file_content = uploaded_file.read().decode("utf-8")
    payload = {"file_type": file_type, "content": file_content}
    st.write(f"Sending to backend: {payload['file_type']} file")
    try:
        response = requests.post("http://localhost:8000/process_transaction", json=payload)
        response.raise_for_status()
        results = response.json()
        #st.write(f"Backend response: {results}")
    except requests.exceptions.RequestException as e:
        st.error(f"HTTP error processing file: {e}")
        results = None

    if results:
        if isinstance(results, dict) and "error" in results:
            st.error(f"Backend error: {results['error']}")
        elif isinstance(results, list):
            st.write("Final Results:")
            try:
    
                results_df = pd.DataFrame(results)
                #st.write("Results DataFrame before styling:")
                #st.write(results_df)
                #st.write("Columns in results_df:", results_df.columns.tolist())  # Debug: Log columns

                def color_risk(val):
                    if val == "Low":
                        return 'background-color: lightgreen'
                    elif val == "Medium":
                        return 'background-color: yellow'
                    elif val == "High":
                        return 'background-color: lightcoral'
                    return ''

                def color_confidence(val):
                    if val >= 0.9:
                        return 'background-color: lightgreen'
                    elif 0.7 <= val < 0.9:
                        return 'background-color: yellow'
                    else:
                        return 'background-color: lightcoral'
                    return ''

                
                styled_df = results_df.style
                if 'RiskCategory' in results_df.columns:
                    styled_df = styled_df.applymap(color_risk, subset=['RiskCategory'])
                if 'ConfidenceScore' in results_df.columns:
                    styled_df = styled_df.applymap(color_confidence, subset=['ConfidenceScore'])
                
                try:
                    st.dataframe(styled_df, use_container_width=True)
                except TypeError:
                    st.dataframe(styled_df)

                st.write("Raw JSON Results:")
                st.json(results)
            except ValueError as e:
                st.error(f"Error creating DataFrame: {e}")
                st.write("Raw results:", results)
        else:
            st.error("Unexpected backend response format")
    else:
        st.error("No results returned. Check the backend logs for errors.")
else:
    st.write("Please upload a CSV or Text file.")