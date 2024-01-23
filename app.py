import streamlit as st
from src.pipeline.prediction_pipeline import extract_all_features,prediction

def main():
    st.title("Phished Url Detector")

    # Input box for feature name
    url = st.text_input("Enter Url:", "")

    # Submit button
    if st.button("Submit"):
        if url:
            extracted_features = extract_all_features(url)
            result = prediction(extracted_features)
            st.success(result)
        else:
            st.warning("Please enter a feature name.")

if __name__ == "__main__":
    main()