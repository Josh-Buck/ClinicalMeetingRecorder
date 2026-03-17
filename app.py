import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import hstack

st.set_page_config(page_title="Clinical Meeting Recorder", layout="wide")
st.title("Clinical Meeting Recorder")
st.markdown("*Transform doctor-patient conversations into structured clinical notes*")

#train model on startup (load saved models later)
@st.cache_resource
def load_model():
    train_df = pd.read_csv('data/raw/MTS-Dialog-TrainingSet.csv')

    section_mapping = {
        'GENHX': 'History', 'FAM/SOCHX': 'History', 'PASTMEDICALHX': 'History',
        'PASTSURGICAL': 'History', 'OTHER_HISTORY': 'History', 'GYNHX': 'History',
        'CC': 'Chief Complaint', 'ROS': 'Exam/Review', 'EXAM': 'Exam/Review',
        'ALLERGY': 'Medications/Allergies', 'MEDICATIONS': 'Medications/Allergies',
        'IMMUNIZATIONS': 'Medications/Allergies',
        'ASSESSMENT': 'Assessment/Diagnosis', 'DIAGNOSIS': 'Assessment/Diagnosis',
        'PLAN': 'Plan/Disposition', 'DISPOSITION': 'Plan/Disposition',
        'EDCOURSE': 'Plan/Disposition', 'PROCEDURES': 'Plan/Disposition',
        'IMAGING': 'Plan/Disposition', 'LABS': 'Plan/Disposition',
    }
    train_df['section_group'] = train_df['section_header'].map(section_mapping)

    tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.95)
    X_tfidf = tfidf.fit_transform(train_df['dialogue'])

    def extract_features(df):
        features = pd.DataFrame()
        features['word_count'] = df['dialogue'].str.split().str.len()
        features['turn_count'] = df['dialogue'].str.count(r'Doctor:|Patient:|\[doctor\]|\[patient\]')
        features['doctor_words'] = df['dialogue'].apply(
            lambda x: len(' '.join(re.findall(r'(?:Doctor|doctor).*?(?=Patient|patient|$)', x, re.DOTALL)).split()))
        features['patient_words'] = df['dialogue'].apply(
            lambda x: len(' '.join(re.findall(r'(?:Patient|patient).*?(?=Doctor|doctor|$)', x, re.DOTALL)).split()))
        features['avg_turn_length'] = features['word_count'] / features['turn_count'].clip(lower=1)
        return features

    X_extra = extract_features(train_df)
    X_combined = hstack([X_tfidf, X_extra.values])
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X_combined)

    model = LogisticRegression(C=0.1, solver='lbfgs', class_weight='balanced', random_state=42, max_iter=2000)
    model.fit(X_scaled, train_df['section_group'])

    return model, tfidf, scaler, extract_features

model, tfidf, scaler, extract_features = load_model()

#user input
st.markdown("---")
input_mode = st.radio("Choose input method:", ["Paste Transcript", "Upload Audio)"], horizontal=True)

if input_mode == "Paste Transcript":
    transcript = st.text_area(
        "Paste a doctor-patient dialogue:",
        height=200,
        placeholder="Doctor: What brings you in today?\nPatient: I've been having headaches for the past week..."
    )

    if transcript and st.button("Analyze", type="primary"):
        #build features for this dialogue
        input_df = pd.DataFrame({'dialogue': [transcript]})
        X_tfidf_input = tfidf.transform(input_df['dialogue'])
        X_extra_input = extract_features(input_df)
        X_input = hstack([X_tfidf_input, X_extra_input.values])
        X_input_scaled = scaler.transform(X_input)

        #predict
        prediction = model.predict(X_input_scaled)[0]
        probabilities = model.predict_proba(X_input_scaled)[0]
        classes = model.classes_

        #display results
        st.markdown("---")
        st.header("Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Section Type", prediction)
        with col2:
            confidence = max(probabilities) * 100
            st.metric("Confidence", f"{confidence:.1f}%")
        with col3:
            word_count = len(transcript.split())
            st.metric("Word Count", word_count)

        #show all class probabilities
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({'Section Type': classes, 'Probability': probabilities})
        prob_df = prob_df.sort_values('Probability', ascending=True)
        st.bar_chart(prob_df.set_index('Section Type'))

else:
    st.info("Audio upload will be available in the final version. For now, paste a transcript above.")

st.markdown("---")
st.caption("DSCI 441 Project | Josh Buck & Alrakhmet Muratbek | Lehigh University | Spring 2026")