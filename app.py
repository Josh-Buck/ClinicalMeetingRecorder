import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
from scipy.sparse import hstack
from scipy.special import softmax

st.set_page_config(page_title="Clinical Meeting Recorder", layout="wide")
st.title("Clinical Meeting Recorder")
st.markdown("*Transform doctor-patient conversations into structured clinical notes using classical ML*")

def extract_dialogue_features(df):
    features = pd.DataFrame()
    features['word_count'] = df['dialogue'].str.split().str.len()
    features['turn_count'] = df['dialogue'].str.count(r'Doctor:|Patient:|\[doctor\]|\[patient\]')
    features['doctor_words'] = df['dialogue'].apply(
        lambda x: len(' '.join(re.findall(r'(?:Doctor|doctor).*?(?=Patient|patient|$)', x, re.DOTALL)).split()))
    features['patient_words'] = df['dialogue'].apply(
        lambda x: len(' '.join(re.findall(r'(?:Patient|patient).*?(?=Doctor|doctor|$)', x, re.DOTALL)).split()))
    features['avg_turn_length'] = features['word_count'] / features['turn_count'].clip(lower=1)
    return features

@st.cache_resource
def load_models():
    models = {}
    for name, path in [('section', 'models/section_classifier.joblib'),
                        ('followup', 'models/followup_classifier.joblib'),
                        ('severity', 'models/severity_classifier.joblib')]:
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

models = load_models()

st.markdown("---")
transcript = st.text_area(
    "Paste a doctor-patient dialogue:",
    height=200,
    placeholder="Doctor: What brings you in today?\nPatient: I've been having headaches for the past week..."
)

if transcript and st.button("Analyze", type="primary"):
    input_df = pd.DataFrame({'dialogue': [transcript]})
    dialogue_feats = extract_dialogue_features(input_df)

    st.markdown("---")
    st.header("Analysis Results")

    # --- Section Classification ---
    if 'section' in models:
        m = models['section']
        X_tfidf = m['tfidf'].transform(input_df['dialogue'])
        X = hstack([X_tfidf, dialogue_feats.values])
        X_scaled = m['scaler'].transform(X)

        prediction = m['model'].predict(X_scaled)[0]
        decision = m['model'].decision_function(X_scaled)
        probs = softmax(decision, axis=1)[0]
        classes = m['model'].classes_

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Section Type", prediction)
        with col2:
            st.metric("Confidence", f"{probs.max() * 100:.1f}%")
        with col3:
            st.metric("Word Count", len(transcript.split()))

        st.subheader("Section Probabilities")
        prob_df = pd.DataFrame({'Section Type': classes, 'Probability': probs})
        prob_df = prob_df.sort_values('Probability', ascending=True)
        st.bar_chart(prob_df.set_index('Section Type'))

    # --- Follow-Up Prediction ---
    if 'followup' in models:
        m = models['followup']
        X_tfidf = m['tfidf'].transform(input_df['dialogue'])
        entity_feats = np.zeros((1, 6))  # num_symptoms, num_diagnoses, num_medications, has_chief_complaint, severity_high, severity_low
        X = hstack([X_tfidf, dialogue_feats.values, entity_feats])
        X_scaled = m['scaler'].transform(X)

        fu_pred = m['model'].predict(X_scaled)[0]
        decision = m['model'].decision_function(X_scaled)
        fu_conf = softmax(np.column_stack([-decision, decision]), axis=1)[0]

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            label = "Yes" if fu_pred == 1 else "No"
            st.metric("Follow-Up Needed", label)
        with col2:
            st.metric("Confidence", f"{fu_conf.max() * 100:.1f}%")

    # --- Severity Classification ---
    if 'severity' in models:
        m = models['severity']
        X_tfidf = m['tfidf'].transform(input_df['dialogue'])
        entity_feats = np.zeros((1, 5))  # num_symptoms, num_diagnoses, num_medications, has_chief_complaint, has_followup
        X = hstack([X_tfidf, dialogue_feats.values, entity_feats])
        X_scaled = m['scaler'].transform(X)

        sev_pred = m['model'].predict(X_scaled)[0]
        decision = m['model'].decision_function(X_scaled)
        sev_conf = softmax(np.column_stack([-decision, decision]), axis=1)[0]

        col1, col2 = st.columns(2)
        with col1:
            label = "High" if sev_pred == 1 else "Not High"
            st.metric("Severity", label)
        with col2:
            st.metric("Confidence", f"{sev_conf.max() * 100:.1f}%")

st.markdown("---")
st.caption("DSCI 441 Project | Josh Buck & Alrakhmet Muratbek | Lehigh University | Spring 2026")
