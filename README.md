# Clinical Meeting Recorder

**DSCI 441 Project — Spring 2026**
**Josh Buck & Alrakhmet Muratbek — Lehigh University**

## Overview

A Streamlit web application that transforms doctor-patient conversations into structured clinical notes using classical machine learning. The system will capture audio, convert speech to text, extract clinical entities, and apply ML models to classify visit types, assess severity, predict follow up needs, and cluster similar encounters.

## Architecture
```
Audio Input -> Whisper -> Raw Transcript -> LLM Entity Extraction -> Feature Engineering -> Classical ML Models -> Structured Clinical Note
```

**Design principle:** Deep learning tools handle data ingestion and feature extraction. The core analytical work uses classical ML techniques: Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost, PCA, and K-Means clustering.

## Methodology

We use a two stage pipeline. First, pre-trained tools handle data transformation: OpenAI Whisper (local, open-source) for speech-to-text and Ollama (llama3.1:8b, running locally) for structured entity extraction from transcripts. These serve as feature engineering steps, not the core project work.

Second, we apply classical ML to the extracted features. We built TF-IDF vectors (500 features) and dialogue-level features (word count, turn count, speaker ratios) from 1,201 doctor-patient dialogues. We trained and compared six classifiers using 5-fold cross-validation and hyperparameter tuning via GridSearchCV. Results were validated with bootstrap resampling (n=200) for 95% confidence intervals and paired hypothesis testing. We used the Kruskal-Wallis test to confirm that dialogue length varies significantly across section types (p < 0.001), justifying its inclusion as a feature.

Best model: Logistic Regression (C=0.1, balanced class weights), macro F1 = 0.591, statistically significantly better than the Naive Bayes baseline (p < 0.001).

## Datasets

- **MTS-Dialog** (primary): 1,201 training + 100 validation doctor-patient dialogues with clinical note section labels.
  https://github.com/abachaa/MTS-Dialog

- **ACI-BENCH** (supplementary): 67 full clinical encounters with complete notes and patient metadata (age, gender, chief complaint).
  https://github.com/microsoft/clinical_visit_note_summarization_corpus

## ML Models

| Task | Models | Best Result |
|------|--------|-------------|
| Section Type Classification | Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost | Logistic Reg, F1=0.591 |
| Severity Classification | In progress (LLM-assisted labeling on ACI-BENCH) | — |
| Follow-up Prediction | Planned | — |
| Encounter Clustering | Planned (K-Means, PCA) | — |

## Tool Stack

- **Speech-to-Text:** OpenAI Whisper
- **Entity Extraction:** Ollama with llama3.1:8b
- **ML:** scikit-learn, XGBoost
- **App:** Streamlit
- **Language:** Python

## Project Structure
```
ClinicalMeetingRecorder/
├── app.py                              # Streamlit application
├── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb    # TF-IDF, model training, comparison
│   ├── 03_severity_classification.ipynb # ACI-BENCH + Ollama extraction
│   ├── 04_statistical_analysis.ipynb   # Bootstrap, hypothesis tests
│   └── 05_entity_extraction.ipynb      # LLM entity extraction pipeline
├── src/
│   ├── audio_processor.py              # Whisper speech-to-text
│   ├── text_processor.py               # NLP preprocessing & TF-IDF
│   ├── entity_extractor.py             # LLM entity extraction
│   ├── classifiers.py                  # Classical ML models
│   └── note_generator.py              # Clinical note assembly
├── models/                             # Saved trained models
├── data/
│   ├── raw/                            # Original datasets
│   └── processed/                      # Extracted entities, features
└── docs/
```