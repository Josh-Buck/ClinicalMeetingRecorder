# ClinicalMeetingRecorder

**DSCI 441 Project - Spring 2206**
**Josh Buck and Alrakhmet Muratbek - Lehigh University**

## Overview

A streamlit web applications that transforms doctor-patient conversations into structured clinical notes using classical machine learning. The system caputres audio, conversts speech to text, extracts clinical entities, and applies ML models to classify severity, categorize visit types, predict follow-up needs, and cluster similar encounters.

## Architecture

```
Audio input -> whisper (local) -> raw transcript -> LLM entity extraction -> feature Engineering -> classical ML models -> strcuctured clinical note
```

**Design principle:** Deep learning tools will be used to handle data ingestion and feature extraction. The core analytical work will use classical ML techniques form our course such as: Naive Bayes, Logistic Regression, SVM, Random forest, XGBoost, PCA, k-means clustering, etc.

## Dataset
- Try MTS Dialog and/or ACI-BENCH

## Project Structure

```
ClinicalMeetingRecorder/
|--app.py
|--requirements.txt
|--notebooks/
|   |--01_eda.ipynb
|--src/
|   |--audio_processor.py
|   |--text_processor.py
|   |--entity_extractor.py
|   |--classifiers.py
|   |--note_generator.py
|-- models/
|-- data/
|   |-- raw/
|   |-- processed/
|-- tests/
|-- docs/
```

## Setup

## Ml Models

## Tool Stack