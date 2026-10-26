# Emotion Detector



## Project Overview

This project detects emotions in text using machine learning and deep learning models. It uses the GoEmotions dataset to predict one or more emotions expressed in a sentence.



# Project Goal

The goal of this project is to explore and compare traditional machine learning and deep learning approaches for multi-label emotion detection in text.



## Models

The application has three models:

Naive Bayes – Uses CountVectorizer to convert text into numerical features.

GRU – Uses a recurrent neural network to classify emotions.

LSTM – Uses a recurrent neural network designed to learn patterns in sequential text.



## Features

Text cleaning and preprocessing.

Exploratory data analysis with visualizations.

Comparison of text vectorization methods, including CountVectorizer and TF-IDF.

Model evaluation using Micro F1 and Macro F1 scores.

Interactive web application built with Streamlit.

Adjustable probability threshold for emotion predictions.

Support for multi-label emotion classification.



## Dataset

The project uses the GoEmotions dataset stored at:

archive/go_emotions_dataset.csv

The dataset contains text samples with corresponding emotion labels.

GoEmotions dataset is a public dataset you can find here: https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset



## Installation

Install the required Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn streamlit torch joblib



## How to Run

1. Run the data analysis and model comparison

```
python .\emoanalysis.py
```

This performs exploratory data analysis, generates visualizations, and compares different text preprocessing and classification configurations.

2. Run the Emotion Detector application

```
streamlit run .\emotiondetectmlpr.py
```

Choose a model, enter text, adjust the probability threshold, and click Predict to see the predicted emotions.



## Model Files

The application saves trained models locally to avoid retraining them every time:

nb_model.pkl – Naive Bayes model.

gru_model.pt – GRU model.

lstm_model.pt – LSTM model.

If these files do not exist, the application trains the selected model and saves it.



## Technologies Used

Python

Pandas and NumPy

Matplotlib and Seaborn

Scikit-learn

PyTorch

Streamlit

GoEmotions dataset
