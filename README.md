# Game Addiction Predictor
This repository contains a machine learning project developed for a university course on Artificial Intelligence and Machine Learning. The project's goal is to analyze a dataset to predict gaming addiction and its potential impact on an individual's mental and physical health. It includes a user-friendly questionnaire-based application to make real-time predictions.

# Features
Addiction Prediction: Predicts whether a person is likely to be addicted to gaming based on their responses to a questionnaire.

Health Correlation: Predicts potential mental and physical health problems associated with gaming behavior.

Machine Learning Model: Utilizes a Support Vector Classifier (SVC) for its predictive models.

Graphical User Interface (GUI): A user-friendly interface built with Tkinter allows individuals to submit their answers and view their results.

Data Visualization: Includes functions to display a correlation heatmap of the dataset and other relevant graphs.

# Technologies Used
Python 3.x

scikit-learn

pandas

numpy

matplotlib

seaborn

Tkinter

joblib

pickle

# Project Structure
aiproject.py: The main script for data preprocessing, model training, and performance evaluation. It trains three SVC models for addiction, mental health, and physical health, then saves them using pickle.

gui.py: The script that creates the Tkinter-based GUI. It loads the pre-trained models and uses them to make predictions based on user input from the questionnaire.

new.csv: The dataset used to train and evaluate the models. It contains the questionnaire responses.

213014020_123014027.pdf: The official project report submitted to the university, detailing the problem statement, methodology, and results.

# Installation and Setup
Clone this repository to your local machine :

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
or download as zip

Install the required Python libraries using pip:

pip install pandas numpy scikit-learn matplotlib seaborn

Note: Tkinter is a standard library and usually comes with Python.

Run the main project script to train and save the models:

python aiproject.py

This will generate three model files (Addiction, mental, and physical) in your project directory.

# Usage
Once the models are saved, you can run the GUI application:

python gui.py

A window will pop up with a questionnaire. Fill out the form by selecting an option for each question and then click "Submit" to see your predictions and the performance metrics of the models. You can also click "Show Heatmap" to view a correlation matrix of the dataset.

# Model Performance
Based on the project report, the trained SVC models achieved the following performance metrics on the test dataset:

| Category | Accuracy | F1 Score | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Addiction** | 0.9928 | 0.9920 | 0.9841 | 1.0 |
| **Mental Health** | 0.9928 | 0.9928 | 0.9928 | 0.9928 |
| **Physical Health** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0
