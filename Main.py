import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from scipy.stats import ttest_1samp
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions



from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

import seaborn as sns

sns.set()

#import csv
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import Sequential
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pickle
from PIL import Image, ImageTk


import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('new.csv')

# Change string into numeric data value
df['thinkofPlayingGameDayLong'] = df['thinkofPlayingGameDayLong'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['spendFreeTimeonGame'] = df['spendFreeTimeonGame'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['feelingofGameAddiction'] = df['feelingofGameAddiction'].apply({'Never' : 1, 'Rarely' : 2, 'Somtimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['playingLongerThenIntended'] = df['playingLongerThenIntended'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['spendLargeTimeonGame'] = df['spendLargeTimeonGame'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['unableToStopPlaying'] = df['unableToStopPlaying'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['gamingToForgetRealLife'] = df['gamingToForgetRealLife'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['gamingToReleaseStress'] = df['gamingToReleaseStress'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['gamingToFeelBetter'] = df['gamingToFeelBetter'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['unableToReduceGaming'] = df['unableToReduceGaming'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['unsuccessfulInfluenceofOthers'] = df['unsuccessfulInfluenceofOthers'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['angryIssue'] = df['angryIssue'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['stressIssue'] = df['stressIssue'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['unsocialIssue'] = df['unsocialIssue'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['fightIssue'] = df['fightIssue'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['breakofRelationship'] = df['breakofRelationship'].apply({'Never' : 1, 'Rarely' : 2, 'Sometime' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['tendToDeceive'] = df['tendToDeceive'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['sleepIssue'] = df['sleepIssue'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['tendToloseHobbies'] = df['tendToloseHobbies'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['tendToNeglectImportantActivities'] = df['tendToNeglectImportantActivities'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['neckandBackPain'] = df['neckandBackPain'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['orthopedicIssues'] = df['orthopedicIssues'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['eyesightIssue'] = df['eyesightIssue'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['hearingIssue'] = df['hearingIssue'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['tendToCocurricularActivities'] = df['tendToCocurricularActivities'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['tendToPresentClass'] = df['tendToPresentClass'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)
df['tendToPeerInterraction'] = df['tendToPeerInterraction'].apply({'Never' : 1, 'Rarely' : 2, 'Sometimes' : 3, 'Often' : 4, 'Very Often' : 5}.get)

df.iloc[:,-10:]
df.info(verbose=True)
df.describe().T





#heatmap
plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap




#addiction Calculation
df['addictionIndicator']= df.iloc[:,:20].sum(axis=1)
df.loc[:, 'addictionIndicator'] = np.where(df.addictionIndicator>=59, 1, 0)
df.head()
#0=not addict, 1=Addict

#Mental Disorder Calculation
df['mentalDisorder']= df.iloc[:,[0,4,6,7,14,15,16,18]].sum(axis=1)
df.loc[:, 'mentalDisorder'] = np.where(df.mentalDisorder>=29, 1, 0)
df.head()
#0=not Mental, 1=Mental

#physical Disorder Calculation
df['physicalDisorder']= df.iloc[:,[20,21,22,23]].sum(axis=1)
df.loc[:, 'physicalDisorder'] = np.where(df.physicalDisorder>=12, 1, 0)
df.head()
#0=not physical, 1=physical

#how many are addicted
count = df['addictionIndicator'].value_counts()
#relative frequency 
tot_addicted = count[1]
tot_nonaddicted = count[0]
tot_participants = tot_addicted + tot_nonaddicted
print('Total Participants:', tot_participants)
print('Total addicted:', tot_addicted)
print('Total non-addicted:', tot_nonaddicted)
print('Percentage of addicted people: {:.2f}%'.format((tot_addicted / tot_participants) * 100))
print('Percentage of non-addicted people: {:.2f}%'.format((tot_nonaddicted / tot_participants) * 100))
#0=not addict, 1=Addict

#among game addictor how many are having mental disorder
print('People having mental disorder with addiction:', len(df[(df['addictionIndicator']==1) & (df['mentalDisorder']==1)]))

#relative frequency of mental disorder among game addictor
perc_mental_disorder = len(df[(df['addictionIndicator']==1) & (df['mentalDisorder']==1)])/tot_addicted
print('People having mental disorder with addiction (Percentage): {:.2f}%'.format(perc_mental_disorder * 100))

#among game addictor how many are having physical disorder
print('People having physical disorder with addiction:', len(df[(df['addictionIndicator']==1) & (df['physicalDisorder']==1)]))

#relative frequency of physical disorder among game addictor
perc_physical_disorder = len(df[(df['addictionIndicator']==1) & (df['physicalDisorder']==1)])/tot_addicted
print('People having physical disorder with addiction (Percentage): {:.2f}%'.format(perc_physical_disorder * 100))

#histogram
p = df.hist(figsize = (20,20))


features = ['Addicted', 'Non-Addicted']
values = [tot_addicted, tot_nonaddicted]
plt.figure(figsize=(10, 10))
plt.bar(features, values, width=0.2)
plt.xlabel('Feature')
plt.ylabel('Frequency')
plt.title('Total number of Addicted and Non-addicted person')
plt.show()


#svm Addiction
X=df.iloc[:, 0:20]
y=df.iloc[:,27]
y

# Assuming X contains features and y contains labels
# Replace X and y with your actual data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize SVC with chosen kernel
svm_model1 = SVC(kernel='linear')

# Train SVC
svm_model1.fit(X_train, y_train)

# Predictions on test set
y_pred = svm_model1.predict(X_test)

# Performance evaluation
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:\n", conf_matrix)

# Example prediction
# Replace X_new with the new data point(s) you want to predict on
X_new = [[1,1,1,1,1,2,1,3,1,2,1,2,1,3,1,4,5,2,4,1]]  # Replace ... with actual feature values
prediction = svm_model1.predict(X_new)
print("Prediction:", prediction)



    #  now save the model for future use
filename = 'Addiction'
pickle.dump(svm_model1, open(filename, 'wb'))











#svm MEntal
V=df.iloc[:, [0, 4, 6, 7, 14, 15, 16, 18]]
n=df.iloc[:,28]
n

# Assuming X contains features and y contains labels
# Replace X and y with your actual data
V_train, V_test, n_train, n_test = train_test_split(V, n, test_size=0.3, random_state=42)

# Initialize SVM with chosen kernel
svm_model2 = SVC(kernel='linear')

# Train SVM
svm_model2.fit(V_train, n_train)

# Predictions on test set
n_pred = svm_model2.predict(V_test)

# Performance evaluation
accuracy5 = accuracy_score(n_test, n_pred)
f15 = f1_score(n_test, n_pred)
precision5 = precision_score(n_test, n_pred)
recall5 = recall_score(n_test, n_pred)
conf_matrix5 = confusion_matrix(n_test, n_pred)

print("Accuracy:", accuracy5)
print("F1 Score:", f15)
print("Precision:", precision5)
print("Recall:", recall5)
print("Confusion Matrix:\n", conf_matrix5)

# Example prediction
# Replace X_new with the new data point(s) you want to predict on
V_new = [[1,1,1,1,1,2,3,3]]  # Replace ... with actual feature values
prediction = svm_model2.predict(V_new)
print("Prediction:", prediction)

    #  now save the model for future use
filename = 'mental'
pickle.dump(svm_model2, open(filename, 'wb'))

















#svm Physical
Z=df.iloc[:, 20:24]
b=df.iloc[:,29]
b

# Assuming X contains features and y contains labels
# Replace X and y with your actual data
Z_train, Z_test, b_train, b_test = train_test_split(Z, b, test_size=0.3, random_state=42)

# Initialize SVM with chosen kernel
svm_model3 = SVC(kernel='linear')

# Train SVM
svm_model3.fit(Z_train, b_train)

# Predictions on test set
b_pred = svm_model3.predict(Z_test)

# Performance evaluation
accuracy6 = accuracy_score(b_test, b_pred)
f16 = f1_score(b_test, b_pred)
precision6 = precision_score(b_test, b_pred)
recall6 = recall_score(b_test, b_pred)
conf_matrix6 = confusion_matrix(b_test, b_pred)

print("Accuracy:", accuracy6)
print("F1 Score:", f16)
print("Precision:", precision6)
print("Recall:", recall6)
print("Confusion Matrix:\n", conf_matrix6)



# Example prediction
# Replace X_new with the new data point(s) you want to predict on
Z_new = [[1,2,1,2]]  # Replace ... with actual feature values
prediction = svm_model3.predict(Z_new)
print("Prediction:", prediction)


    #  now save the model for future use
filename = 'physic'
pickle.dump(svm_model3, open(filename, 'wb'))














null_values = df.isnull().sum()

# Check if there are any null values in the DataFrame
if null_values.any():
    print("There are null values in the DataFrame:")
    print(null_values)
else:
    print("No null values in the DataFrame.")
    
    
    df.info(verbose=True)
