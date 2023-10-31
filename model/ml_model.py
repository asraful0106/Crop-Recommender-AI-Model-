import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path =r"https://drive.google.com/uc?export=download&id=1TG6nhWQB2EPYm0RaqrvCvrJysTf8ajju"
df = pd.read_csv(path)
x = df.iloc[:, 0:-1]
y = df.iloc[:,-1]
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
y=LE.fit_transform(y)
from sklearn.model_selection import train_test_split as split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
x_train_rf, x_test_rf, y_train_rf, y_test_rf = split(x, y, test_size=0.20, random_state=1)


model_rf = RandomForestClassifier(random_state=50)

model_rf.fit(x_train_rf, y_train_rf)

y_prediction_rf = model_rf.predict(x_test_rf)

print(f"Original Dataset Accuracy: {accuracy_score(y_test_rf, y_prediction_rf)}")

from joblib import dump
output_path = os.path.abspath('G:\DIU\cropRecomendationSystem\savedModels\model_rf.joblib')
dump(model_rf, output_path)