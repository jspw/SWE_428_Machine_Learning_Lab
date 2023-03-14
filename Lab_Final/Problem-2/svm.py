from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import category_encoders as ce
from scipy.stats import chi2_contingency
import pandas as pd
import csv
import os
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

labeled_data = "./bank-data/bank-full-labeled.csv"


def convert_raw_data(labeled_data):
    # Load the raw data CSV file
    raw_df = pd.read_csv("./bank-data/bank-full.csv", delimiter=";")

    # Replace semicolons (;) with commas (,) as the delimiter
    raw_df.to_csv("./bank-data/bank-full-comma-separated.csv", index=False)

    # Rename the column headers
    labeled_df = pd.read_csv("./bank-data/bank-full-comma-separated.csv")
    labeled_df.columns = ["age", "job", "marital", "education", "default", "balance",
                          "housing", "loan", "contact", "day", "month", "duration",
                          "campaign", "pdays", "previous", "poutcome", "y"]

    # Save the labeled data as a CSV file
    labeled_df.to_csv(labeled_data, index=False)


convert_raw_data(labeled_data)


df = pd.read_csv(labeled_data)
df.head()


X = df.drop('y', axis=1)
y = np.where(df['y'] == 'yes', 1, 0)


label_encoder = LabelEncoder()
X['job'] = label_encoder.fit_transform(X['job'])
X['marital'] = label_encoder.fit_transform(X['marital'])
X['education'] = label_encoder.fit_transform(X['education'])
X['default'] = label_encoder.fit_transform(X['default'])
X['housing'] = label_encoder.fit_transform(X['housing'])
X['loan'] = label_encoder.fit_transform(X['loan'])
X['contact'] = label_encoder.fit_transform(X['contact'])
X['month'] = label_encoder.fit_transform(X['month'])
X['poutcome'] = label_encoder.fit_transform(X['poutcome'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=19)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=19)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
