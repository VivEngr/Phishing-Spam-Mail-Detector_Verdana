# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load data from csv to pandas data frame
raw_mail_data = pd.read_csv('C:\\Users\\V.O OLATUNJI\\Desktop\\VSCODE\\Projects\\Phishing Mail Detection\\email_data.csv')

# Convert null values to null strings
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# Label encoding
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

X = mail_data['Message']
Y = mail_data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()
model.fit(X_train_features, Y_train)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        input_mail = [request.form["email"]]
        input_data_features = feature_extraction.transform(input_mail)
        prediction = "Ham mail" if model.predict(input_data_features)[0] == 1 else "Spam mail"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
