import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import matplotlib.pyplot as plt 
import re
import json
import numpy as np

class XGBoostClassifier():
    """Class which implements the eXtreme Gradient Boosting model to classify
    text as either positive or negative."""
    def __init__(self, lemmatization=True, stemming=False):
        self.unwanted_words = set(stopwords.words("english"))
        self.unwanted_words.remove("not")

        self.train_data, self.test_data = self.load_data('data/train'), self.load_data('data/test')

        self.vectorizer = TfidfVectorizer(stop_words=list(self.unwanted_words), lowercase=True, ngram_range=(1, 3), min_df=4)
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

        self.label_encoder = LabelEncoder()

        # Boolean values if we want stemming and/or lemmatization
        self.lemmatization = lemmatization
        self.stemming = stemming

        # Define our lemmatization and stemming helper objects
        if self.lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        if self.stemming:
            self.stemmer = PorterStemmer()

        print("Setting up model and training on data...")
        self.setup_data()
        self.model = XGBClassifier()
        self.model.fit(self.X_train, self.y_train)
        print("Training completed.")

    def setup_data(self):
        """Method which sets up the data with the help of the vectorizer, converting the texts to document-term-matrix.
        The proportion of data to be used can be adjusted with the SAMPLE_FRACTION parameter"""
        # Preprocess the text data
        self.train_data['review'] = self.train_data['review'].apply(self.pre_process)
        self.test_data['review'] = self.test_data['review'].apply(self.pre_process)

        # Vectorize the data
        self.X_train = self.vectorizer.fit_transform(self.train_data['review'])
        self.y_train = self.label_encoder.fit_transform(self.train_data['label'])
        self.X_test = self.vectorizer.transform(self.test_data['review'])
        self.y_test = self.label_encoder.transform(self.test_data['label'])

    def load_data(self, directory):
        """Loads the data into a Pandas Dataframe, for easy storage and data handling."""
        reviews, labels = [], []
        for label in ['pos', 'neg']:
            folder_path = os.path.join(directory, label)
            for filename in os.listdir(folder_path):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    reviews.append(file.read())
                    labels.append(label)
        return pd.DataFrame({'review': reviews, 'label': labels})

    def pre_process(self, text):
        """Pre-processes the given text according to the classifier's parameters."""
        if isinstance(text, str):
            if self.lemmatization:
                words = [self.lemmatizer.lemmatize(word).lower() for word in word_tokenize(text)]
            if self.stemming:
                words = [self.stemmer(word).lower() for word in word_tokenize(text)]

            filtered_words = [re.sub(r'[^a-zA-Z]', '', word) for word in words if word.lower() not in self.unwanted_words]
            return " ".join(filtered_words)
        else:
            return ""

    def predict_label(self, text):
        """Predicts the label given a text."""
        return self.model.predict(self.vectorizer.transform([text]))

    def evaluate_on_json_data(self, json_file_path):
        """Evaluates the model on the json-files from Amazon reviews."""
        # Load JSON data
        json_data = self.load_json_data(json_file_path)

        # Filter out neutral data points
        json_data_filtered = json_data[(json_data['overall'] <= 2) | (json_data['overall'] >= 4)]

        # Extract relevant features for prediction
        json_reviews = json_data_filtered['reviewText'].apply(self.pre_process)

        # Predict sentiments
        json_predictions = self.model.predict(self.vectorizer.transform(json_reviews))

        # Create binary labels for evaluation
        y_true = ['neg' if overall <= 2 else 'pos' for overall in json_data_filtered['overall']]

        # Evaluate the model on filtered JSON data
        self.evaluate_model(y_true, json_predictions)

    def load_json_data(self, file_path):
        """Loads the json-data into a Pandas Dataframe object."""
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
        return pd.DataFrame(data)

    def evaluate_model(self, y_true, y_pred):
        """Evaluates the model by printing the accuracy, precision, recall and F1-score, given the predicted labels and correct labels.
        Also plots the confusion matrix."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f'Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}')

        # Plot the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        class_labels = ['Negative', 'Positive']

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


if __name__ == "__main__":
    classifier = XGBoostClassifier()
    y_pred = classifier.model.predict(classifier.X_test)
    classifier.evaluate_model(classifier.y_test, y_pred)

    # Testing for some other general products
    print(classifier.predict_label("I got this as my second charging cable for a MacBook Air M2. A working product. I really like how Apple went back to the MagSafe charging port. Fast charging when used together with a sufficiently powerful power source."))
    print(classifier.predict_label("The original one stopped working a few weeks ago. I received delivery of the MacBook Air M2 at the end of February. It’s not been out of my house and lives on a desk. Been attempting to charge with the USB cable but it’s not very good. I’m shocked at the price for such a low-quality product."))