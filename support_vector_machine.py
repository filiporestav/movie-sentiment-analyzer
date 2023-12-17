import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import json
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Adjust if you want to train on a subset of the data, for faster execution.
SAMPLE_FRACTION = 0.1

class SVMClassifier():
    """This class is is classifying texts based on the Support Vector Machine (SVM) algorithm."""
    def __init__(self, lemmatization=True, stemming=False):
        self.unwanted_words = set(stopwords.words("english"))
        self.unwanted_words.remove("not")

        # The Pandas DataFrame tables holding our training and test data
        self.train_data, self.test_data = self.load_data('data/train'), self.load_data('data/test')

        self.vectorizer = TfidfVectorizer(stop_words=list(self.unwanted_words), lowercase=True, ngram_range=(1, 3), min_df=4)
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

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
        self.model = SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced', verbose=True, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print("Training completed.")

    def setup_data(self, sample_fraction=SAMPLE_FRACTION, random_state=42):
        """Method which sets up the data with the help of the vectorizer, converting the texts to document-term-matrix.
        The proportion of data to be used can be adjusted with the SAMPLE_FRACTION parameter"""
        train_sample = self.train_data.sample(frac=sample_fraction, random_state=random_state)
        test_sample = self.test_data.sample(frac=sample_fraction, random_state=random_state)

        self.X_train, self.y_train = self.vectorizer.fit_transform(train_sample['review']), train_sample['label']
        self.X_test, self.y_test = self.vectorizer.transform(test_sample['review']), test_sample['label']

        train_sample['review'] = train_sample['review'].apply(self.pre_process)
        test_sample['review'] = test_sample['review'].apply(self.pre_process)

        self.X_train, self.y_train = self.vectorizer.transform(train_sample['review']), train_sample['label']
        self.X_test, self.y_test = self.vectorizer.transform(test_sample['review']), test_sample['label']

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

    def evaluate_model(self, y_true, y_pred):
        """Evaluates the model by comparing the predicted labels with the correct labels.
        Prints out the accuracy, precision, recall and F1-score."""
        accuracy, precision, recall, f1 = accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, pos_label='pos'), recall_score(y_true, y_pred, pos_label='pos'), f1_score(y_true, y_pred, pos_label='pos')
        print(f'Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}')
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plots the confusion matrix given the predicted labels and correct labels."""
        cm = confusion_matrix(y_true, y_pred, labels=['pos', 'neg'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['pos', 'neg'], yticklabels=['pos', 'neg'])
        plt.xlabel('Predicted sentiment')
        plt.ylabel('True sentiment')
        plt.title('Confusion Matrix')
        plt.show()

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
    
    def run_demo(self):
        """Runs a demo with content from the user, which the model in turn tries to predict."""
        while True:
            review = input("Write a review (press enter to exit): ")
            if review == "": break
            print(self.predict_label(review))

if __name__ == "__main__":
    classifier = SVMClassifier(lemmatization=True, stemming=False)

    # Predict and evaluate on the test set
    y_pred = classifier.model.predict(classifier.X_test)
    classifier.evaluate_model(classifier.y_test, y_pred)

    # Plot Confusion Matrix
    classifier.plot_confusion_matrix(classifier.y_test, y_pred)

    # Testing for some other general products
    print(classifier.predict_label("I got this as my second charging cable for a MacBook Air M2. A working product. I really like how Apple went back to the MagSafe charging port. Fast charging when used together with a sufficiently powerful power source."))
    print(classifier.predict_label("The original one stopped working a few weeks ago. I received delivery of the MacBook Air M2 at the end of February. It’s not been out of my house and lives on a desk. Been attempting to charge with the USB cable but it’s not very good. I’m shocked at the price for such a low-quality product."))

    # Run the demo
    classifier.run_demo()
