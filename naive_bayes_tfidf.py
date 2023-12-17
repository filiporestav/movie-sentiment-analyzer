import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from nltk import word_tokenize
from nltk.corpus import stopwords, words as nltk_words
from nltk.stem import WordNetLemmatizer, PorterStemmer
import matplotlib.pyplot as plt
import re
import json
import numpy as np

class NaiveBayesClassifierTFIDF():
    """Class which uses Term Frequency-Inverse Document Frequency (TF-IDF) to train a Naive Bayes model
    to predict the sentiment of reviews"""
    def __init__(self, lemmatization=True, stemming=False):
        self.unwanted_words = set(stopwords.words("english"))
        self.unwanted_words.remove("not")
        self.nltk_word_set = set(nltk_words.words())  # Set of words from NLTK

        self.total_words = 0 # Number of 'real' words in the vocabulary provided in the dataset

        print("Loading data...")
        self.train_data = self.load_data('data/train')
        self.test_data = self.load_data('data/test')

        self.average_word_polarity = self.load_word_exp_rating()

        self.vectorizer = TfidfVectorizer(stop_words=list(self.unwanted_words), lowercase=True, ngram_range=(1, 3), min_df=4)

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

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
        self.model = MultinomialNB()
        self.model.fit(self.X_train, self.y_train)
        print("Training completed.")

    def setup_data(self):
        """Sets up the data by transforming the reviews to document-term matrix."""
        self.train_data['review'] = self.train_data['review'].apply(self.pre_process)
        self.test_data['review'] = self.test_data['review'].apply(self.pre_process)

        self.X_train = self.vectorizer.fit_transform(self.train_data['review'])
        self.y_train = self.train_data['label']

        self.X_test = self.vectorizer.transform(self.test_data['review'])
        self.y_test = self.test_data['label']

    def load_data(self, directory):
        """Loads the data into a Pandas Dataframe object."""
        reviews = []
        labels = []
        for label in ['pos', 'neg']:
            folder_path = os.path.join(directory, label)
            for filename in os.listdir(folder_path):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    review = file.read()
                    reviews.append(review)
                    labels.append(label)
        return pd.DataFrame({'review': reviews, 'label': labels})

    def load_word_exp_rating(self):
        """Loads the average word polarity rating, using the retrieved data from Stanford AI Lab"""
        dictionary = {}
        vocab = 'data/imdb.vocab'
        word_polarity = 'data/imdbEr.txt'
        with open(vocab, 'r', encoding='utf-8') as file1, open(word_polarity, 'r', encoding='utf-8') as file2:
            for line1, line2 in zip(file1, file2):
                word = line1.strip()
                avg_word_polarity = line2.strip()
                avg_word_polarity = float(avg_word_polarity)

                if word in self.nltk_word_set:
                    self.total_words += 1
                    dictionary[word] = avg_word_polarity
        return dictionary

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

    def plot_most_pos_and_neg_words(self, top_n=20):
        """Plots the most positive and negative words, according to the data from Stanford AI Lab."""
        sorted_words = sorted(self.average_word_polarity.items(), key=lambda x: x[1], reverse=True)
        top_positive_words = sorted_words[:top_n]
        top_negative_words = sorted_words[-top_n:]

        positive_words, positive_polarity = zip(*top_positive_words)
        negative_words, negative_polarity = zip(*top_negative_words)

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.barh(positive_words, positive_polarity, color='green')
        plt.xlabel('Average Polarity')
        plt.title(f'Top {top_n} Positive Words')
        plt.xlim(0, max(positive_polarity) + 0.1)

        plt.subplot(1, 2, 2)
        plt.barh(negative_words, negative_polarity, color='red')
        plt.xlabel('Average Polarity')
        plt.title(f'Top {top_n} Negative Words')

        plt.tight_layout()
        plt.show()

    def evaluate_model(self, y_true, y_pred):
        """Evaluates the model by comparing the predicted labels with the correct labels.
        Prints out the accuracy, precision, recall and F1-score."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label='pos')
        recall = recall_score(y_true, y_pred, pos_label='pos')
        f1 = f1_score(y_true, y_pred, pos_label='pos')

        print(f'Accuracy: {accuracy:.2%}')
        print(f'Precision: {precision:.2%}')
        print(f'Recall: {recall:.2%}')
        print(f'F1 Score: {f1:.2%}')

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['pos', 'neg'])

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['pos', 'neg'], yticklabels=['pos', 'neg'])
        plt.xlabel('Predicted review')
        plt.ylabel('True review')
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

    def plot_top_features(self, n=20):
        """Plots the most positive and negative features (in this case words) from our matrix."""
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        class_labels = self.model.classes_

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))  # Create two subplots side by side

        for i, class_label in enumerate(class_labels):
            # Use log probabilities to get feature importances
            feature_log_probs = self.model.feature_log_prob_[i]
            top_n_features_idx = feature_log_probs.argsort()[-n:][::-1]
            top_features = feature_names[top_n_features_idx]
            top_log_probs = feature_log_probs[top_n_features_idx]

            ax = axs[i]
            ax.barh(top_features, np.exp(top_log_probs), label=f'Class: {class_label}', color='green' if class_label == 'pos' else 'red')
            ax.set_xlabel('Probability')
            ax.set_ylabel('Feature')
            ax.set_title(f'Top {n} Features for {class_label} Class')
            ax.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    classifier = NaiveBayesClassifierTFIDF(lemmatization=True, stemming=False)

    classifier.plot_most_pos_and_neg_words()

    # Evaluate the Model on Test Data
    y_pred = classifier.model.predict(classifier.X_test)
    classifier.evaluate_model(classifier.y_test, y_pred)
    accuracy = accuracy_score(classifier.y_test, y_pred)

    classifier.plot_top_features()

    print(classifier.predict_label("This was a really good movie."))  # Should be pos
    print(classifier.predict_label("This was a really bad movie."))  # Should be neg

    # Testing for some other general products
    print(classifier.predict_label("I got this as my second charging cable for a MacBook Air M2. A working product. I really like how Apple went back to the MagSafe charging port. Fast charging when used together with a sufficiently powerful power source."))
    print(classifier.predict_label("The original one stopped working a few weeks ago. I received delivery of the MacBook Air M2 at the end of February . It’s not been out of my house and lives on a desk. Been attempting to charge with the USB cable but it’s not very good. I’m shocked at the price for such a low-quality product."))

    # Evaluate the Model on JSON Data, from Amazon (All Beauty category, ~5k reviews): https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
    classifier.evaluate_on_json_data('data/test/All_Beauty_5.json')

    classifier.run_demo()