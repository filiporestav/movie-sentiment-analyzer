import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import matplotlib.pyplot as plt
import re
import json
import numpy as np
from gensim.models import Word2Vec

class NaiveBayesClassifierWord2Vec():
    """This class is classifying texts as positive or negative, using the Naive Bayes algorithm
    together with Word2vec."""
    def __init__(self, lemmatization=True, stemming=False):
        self.unwanted_words = set(stopwords.words("english"))
        self.unwanted_words.remove("not")

        self.train_data = self.load_data('data/train')
        self.test_data = self.load_data('data/test')

        self.vector_size = 1000
        self.word2vec_model = None

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
        self.model = LogisticRegression(max_iter=1000, random_state=123)
        self.model.fit(self.X_train, self.y_train)
        print("Training completed.")

    def setup_data(self):
        """Sets up the data by training a Word2vec model and transforming the
        texts with the trained model."""
        self.train_data['review'] = self.train_data['review'].apply(self.pre_process)
        self.test_data['review'] = self.test_data['review'].apply(self.pre_process)

        # Train Word2Vec model
        self.word2vec_model = Word2Vec(sentences=self.train_data['review'].apply(word_tokenize),
                                       vector_size=self.vector_size, window=3, min_count=1, workers=4, seed=123)

        # Transform the training and test data using Word2Vec
        self.X_train = self.transform_word2vec(self.train_data['review'])
        self.y_train = self.train_data['label']
        self.X_test = self.transform_word2vec(self.test_data['review'])
        self.y_test = self.test_data['label']

    def transform_word2vec(self, texts):
        """Transforms a list of pre-processed texts into numerical feature vectors
        using a pre-trained Word2Vec model.

        Args:
            texts (list): A list of pre-processed texts.

        Returns:
            np.ndarray: A 2D numpy array where each row corresponds to the average
            word vector representation of a text.
        """
        # Transform each review into the average word vector
        word_vectors = []
        for text in texts:
            vectors = [self.word2vec_model.wv[word] for word in word_tokenize(text) if word in self.word2vec_model.wv]
            if vectors:
                text_vector = np.mean(vectors, axis=0)
                word_vectors.append(text_vector)
            else:
                # If none of the words are in the vocabulary, use a zero vector
                word_vectors.append(np.zeros(self.vector_size))
        return np.vstack(word_vectors)

    def load_data(self, directory):
        """Loads textual data from a directory containing subdirectories for each class.

        Args:
            directory (str): The path to the main directory containing subdirectories
            for each class ('pos' and 'neg').

        Returns:
            pd.DataFrame: A pandas DataFrame containing two columns - 'review' for the
            textual content of reviews and 'label' for the corresponding class labels ('pos' or 'neg').
        """
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

    def pre_process(self, text):
        """Pre-processes a given text by applying lemmatization, stemming, and word filtering.

        Args:
            text (str): The input text to be pre-processed.

        Returns:
            str: The pre-processed text after lemmatization, stemming (if enabled),
            and removal of unwanted words and non-alphabetic characters.
        """
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
        """Predicts a label (positive or negative) given a text."""
        text_vector = self.transform_word2vec([text])
        return self.model.predict(text_vector)

    def evaluate_model(self, y_true, y_pred):
        """Evaluates the model given the predicted labels and the correct labels.
        Also plots a confusion matrix."""
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

        # Plot the heatmap with percentages
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['pos', 'neg'], yticklabels=['pos', 'neg'])
        plt.xlabel('Predicted review')
        plt.ylabel('True review')
        plt.title('Confusion Matrix')
        plt.show()

    def evaluate_on_json_data(self, json_file_path):
        """Evaluates the trained model on the json-files from Amazon reviews."""
        # Load JSON data
        json_data = self.load_json_data(json_file_path)

        # Filter out neutral data points (rating 3)
        json_data_filtered = json_data[(json_data['overall'] <= 2) | (json_data['overall'] >= 4)]

        # Extract relevant features for prediction
        json_reviews = json_data_filtered['reviewText'].apply(self.pre_process)

        # Transform the reviews using Word2Vec
        json_vectors = self.transform_word2vec(json_reviews)

        # Predict sentiments
        json_predictions = self.model.predict(json_vectors)

        # Create binary labels for evaluation
        y_true = ['neg' if overall <= 2 else 'pos' for overall in json_data_filtered['overall']]

        # Evaluate the model on filtered JSON data
        self.evaluate_model(y_true, json_predictions)

    def load_json_data(self, file_path):
        """Loads json-data into a Pandas Dataframe object."""
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
        return pd.DataFrame(data)

    def run_demo(self):
        """Runs a demo with content from the user, which the model in turn tries to predict."""
        while True:
            review = input("Write a review (press enter to exit): ")
            if review == "":
                break
            print(self.predict_label(review))

if __name__ == "__main__":
    classifier = NaiveBayesClassifierWord2Vec()

    # Evaluate the Model on Test Data
    y_pred = classifier.model.predict(classifier.X_test)
    classifier.evaluate_model(classifier.y_test, y_pred)

    #classifier.plot_top_features()

    print(classifier.predict_label("This was a really good movie."))  # Should be pos
    print(classifier.predict_label("This was a really bad movie."))  # Should be neg

    # Testing for some other general products
    print(classifier.predict_label("I got this as my second charging cable for a MacBook Air M2. A working product. I really like how Apple went back to the MagSafe charging port. Fast charging when used together with a sufficiently powerful power source."))
    print(classifier.predict_label("The original one stopped working a few weeks ago. I received delivery of the MacBook Air M2 at the end of February . It’s not been out of my house and lives on a desk. Been attempting to charge with the USB cable but it’s not very good. I’m shocked at the price for such a low-quality product."))

    # Evaluate the Model on JSON Data, from Amazon (All Beauty category, ~5k reviews): https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
    classifier.evaluate_on_json_data('data/test/All_Beauty_5.json')

    classifier.run_demo()
