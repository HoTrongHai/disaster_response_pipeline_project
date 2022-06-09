import sys

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine

import nltk

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle


def load_data(database_filepath):
    """
    Load data from database
    :param database_filepath: Database file path
    :return: X: X input, Y label (multiple dimension) and list of labels
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("DisasterMessageCategory", engine)

    X = df['message'].values
    Y_frame = df.drop(labels=['id', 'message', 'original', 'genre'], axis=1)
    Y = Y_frame.to_numpy()

    category_names = list(Y_frame.columns)

    return X, Y, category_names



def tokenize(text):
    """
    Tokenize text string into token: lower case >> remove stop words >> lemmatize
    :param text: Given string
    :return: List of token
    """
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Create the model using pipeline and GridSeach.
    The pipeline: Count Vectorize >> TfidfTransformer >> Mutiple output classifier
    :return: The built model
    """

    pipeline = Pipeline([
        # ('features', FeatureUnion([
        #
        #     ('text_pipeline', Pipeline([
        #         ('vect', CountVectorizer(tokenizer=tokenize)),
        #         ('tfidf', TfidfTransformer())
        #     ])),
        #
        #     ('starting_verb', StartingVerbExtractor())
        # ])),

        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    print(pipeline.get_params())
    parameters = {
        'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model by using f1 score
    :param model: The trained model
    :param X_test: X test input
    :param Y_test: Y test label
    :param category_names: List of category/label
    :return: None
    """
    y_pred = model.predict(X_test)

    for i, label_name in enumerate(category_names):
        cr_y_i = classification_report(Y_test[:, i], y_pred[:, i])
        print(cr_y_i)


def save_model(model, model_filepath):
    """
    Save trained model to file path
    :param model: The trained model
    :param model_filepath: saved file path
    :return: None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
