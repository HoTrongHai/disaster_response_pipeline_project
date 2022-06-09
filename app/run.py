import json
import pickle

import pandas as pd
import plotly
from flask import Flask
from flask import render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar, Pie
# from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()
#
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)
#
#     return clean_tokens


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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')

# sql_path = "D:\hotronghai\OneDrive\Python\DS_Nano_WebDev\disaster_response_pipeline_project_1\data\DisasterResponse.db"
# engine = create_engine(f'sqlite:///{sql_path}')

df = pd.read_sql_table('DisasterMessageCategory', engine)


# class MyCustomUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == "__main__":
#             module = "train_classifier"
#         return super().find_class(module, name)
#
#
# def load_model(model_file_path):
#     with open(model_file_path, 'rb') as f:
#         unpickler = MyCustomUnpickler(f)
#         return unpickler.load()
#         # return pickle.load(f)

def load_model(model_file_path):
    with open(model_file_path, 'rb') as f:
        return pickle.load(f)


# load model
# model = joblib.load("../models/your_model_name.pkl")

model = load_model("../models/classifier.pkl")


# model_file_path = "D:\hotronghai\OneDrive\Python\DS_Nano_WebDev\disaster_response_pipeline_project_1\models\classifier.pkl"
# model = load_model(model_file_path)

def _group_by_category(_df: pd.DataFrame):
    category_names = _df.columns[4:]
    return {category_name: _df[category_name].sum() for category_name in category_names}


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_sum = _group_by_category(df)

    print(category_sum.keys())

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=list(category_sum.keys()),
                    y=list(category_sum.values())
                )
            ],

            'layout': {
                'title': 'Distribution of Message Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
