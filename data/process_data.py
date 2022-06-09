import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

def _transform_category(df_category: pd.DataFrame):
    """
    Transform category values into multiple columns. For example current category value is "related-1;request-0" will be
    change into columns ["related", "request"] with values [1, 0]
    :param df_category: The dataframe contain `categories` columns
    :return: Transformed dataframe
    """
    df_category_expanded = df_category["categories"].str.split(";", expand=True)

    def _extract_label_names(record):
        _label_names = [re.sub('-[0-9]+', "", col) for col in record]
        return _label_names

    def _extract_label_values(series: pd.Series, label_name):
        return series.apply(lambda c: c.replace(f"{label_name}-", ""))

    # Get label names and set new columns
    label_names = _extract_label_names(df_category_expanded.iloc[0])
    df_category_expanded.columns = label_names

    # Convert label values of each series into numberic
    for label_name in label_names:
        df_category_expanded[label_name] = _extract_label_values(df_category_expanded[label_name], label_name).astype('int8')


    return pd.concat([df_category.drop(labels=['categories'], axis=1), df_category_expanded], axis=1)


def _merge_dataset(df_message, df_category):
    """
    Merge 2 dataframe with condition 'id's are same
    :param df_message: The message dataframe
    :param df_category: The category dataframe
    :return: Merged dataframe
    """
    return pd.merge(df_message, df_category, on=['id'])

def check_duplicated(_df: pd.DataFrame):
    """
    Check duplicated records in the dataframe
    :param _df: The merged dataframe
    :return: None. The dataframe will be removed inplace
    """

    duplicate_rows = _df[_df.duplicated()]
    print(f'Found {len(duplicate_rows.index)} duplicated values...')
    print('We will removed these duplicated values...')

    _df.drop_duplicates(inplace=True)
    return _df

def load_data(messages_filepath, categories_filepath):
    """
    Load the message and category dataframes from file paths
    :param messages_filepath: messsage file path
    :param categories_filepath: category file path
    :return: Tuple of message and category dataframes
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return _merge_dataset(messages, categories)


def clean_data(df):
    """
    Transform category column and remove duplicate from merge dataframe
    :param df: The merged dataframe
    :return: The cleaned dataframe
    """
    df = _transform_category(df)
    check_duplicated(df)
    return df


def save_data(df, database_filepath):
    """
    Save dataframe into database with specified file path
    :param df: The cleaned dataframe
    :param database_filepath: The datafile file path
    :return: None
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql("DisasterMessageCategory", engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()