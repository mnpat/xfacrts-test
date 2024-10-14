import pandas as pd
import numpy as np
import io
import os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import traceback
import re
import sys
import time
import json
import logging
from time import strftime, gmtime
import argparse
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
# from pyhive import hive
import warnings
warnings.filterwarnings("ignore")
from datetime import date
from pandas.io.sql import DatabaseError
import datetime


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

BASE_PATH = os.path.join("/", "opt", "ml")
PROCESSING_PATH = os.path.join(BASE_PATH, "processing")
PROCESSING_PATH_INPUT = os.path.join(PROCESSING_PATH, "input")
PROCESSING_PATH_OUTPUT = os.path.join(PROCESSING_PATH, "output")


def extract_data(file_path, percentage=100):
    print("Function Name: extract_data")
    print(f"Function Name: extract_data | file_path: {file_path} ")
    try:
        files = [f for f in listdir(file_path) if isfile(join(file_path, f)) and f.endswith(".csv")]
        LOGGER.info("{}".format(files))

        frames = []

        for file in files:
            df = pd.read_csv(
                os.path.join(file_path, file),
                sep=",",
                quotechar='"',
                quoting=csv.QUOTE_ALL,
                escapechar='\\',
                encoding='utf-8',
                on_bad_lines='skip',
                dtype={'acct_num':object}
            )

            df = df.head(int(len(df) * (percentage / 100)))

            frames.append(df)

        df = pd.concat(frames)

        return df
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

def load_data(df, file_path, file_name, header=True):
    print("Function Name: load_data")
    print(f"Function Name: load_data | file_path: {file_path}  | file_name: {file_name}")
    print(list(df.columns))
    df.head()
    
    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        path = os.path.join(file_path, file_name + ".csv")

        LOGGER.info("Saving file in {}".format(path))

        df.to_csv(
            path,
            index=False,
            header=header,
            encoding="utf-8",
        )
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e


def transform_data(df):
    print("Function Name: transform_data")
    print(list(df.columns))
    df.head()
    try:

        LOGGER.info("Original count: {}".format(len(df.index)))        
        # deriving new features
        df['is_actual_pay_method_eq_conf_pay_method'] = df['is_auto_paid'].eq(df['is_autopay_enabled'])
        
        # defining target column
        target_col = 'target'
        
        # encoding of the data
        # creating the list of categorical column except acct_num column
        categorical_feat_1 = [col for col in df.columns if ((df[col].dtypes=='object') | (df[col].dtypes=='bool')) & (col!='acct_num')]
        # fitting the taining data into label encoder
        le = LabelEncoder()
        for i in categorical_feat_1:
            df[i] = le.fit_transform(df[i])
            
        # scaling of the training data
        # list of all numeric features except target column
        numerical_feat = [feat for feat in df.columns if ((df[feat].dtypes=='int64') | (df[feat].dtypes=='float64'))
                  & ~(feat==target_col)]
        # fitting the training data into scaling
        scaler = MinMaxScaler(feature_range = (0,1))
        for i in numerical_feat:
            df[i] = scaler.fit_transform(df[i].values.reshape(-1, 1))
        
        # removing the unwanted column
        remove_col = ['acct_num', 'bill_date']
        target_col = ['target']
        
        df = df.drop(remove_col, axis=1)
        df_cols = list(df.columns)
        df_cols.remove(target_col[0])
        df_cols = target_col + df_cols
        print(df_cols)

        df = df.loc[:,df_cols]
        df.head()

        return df
    
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    LOGGER.info("Arguments: {}".format(args))
    print("Received arguments {}".format(args))

    df = extract_data(PROCESSING_PATH_INPUT, 100)

    df = transform_data(df)

    data_train, data_validation = train_test_split(df, test_size=0.2)

    load_data(data_train, os.path.join(PROCESSING_PATH_OUTPUT, "train"), "train")
    load_data(data_validation, os.path.join(PROCESSING_PATH_OUTPUT, "validation"), "validation")

    # Creating test dataset for batch inference
    data_test = data_validation.drop('target', axis=1)
    load_data(data_test, os.path.join(PROCESSING_PATH_OUTPUT, "inference"), "data", True)
