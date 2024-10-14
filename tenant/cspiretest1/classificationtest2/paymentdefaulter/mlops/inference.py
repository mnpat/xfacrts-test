#  Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import json
import os
import pickle as pkl

import numpy as np
import xgboost as xgb
import tempfile

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from io import StringIO


class ModelHandler(object):

    def __init__(self):
        self.initialized = False

    def model_fn(self, context):
        """
        Deserialize and return fitted model.
        :param context: Initial context contains model server system properties.
        """
        # Debugging
        print("Execution function: model_fn")
        print("model_fn:input: type of context: ", type(context))
        print("model_fn:input: context: ", context)


        self.initialized = True
        properties = context.system_properties
        print("model_fn:properties: type of properties: ", type(properties))
        print("model_fn:properties: properties: ", properties)

        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir")
        print("model_fn:model_dir: model_dir: ", model_dir)
        model_file = "xgboost-model"
        booster = pkl.load(open(os.path.join(model_dir, model_file), "rb"))
        
        return booster


    def input_fn(self, input_data_request):
        """
        The SageMaker XGBoost model server receives the request data body and the content type,
        and invokes the `input_fn`.

        Return a DMatrix (an object that can be passed to predict_fn).
        """
        print("Execution function: input_fn")

        input_data = input_data_request[0]['body']
        # input_data = list(input_data_request[0]['body'])

        # data_str = bytes_data.decode('utf-8')
        
        print("input_fn:input_data: type of input_data: ", type(input_data))
        print("input_fn:input_data: input_data: ", input_data)
        
        temp_file_location = None
        
        try:
            with tempfile.NamedTemporaryFile(delete=False) as csv_file:
                temp_file_location = csv_file.name
                csv_file.write(input_data)

            print("temp_file_location: ", temp_file_location)
            print("type of temp_file_location: ", type(temp_file_location))

            # print("listing file in temp directory: ", os.listdir(temp_file_location))
            # here expecting a bytearray with all the the data
            # converting bytearray to pandas dataframe
            print('11111111111111111111111111111111111111111111111111111111111')
            data_str = input_data.decode('utf-8')
            df = pd.read_csv(StringIO(data_str),dtype = {'acct_num':'str'})
            # print(df.head())
            print("df column names: ",df.columns)
            # deriving new features
            df['is_actual_pay_method_eq_conf_pay_method'] = df['is_auto_paid'].eq(df['is_autopay_enabled'])
            # encoding of the data
            # creating the list of categorical column except acct_num column
            print('222222222222222222222222222222222222222222222222222222222222222222222222')
            categorical_feat_1 = [col for col in df.columns if ((df[col].dtypes=='object') | (df[col].dtypes=='bool')) & (col!='acct_num')]
            # fitting the inference data into label encoder
            le = LabelEncoder()
            for i in categorical_feat_1:
                df[i] = le.fit_transform(df[i])

            # scaling of the inference data
            # list of all numeric features
            numerical_feat=[feat for feat in df.columns if ((df[feat].dtypes=='int64') | (df[feat].dtypes=='float64'))]
            # fitting the inference data into scaling
            scaler = MinMaxScaler(feature_range = (0,1))
            for i in numerical_feat:
                df[i] = scaler.fit_transform(df[i].values.reshape(-1, 1))
            # removing the unwanted column
            remove_col = ['acct_num']
            df = df.drop(remove_col,axis = 1)
            print('3333333333333333333333333333333333333333333333333333333333333333333333')
        finally:
            if temp_file_location and os.path.exists(temp_file_location):
                os.remove(temp_file_location)
        return df
        # else:
        #     raise ValueError("Content type {} is not supported.".format(request_content_type))


    def predict_fn(self, input_data, model):
        """
        SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.

        Return a two-dimensional NumPy array where the first columns are predictions
        and the remaining columns are the feature contributions (SHAP values) for that prediction.
        """
        print("Execution function: predict_fn")
        
        print("predict_fn:input_data: type of input_data: ", type(input_data))
        print("predict_fn:input_data: input_data: ", input_data)

        print("predict_fn:model: type of model: ", type(model))
        print("predict_fn:model: model: ", model)

        prediction = model.predict(input_data)
        # feature_contribs = model.predict(input_data, pred_contribs=True, validate_features=False)
        # output = np.hstack((prediction[:, np.newaxis], feature_contribs))
        output = prediction
        return output


    def output_fn(self, predictions):
        """
        After invoking predict_fn, the model server invokes `output_fn`.
        """
        print("Execution function: output_fn")
        
        print("output_fn:predictions: type of predictions: ", type(predictions))
        print("output_fn:predictions: predictions: ", predictions)

        # content_type = "text/csv"

        # if content_type == "text/csv":
        #     return ",".join(str(x) for x in predictions[0])

        # print("predictions: ", predictions.tolist())

        return [predictions.tolist()]
        # return predictions
        
        # else:
        #     raise ValueError("Content type {} is not supported.".format(content_type))
        

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        print("Execution function: handle")
        
        model = self.model_fn(context)
        model_input = self.input_fn(data)
        model_out = self.predict_fn(model_input, model)
        
        return self.output_fn(model_out)


_service = ModelHandler()


def handle(data, context):
    # if not _service.initialized:
    #     _service.model_fn(context)

    if data is None:
        return None

    return _service.handle(data, context)