import os
import pandas as pd
import numpy as np
import sys

from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class data_transformation_config:
    preprocessor_obj_filepath = os.path.join("artifacts", "preprocessor.pkl")

class data_transformation:
   def __init__(self):
       self.data_transformation_config = data_transformation_config()

   def get_data_transformation_obj(self):
       try:
           numerical_features = ['reading_score', 'writing_score']
           categorical_features= ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

           numerical_pipeline = Pipeline(
               steps=[
                   ("imputer", SimpleImputer(strategy="median")),
                   ("scaler", StandardScaler())
               ]
           )
           categorical_pipeline = Pipeline(
               steps=[
                   ("imputer", SimpleImputer(strategy="most_frequent")),
                   ("encoding", OneHotEncoder(sparse_output=True)),  
                   ("scaler", StandardScaler(with_mean=False))  
               ]
           )
           preprocessor = ColumnTransformer(
               [
                   ("numerical_pipeline", numerical_pipeline, numerical_features),
                   ("categorical_pipeline", categorical_pipeline, categorical_features)
               ]
           )
           return preprocessor
    
       except Exception as e:
           raise CustomException(e, sys)
    
   def initiate_data_transformation(self, train_path, test_path):
       try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        preprocessing_obj = self.get_data_transformation_obj()
        # print(train_df.columns)
        # print(test_df.columns)

        target_column_name="math_score"
        numerical_columns = ["writing_score", "reading_score"]

        input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
        target_feature_train_df=train_df[target_column_name]

        input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
        target_feature_test_df=test_df[target_column_name]
        
        input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

        train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        save_object(
                file_path=self.data_transformation_config.preprocessor_obj_filepath,
                obj=preprocessing_obj

            )

        return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_filepath,
            )

       except Exception as e:
           raise CustomException(e,sys)
       