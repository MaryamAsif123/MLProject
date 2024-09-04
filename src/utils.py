import os
import sys
import numpy as np  # type: ignore
import pandas as pd # type: ignore
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(x_train, x_test, y_train, y_test, models):
    try:
        reports={}

        for i in range(len(list(models))):
            model= list(models.values())[i]
            model.fit(x_train,y_train)

            y_test_pred = model.predict(x_test)
            test_model_score = r2_score(y_test, y_test_pred)
            reports[list(models.keys())[i]] = test_model_score

        return reports
    
    except Exception as e:
        raise CustomException(e,sys)