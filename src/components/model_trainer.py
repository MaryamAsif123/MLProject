import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import evaluate_models, save_object
from src.exception import CustomException
from sklearn.metrics import r2_score

@dataclass
class modelTrainerConfig:
    model_trainer_filepath = os.path.join("artifacts", "model.pkl")

class model_trainer:
    def __init__(self):
        self.model_trainer_config = modelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            x_train, y_train, x_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            model_report:dict=evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            if best_model_score<0.60:
                raise CustomException("No best model found")\
                
            save_object(
                file_path=self.model_trainer_config.model_trainer_filepath,
                obj=best_model
            )

            return {
                "best_model_name": best_model_name,
                "best_model_score": best_model_score
            }

        except Exception as e:
            raise CustomException(e,sys)