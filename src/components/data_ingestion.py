import os
import sys
import pandas as pd # type: ignore

from src.exception import CustomException
from sklearn.model_selection import train_test_split # type: ignore
from dataclasses import dataclass
from src.components.data_transformation import data_transformation
from src.components.model_trainer import model_trainer, modelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df=pd.read_csv('notebook\data.csv')
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    data_ingestion=DataIngestion()
    train_data,test_data=data_ingestion.initiate_data_ingestion()

    data_transformation = data_transformation()
    train_array, test_array,_ =data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer= model_trainer()
    print(model_trainer.initiate_model_trainer(train_array, test_array))


