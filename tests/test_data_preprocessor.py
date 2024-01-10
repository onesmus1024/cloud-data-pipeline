import pytest
import pandas as pd

from src.cloud_data_pipeline.data_preprocessor import s3_uri_parser
from src.cloud_data_pipeline.data_preprocessor import validate_s3_uri
from src.cloud_data_pipeline.data_preprocessor import create_model
from src.cloud_data_pipeline.data_preprocessor import catboost
from src.cloud_data_pipeline.data_preprocessor import xgboost
from src.cloud_data_pipeline.data_preprocessor import lightgbm



@pytest.fixture
def s3_uri():
    return "s3://bucket-name/path/to/data.csv"





def test_s3_uri_parser(s3_uri):
    bucket_name, file_name = s3_uri_parser(s3_uri)
    assert bucket_name == "bucket-name"
    assert file_name == "data.csv"


def test_validate_s3_uri(s3_uri):
    assert validate_s3_uri(s3_uri) == True


    

    





