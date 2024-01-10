import json
import pandas as pd
import boto3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA





s3_client = boto3.client('s3')

def lambda_handler(event, context):

    s3_URI = event['s3_URI']
    run_type = event['run_type']
    ml_type = event['ml_type']
    model_type = event['model_type']
    predict_data = event['predict_data']

    # Validate the s3_URI
    if not validate_s3_uri(s3_URI):
        raise ValueError('Invalid s3_URI')
    
    # Parse the s3_URI
    bucket_name,file_name = s3_uri_parser(s3_URI)

    # Read the file from s3
    df = read_file_from_s3(bucket_name,file_name)

    # Preprocess the data
    df = preprocessor(df)

    # Create the model or predict using the model
    if run_type == 'train':
        model = create_model(df, ml_type, model_type)
    elif run_type == 'predict':
        df = pd.DataFrame(predict_data)
        model = read_file_from_s3(bucket_name, file_name)
        df = predict(model, df)

        return {
            'statusCode': 200,
            'body': json.dumps(df.to_dict(orient='records'))
        }
    else:
        raise ValueError('Invalid run_type')
    

    # Save the model
    save_model(model, bucket_name, file_name)

    return {
        'statusCode': 200,
        'body': json.dumps('Model saved successfully')
    }

    


def s3_uri_parser(s3_URI: str) -> tuple:
    """
        Parse the s3_URI
        Args:
            s3_URI (str): s3_URI
        Returns:
            tuple: bucket_name, folder_name, file_name
        Example:
            s3_uri_parser('s3://bucket-name/folder-name/to/file-name.csv')
            ('bucket-name', 'folder-name/to', 'file-name.csv')

            s3_uri_parser('s3://bucket-name/folder-name/file-name.csv')
            ('bucket-name', 'folder-name', 'file-name.csv')

            s3_uri_parser('s3://bucket-name/file-name.csv')
            ('bucket-name', '', 'file-name.csv')

            s3_uri_parser('s3://bucket-name/folder-name/')
            ('bucket-name', 'folder-name', '')
            
    """
    s3_URI = s3_URI.replace('s3://', '')
    bucket_name = s3_URI.split('/')[0]
    file_name = s3_URI.split('/')[-1]
    
    return bucket_name, file_name


def validate_s3_uri(s3_URI: str)-> bool:
    """
        Validate the s3_URI
        Args:
            s3_URI (str): s3_URI
        Returns:
            bool: True if the s3_URI is valid, False otherwise
        Example:
            validate_s3_uri('s3://bucket-name/folder-name/file-name.csv')
            True

            validate_s3_uri('s3://bucket-name/folder-name/file-name.csv')
            False
    """
    if s3_URI.startswith('s3://'):
        return True
    else:
        return False
    

def read_file_from_s3(bucket_name: str, file_name: str) -> pd.DataFrame:
    """
        Read the file from s3
        Args:
            bucket_name (str): bucket_name
            folder_name (str): folder_name
            file_name (str): file_name
        Returns:
            pd.DataFrame: DataFrame
        Example:
            read_file_from_s3('bucket-name', 'folder-name', 'file-name.csv')
            DataFrame
    """
    obj = s3_client.get_object(Bucket=bucket_name, Key=file_name)
    df = pd.read_csv(obj['Body'])
    return df


def preprocessor(df: pd.DataFrame) -> pd.DataFrame:
    """
        Preprocess the data
        Args:
            df (pd.DataFrame): DataFrame
        Returns:
            pd.DataFrame: DataFrame
        Example:
            preprocessor(df)
            DataFrame
    """

    # Drop the duplicates
    df = df.drop_duplicates()

    # Drop the rows with null values
    df = df.dropna()

    # Drop the rows with negative values
    df = df[df['value'] >= 0]

    return df


def create_model(df: pd.DataFrame, ml_type: str, model_type: str) -> tuple:
    """
        Create the model
        Args:
            df (pd.DataFrame): DataFrame
            ml_type (str): ml_type
            model_type (str): model_type
        Returns:
            tuple: model, X_train, X_test, y_train, y_test
        Example:
            create_model(df, 'supervised', 'linear_regression')
            (model, X_train, X_test, y_train, y_test)
    """
    if ml_type == 'supervised':
        if model_type == 'linear_regression':
            model, X_train, X_test, y_train, y_test = linear_regression(df)
            return model, X_train, X_test, y_train, y_test
        elif model_type == 'logistic_regression':
            model, X_train, X_test, y_train, y_test = logistic_regression(df)
            return model, X_train, X_test, y_train, y_test
        elif model_type == 'decision_tree':
            model, X_train, X_test, y_train, y_test = decision_tree(df)
            return model, X_train, X_test, y_train, y_test
        elif model_type == 'random_forest':
            model, X_train, X_test, y_train, y_test = random_forest(df)
            return model, X_train, X_test, y_train, y_test
        elif model_type == 'xgboost':
            model, X_train, X_test, y_train, y_test = xgboost(df)
            return model, X_train, X_test, y_train, y_test
        elif model_type == 'lightgbm':
            model, X_train, X_test, y_train, y_test = lightgbm(df)
            return model, X_train, X_test, y_train, y_test
        elif model_type == 'catboost':
            model, X_train, X_test, y_train, y_test = catboost(df)
            return model, X_train, X_test, y_train, y_test
        else:
            raise ValueError('Invalid model_type')
    elif ml_type == 'unsupervised':
        if model_type == 'k_means':
            model, X_train, X_test, y_train, y_test = k_means(df)
            return model, X_train, X_test, y_train, y_test
        elif model_type == 'pca':
            model, X_train, X_test, y_train, y_test = pca(df)
            return model, X_train, X_test, y_train, y_test
        else:
            raise ValueError('Invalid model_type')
    else:
        raise ValueError('Invalid ml_type')
    




def linear_regression(df: pd.DataFrame) -> tuple:
    """
        Linear Regression
        Args:
            df (pd.DataFrame): DataFrame
        Returns:
            tuple: model, X_train, X_test, y_train, y_test
        Example:
            linear_regression(df)
            (model, X_train, X_test, y_train, y_test)
    """
    # Split the data into X and y
    X = df[['value']]
    y = df['target']

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the model
    model = LinearRegression()

    # Fit the model
    model.fit(X_train, y_train)

    return model


def logistic_regression(df: pd.DataFrame) -> tuple:
    """
        Logistic Regression
        Args:
            df (pd.DataFrame): DataFrame
        Returns:
            tuple: model, X_train, X_test, y_train, y_test
        Example:
            logistic_regression(df)
            (model, X_train, X_test, y_train, y_test)
    """
    # Split the data into X and y
    X = df[['value']]
    y = df['target']

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the model
    model = LogisticRegression()

    # Fit the model
    model.fit(X_train, y_train)

    return model


def decision_tree(df: pd.DataFrame) -> tuple:
    """
        Decision Tree
        Args:
            df (pd.DataFrame): DataFrame
        Returns:
            tuple: model, X_train, X_test, y_train, y_test
        Example:
            decision_tree(df)
            (model, X_train, X_test, y_train, y_test)
    """
    # Split the data into X and y
    X = df[['value']]
    y = df['target']

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the model
    model = DecisionTreeRegressor()

    # Fit the model
    model.fit(X_train, y_train)

    return model


def random_forest(df: pd.DataFrame) -> tuple:
    """
        Random Forest
        Args:
            df (pd.DataFrame): DataFrame
        Returns:
            tuple: model, X_train, X_test, y_train, y_test
        Example:
            random_forest(df)
            (model, X_train, X_test, y_train, y_test)
    """
    # Split the data into X and y
    X = df[['value']]
    y = df['target']

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the model
    model = RandomForestRegressor()

    # Fit the model
    model.fit(X_train, y_train)

    return model



def xgboost(df: pd.DataFrame) -> tuple:
    """
        XGBoost
        Args:
            df (pd.DataFrame): DataFrame
        Returns:
            tuple: model, X_train, X_test, y_train, y_test
        Example:
            xgboost(df)
            (model, X_train, X_test, y_train, y_test)
    """
    # Split the data into X and y
    X = df[['value']]
    y = df['target']

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the model
    model = GradientBoostingRegressor()

    # Fit the model
    model.fit(X_train, y_train)

    return model


def lightgbm(df: pd.DataFrame) -> tuple:
    """
        LightGBM
        Args:
            df (pd.DataFrame): DataFrame
        Returns:
            tuple: model, X_train, X_test, y_train, y_test
        Example:
            lightgbm(df)
            (model, X_train, X_test, y_train, y_test)
    """
    # Split the data into X and y
    X = df[['value']]
    y = df['target']

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the model
    model = GradientBoostingRegressor()

    # Fit the model
    model.fit(X_train, y_train)

    return model


def catboost(df: pd.DataFrame) -> tuple:
    """
        CatBoost
        Args:
            df (pd.DataFrame): DataFrame
        Returns:
            tuple: model, X_train, X_test, y_train, y_test
        Example:
            catboost(df)
            (model, X_train, X_test, y_train, y_test)
    """
    # Split the data into X and y
    X = df[['value']]
    y = df['target']

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the model
    model = GradientBoostingRegressor()

    # Fit the model
    model.fit(X_train, y_train)

    return model


def k_means(df: pd.DataFrame) -> tuple:
    """
        K-Means Clustering
        Args:
            df (pd.DataFrame): DataFrame
        Returns:
            tuple: model, X_train, X_test, y_train, y_test
        Example:
            k_means(df)
            (model, X_train, X_test, y_train, y_test)
    """
    # Split the data into X and y
    X = df[['value']]

    # Create the model
    model = KMeans(n_clusters=2)

    # Fit the model
    model.fit(X)

    return model


def pca(df: pd.DataFrame) -> tuple:
    """
        PCA
        Args:
            df (pd.DataFrame): DataFrame
        Returns:
            tuple: model, X_train, X_test, y_train, y_test
        Example:
            pca(df)
            (model, X_train, X_test, y_train, y_test)
    """
    # Split the data into X and y
    X = df[['value']]

    # Create the model
    model = PCA(n_components=2)

    # Fit the model
    model.fit(X)

    return model


def save_model(model: object, bucket_name: str, file_name: str) -> None:
    """
        Save the model
        Args:
            model (object): model
            bucket_name (str): bucket_name
            folder_name (str): folder_name
            file_name (str): file_name
        Returns:
            None
        Example:
            save_model(model, 'bucket-name', 'folder-name', 'file-name.csv')
            None
    """
   
    s3_client.put_object(Body=model, Bucket=bucket_name, Key=file_name)


def predict(model: object, df: pd.DataFrame) -> pd.DataFrame:
    """
        Predict using the model
        Args:
            model (object): model
            df (pd.DataFrame): DataFrame
        Returns:
            pd.DataFrame: DataFrame
        Example:
            predict(model, df)
            DataFrame
    """
    # Predict using the model
    y_pred = model.predict(df)

    # Add the predictions to the DataFrame
    df['predictions'] = y_pred

    return df




if __name__ == '__main__':
    # Create the model
    event = {
        's3_URI': 's3://bucket-name/path/to/data.csv',
        'run_type': 'train',
        'ml_type': 'supervised',
        'model_type': 'linear_regression',
        'predict_data': [
            {
                'value': 1
            },
            {
                'value': 2
            },
            {
                'value': 3
            }
        ]
    }

    lambda_handler(event, None)
