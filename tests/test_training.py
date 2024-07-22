import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../training')))
from ml.data import process_data
from ml.model import train_model, compute_model_metrics

# Clean data
data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'census.csv'))
data.columns = data.columns.str.strip()
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Test for process_data
def test_process_data_types():
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    label = 'salary'
    
    X, y, encoder, lb = process_data(
        data, 
        categorical_features=categorical_features, 
        label=label, 
        training=True
    )
    
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert isinstance(encoder, OneHotEncoder), "encoder should be a OneHotEncoder"
    assert isinstance(lb, LabelBinarizer), "lb should be a LabelBinarizer"

# Test for train_model
def test_train_model():
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    label = 'salary'
    
    X, y, _, _ = process_data(
        data, 
        categorical_features=categorical_features, 
        label=label, 
        training=True
    )
    
    model = train_model(X, y)
    assert isinstance(model, LogisticRegression), "The model should be a LogisticRegression instance"

# Test for compute_model_metrics
def test_compute_model_metrics():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_preds = np.array([0, 1, 0, 0, 1, 1])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
    
    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float"
    assert isinstance(fbeta, float), "Fbeta should be a float"
    
    assert not np.isnan(precision), "Precision should not be NaN"
    assert not np.isnan(recall), "Recall should not be NaN"
    assert not np.isnan(fbeta), "Fbeta should not be NaN"