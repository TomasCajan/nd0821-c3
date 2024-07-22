import unittest
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../training')))
from ml.data import process_data
from ml.model import train_model, compute_model_metrics

#Clean data
data = pd.read_csv("../data/census.csv")
data.columns = data.columns.str.strip()
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

class TestProcessData(unittest.TestCase):
    """
    Unit tests for the process_data function.
    """
    
    def setUp(self):
        """
        Set up the data and parameters for the tests.
        """
        self.data = data
        self.categorical_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        self.label = 'salary'

    def test_process_data_types(self):
        """
        Test if process_data returns the correct data types.
        """
        X, y, encoder, lb = process_data(
            self.data, 
            categorical_features=self.categorical_features, 
            label=self.label, 
            training=True
        )
        
        self.assertIsInstance(X, np.ndarray, "X should be a numpy array")
        self.assertIsInstance(y, np.ndarray, "y should be a numpy array")
        self.assertIsInstance(encoder, OneHotEncoder, "encoder should be a OneHotEncoder")
        self.assertIsInstance(lb, LabelBinarizer, "lb should be a LabelBinarizer")

class TestTrainModel(unittest.TestCase):
    """
    Unit tests for the train_model function.
    """
    
    def setUp(self):
        """
        Set up the data and parameters for the tests.
        """
        self.data = data
        self.categorical_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        self.label = 'salary'

    def test_train_model(self):
        """
        Test if train_model returns a LogisticRegression instance.
        """
        X, y, _, _ = process_data(
            self.data, 
            categorical_features=self.categorical_features, 
            label=self.label, 
            training=True
        )
        
        model = train_model(X, y)
        self.assertIsInstance(model, LogisticRegression, "The model should be a LogisticRegression instance")

class TestComputeModelMetrics(unittest.TestCase):
    """
    Unit tests for the compute_model_metrics function.
    """
    
    def setUp(self):
        """
        Set up the true and predicted labels for the tests.
        """
        self.y_true = np.array([0, 1, 1, 0, 1, 0])
        self.y_preds = np.array([0, 1, 0, 0, 1, 1])
    
    def test_compute_model_metrics(self):
        """
        Test if compute_model_metrics returns a tuple of three floats.
        """
        precision, recall, fbeta = compute_model_metrics(self.y_true, self.y_preds)
        
        self.assertIsInstance(precision, float, "Precision should be a float")
        self.assertIsInstance(recall, float, "Recall should be a float")
        self.assertIsInstance(fbeta, float, "Fbeta should be a float")
        
        self.assertFalse(np.isnan(precision), "Precision should not be NaN")
        self.assertFalse(np.isnan(recall), "Recall should not be NaN")
        self.assertFalse(np.isnan(fbeta), "Fbeta should not be NaN")

if __name__ == '__main__':
    unittest.main()