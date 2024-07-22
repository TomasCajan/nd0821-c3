from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
import os
import pickle
import io

from ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = LogisticRegression()

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [500, 800]
    }

    f1_scorer = make_scorer(f1_score, average='weighted')
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=f1_scorer)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : 
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    return predictions

def save_model(model, folder_path):
    """
    Save the trained model to a specified folder as a .pkl file.

    Parameters:
    model (object): The trained model to save.
    folder_path (str): The path to the folder where the model should be saved.
    file_name (str): The name of the file (without extension).

    Returns:
    str: The path to the saved model file.
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path,'trained_model.pkl')
    
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    
    return file_path

def load_model(folder_path):
    """
    Load a trained model from a specified folder.

    Parameters:
    folder_path (str): The path to the folder where the model is saved.
    file_name (str): The name of the file (without extension).

    Returns:
    object: The loaded model.
    """
    file_path = os.path.join(folder_path,'trained_model.pkl')
    
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    
    return model

def save_transformers(encoder, lb, folder_path):
    """
    Save the trained encoder and label binarizer to specified folder as .pkl files.

    Parameters:
    encoder (OneHotEncoder): The trained OneHotEncoder to save.
    lb (LabelBinarizer): The trained LabelBinarizer to save.
    folder_path (str): The path to the folder where the transformers should be saved.
    """
    os.makedirs(folder_path, exist_ok=True)
    
    encoder_path = os.path.join(folder_path,'fitted_encoder.pkl')
    lb_path = os.path.join(folder_path, 'fitted_binarizer.pkl')
    
    with open(encoder_path, 'wb') as file:
        pickle.dump(encoder, file)
    
    with open(lb_path, 'wb') as file:
        pickle.dump(lb, file)
    
    return encoder_path, lb_path

def load_transformers(folder_path):
    """
    Load the trained encoder and label binarizer from specified folder.

    Parameters:
    folder_path (str): The path to the folder where the transformers are saved.
    encoder_file_name (str): The name of the encoder file (without extension).
    lb_file_name (str): The name of the label binarizer file (without extension).

    Returns:
    tuple: The loaded OneHotEncoder and LabelBinarizer objects.
    """
    encoder_path = os.path.join(folder_path, 'fitted_encoder.pkl')
    lb_path = os.path.join(folder_path, 'fitted_binarizer.pkl')
    
    with open(encoder_path, 'rb') as file:
        encoder = pickle.load(file)
    
    with open(lb_path, 'rb') as file:
        lb = pickle.load(file)
    
    return encoder, lb

def evaluate_model_slices(model, original_data, categorical_features, label, column_name, encoder, lb, output_path):
    """
    Evaluate the model performance on slices of data defined by unique values in a specified column.

    Inputs
    ------
    model : object
        Trained machine learning model.
    original_data : pd.DataFrame
        Original data containing the specified column.
    categorical_features: list[str]
        List containing the names of the categorical features.
    label : str
        Name of the label column in `original_data`.
    column_name : str
        Name of the column to slice data by.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer.
    output_path : str
        Path to save the output text file.
    """
    unique_values = original_data[column_name].unique()
    output = []

    for value in unique_values:
        slice_data = original_data[original_data[column_name] == value]
        
        X_slice, y_slice, _, _ = process_data(
            slice_data, 
            categorical_features=categorical_features, 
            label=label, 
            training=False, 
            encoder=encoder, 
            lb=lb
        )
        
        preds_slice = model.predict(X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)
        
        metrics_output = (
            f"Metrics for {column_name} = {value}:\n"
            f"  Precision: {precision:.4f}\n"
            f"  Recall: {recall:.4f}\n"
            f"  F1 Score: {fbeta:.4f}\n"
        )
        print(metrics_output)
        output.append(metrics_output)

    with open(output_path, 'w') as file:
        file.writelines(output)

def create_model_card(model_path, train_data, test_data, metrics, output_path="model_card.md"):
    """
    Create a Markdown file for the model card with specified details.

    Parameters:
    -----------
    model_path : str
        Path to the trained model file (pickle file).
    train_data_description : str
        Description of the training data.
    test_data_description : str
        Description of the evaluation data.
    metrics : tuple
        Tuple containing metrics.
    output_path : str, optional
        Path to save the Markdown file, default is 'model_card.md'.
    """
    
    # Load the trained model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Extract metrics
    precision, recall, fbeta = metrics

    content = f"""
    # Model Card

    ---

    ## Model Details
    This is a Logistic Regression model trained on the census dataset.
    It was trained using a k-fold cross-validated grid search and comes with the following setup:  {model}

    ---

    ## Intended Use
    Created for classification of salary categories within the census dataset.

    ---

    ## Training Data
    Model was trained on the census+income dataset publicly available here:
    https://archive.ics.uci.edu/dataset/20/census+income
    Specifically on 80% of the dataset, having exact shape of {train_data.shape}
    

    ---

    ## Evaluation Data
    Model was evaluated on the remaining 20% of the dataset mentioned above, specifically : {test_data.shape}
    
    ---

    ## Metrics
    Model scores following metrics of:
    - Precision: {precision:.4f}
    - Recall: {recall:.4f}
    - F1 Score: {fbeta:.4f}

    ---

    ## Ethical Considerations
    Model unfortunately has tendency to positively discriminate white and Asian people.

    ---

    ## Caveats and Recommendations
    Use at your own risk.
        """

    with open(output_path, 'w') as file:
        file.write(content.strip())
