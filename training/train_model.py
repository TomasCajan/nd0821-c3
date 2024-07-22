import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import *

def main():
    """
    Function for training of model and encoders for machine learning pipeline.

    Steps:
    1. Load and preprocess the dataset.
    2. Split the data into training and test sets.
    3. Process the training data to prepare it for model training.
    4. Train a logistic regression model using the processed training data.
    5. Save the trained model and the preprocessing transformers.
    6. Load the saved model and transformers for inference.
    7. Evaluate the model on slices of the training data and save the evaluation results.
    8. Process the test data using the loaded transformers.
    9. Make predictions on the processed test data using the trained model.
    10. Compute performance metrics for the model's predictions.
    11. Create and save a model card documenting the model details, data, metrics, and other relevant information.

    """

    data = pd.read_csv("../data/census.csv")
    data.columns = data.columns.str.strip()
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    trained_model = train_model(X_train, y_train)

    save_model(trained_model, "../model")
    save_transformers(encoder, lb, "../model")

    prediction_model = load_model("../model")
    inference_encoder, inference_binarizer = load_transformers("../model")

    evaluate_model_slices(prediction_model, train, cat_features, "salary", 'race', inference_encoder, inference_binarizer, "../screenshots/slice_output.txt")

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=inference_encoder, lb=inference_binarizer
    )

    model_predicitons = inference(prediction_model, X_test)
    model_metrics = compute_model_metrics(y_test, model_predicitons)

    create_model_card("../model/trained_model.pkl", train, test, model_metrics, output_path="../model_card.md")

if __name__ == "__main__":
    main()
