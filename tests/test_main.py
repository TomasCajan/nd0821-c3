"""
Note: Run this test session using pytest.
"""
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_greet():
    """
    Test the GET method to ensure the status code and response content are correct.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Inference API. Use the POST /predict endpoint to get predictions."}

def test_post_predict_positive():
    """
    Test the POST /predict method for a positive inference.
    """
    response = client.post("/predict", json={
        "data": [
            {
                "age": 15,
                "workclass": "Self-emp-not-inc",
                "fnlwgt": 17516,
                "education": "7th-8th",
                "education-num": 2,
                "marital-status": "Never-married",
                "occupation": "Other-service",
                "relationship": "Not-in-family",
                "race": "Black",
                "sex": "Female",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 10,
                "native-country": "United-States"
            }
        ]
    })
    assert response.status_code == 200
    assert response.json() == {"predictions": ["<=50K"]}

def test_post_predict_negative():
    """
    Test the POST /predict method for a negative inference.
    """
    response = client.post("/predict", json={
        "data": [
            {
                "age": 35,
                "workclass": "Federal-gov",
                "fnlwgt": 83311,
                "education": "Masters",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 15024,
                "capital-loss": 0,
                "hours-per-week": 60,
                "native-country": "United-States"
            }
        ]
    })
    assert response.status_code == 200
    assert response.json() == {"predictions": [">50K"]}

if __name__ == "__main__":
    test_get_greet()
    test_post_predict_positive()
    test_post_predict_negative()