import requests
import json

API_URL = "http://127.0.0.1:5000/predict"


def test_status_code():
    payload = {
        "features": [300, 298, 1500, 40, 5, 1, 0, 0, 0, 0, 0, 0]

    }
    response = requests.post(API_URL, json=payload)
    print("Status Code:", response.status_code)
    print("Response:", response.text)
    assert response.status_code == 200
    print("Test 1 Passed: Status code 200")



def test_response_format():
    payload = {
        "features": [300, 298, 1500, 40, 5, 1, 0, 0, 0, 0, 0, 0]

    }
    response = requests.post(API_URL, json=payload)
    data = response.json()
    assert isinstance(data, dict)
    print("Test 2 Passed: Response is JSON object")


def test_prediction_field():
    payload = {
        "features": [300, 298, 1500, 40, 5, 1, 0, 0, 0, 0, 0, 0]

    }
    response = requests.post(API_URL, json=payload)
    data = response.json()
    assert "prediction" in data
    print("Test 3 Passed: 'prediction' field exists")


if __name__ == "__main__":
    test_status_code()
    test_response_format()
    test_prediction_field()
    print("All smoke tests passed successfully.")
