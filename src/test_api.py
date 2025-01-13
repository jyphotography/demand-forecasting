import requests
import json

def test_prediction():
    # Test data
    test_item = {
        'item_id': 'c578da8e8841',
        'store_id': '1',
        'date': '27.09.2024'
    }
    
    # Make request to the API
    url = 'http://localhost:9696/predict'
    response = requests.post(url, json=test_item)
    
    # Print results
    print('Status code:', response.status_code)
    print('Response:')
    print(json.dumps(response.json(), indent=2))

if __name__ == '__main__':
    test_prediction() 