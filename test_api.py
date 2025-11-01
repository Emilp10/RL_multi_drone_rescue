"""Quick test script to verify API endpoints are responding correctly."""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(method, endpoint, data=None, params=None):
    """Test an API endpoint and print the response."""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, params=params, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        
        print(f"\n{'='*60}")
        print(f"{method} {endpoint}")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)[:500]}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"\n{'='*60}")
        print(f"{method} {endpoint}")
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    print("Testing Multi-Drone Rescue API")
    print("="*60)
    
    # Test root endpoint
    test_endpoint("GET", "/")
    
    # Test info endpoint
    test_endpoint("GET", "/info")
    
    # Test metrics endpoint
    test_endpoint("GET", "/metrics", params={"agents": 3})
    
    # Test reset endpoint
    test_endpoint("POST", "/reset", data={"agents": 3, "greedy": True, "seed": 123})
    
    # Test step endpoint
    test_endpoint("POST", "/step", data={"greedy": True})
    
    print("\n" + "="*60)
    print("API test complete!")
