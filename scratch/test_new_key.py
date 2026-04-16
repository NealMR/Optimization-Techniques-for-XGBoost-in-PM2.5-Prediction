import requests

api_key = "81f896ce794c3314db30af32d8458b19"
print(f"Testing New API Key: {api_key}")

# Delhi coordinates
lat, lon = 28.6139, 77.2090
url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"

response = requests.get(url)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
