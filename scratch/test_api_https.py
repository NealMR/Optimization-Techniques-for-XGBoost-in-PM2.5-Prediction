import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OWM_API_KEY")

print(f"Testing API Key with HTTPS: {api_key}")

# Delhi coordinates
lat, lon = 28.6139, 77.2090
url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"

response = requests.get(url)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
