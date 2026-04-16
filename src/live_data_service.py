import requests
import json
import os
import pandas as pd

class LiveDataService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.geo_url = "http://api.openweathermap.org/geo/1.0/direct"
        self.pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution"
        
        # Load medians for fallback
        medians_path = os.path.join("results", "feature_medians.json")
        if os.path.exists(medians_path):
            with open(medians_path, "r") as f:
                self.medians = json.load(f)
        else:
            self.medians = {}
            
        # Hardcoded Coordinate Cache for CPCB Cities (Performance & Reliability)
        self.city_coordinates = {
            "Ahmedabad": (23.0225, 72.5714), "Aizawl": (23.7307, 92.7173), "Amaravati": (16.5062, 80.6480),
            "Amritsar": (31.6340, 74.8723), "Bengaluru": (12.9716, 77.5946), "Bhopal": (23.2599, 77.4126),
            "Brajrajnagar": (21.8236, 83.9189), "Chandigarh": (30.7333, 76.7794), "Chennai": (13.0827, 80.2707),
            "Coimbatore": (11.0168, 76.9558), "Delhi": (28.6139, 77.2090), "Ernakulam": (9.9816, 76.2999),
            "Gurugram": (28.4595, 77.0266), "Guwahati": (26.1445, 91.7362), "Hyderabad": (17.3850, 78.4867),
            "Jaipur": (26.9124, 75.7873), "Jorapokhar": (23.7042, 86.4111), "Kochi": (9.9312, 76.2673),
            "Kolkata": (22.5726, 88.3639), "Lucknow": (26.8467, 80.9462), "Mumbai": (19.0760, 72.8777),
            "Patna": (25.5941, 85.1376), "Shillong": (25.5788, 91.8831), "Talcher": (20.9500, 85.2333),
            "Thiruvananthapuram": (8.5241, 76.9366), "Visakhapatnam": (17.6868, 83.2185)
        }

    def get_coordinates(self, city_name):
        """Get Lat/Lon for a city in India (uses cache first)."""
        if city_name in self.city_coordinates:
            return self.city_coordinates[city_name], None
        
        params = { "q": f"{city_name},IN", "limit": 1, "appid": self.api_key }
        try:
            response = requests.get(self.geo_url, params=params)
            if response.status_code == 401: return None, "API Key Unauthorized."
            data = response.json()
            if data:
                coords = (data[0]["lat"], data[0]["lon"])
                self.city_coordinates[city_name] = coords # Update cache
                return coords, None
            return None, f"City '{city_name}' not found."
        except Exception as e:
            return None, str(e)

    def fetch_live_data(self, city_name):
        """Fetch live air pollution data and map to features."""
        coords, error = self.get_coordinates(city_name)
        if error: return None, error
        
        lat, lon = coords
        params = { "lat": lat, "lon": lon, "appid": self.api_key }
        try:
            response = requests.get(self.pollution_url, params=params)
            if response.status_code == 401: return None, "Unauthorized API Key."
            response.raise_for_status()
            data = response.json()
            
            if "list" in data and len(data["list"]) > 0:
                comp = data["list"][0]["components"]
                live_features = {
                    "PM10": comp.get("pm10", self.medians.get("PM10")),
                    "NO": comp.get("no", self.medians.get("NO")),
                    "NO2": comp.get("no2", self.medians.get("NO2")),
                    "NOx": comp.get("no", 0) + comp.get("no2", 0) or self.medians.get("NOx"), 
                    "NH3": comp.get("nh3", self.medians.get("NH3")),
                    "CO": comp.get("co", 0) / 1000.0 or self.medians.get("CO"), 
                    "SO2": comp.get("so2", self.medians.get("SO2")),
                    "O3": comp.get("o3", self.medians.get("O3")),
                    "Benzene": self.medians.get("Benzene"),
                    "Toluene": self.medians.get("Toluene"),
                    "Xylene": self.medians.get("Xylene"),
                    "City": city_name
                }
                return live_features, None
            return None, "No pollution data list found."
        except Exception as e:
            return None, f"Fetch Error: {str(e)}"

