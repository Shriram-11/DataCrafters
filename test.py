import requests
import json

url = "http://127.0.0.1:8000/api/upload_csv/"
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzMwOTAzMTg0LCJpYXQiOjE3MzA5MDEzODQsImp0aSI6IjAyNzczOTgwZTMxNzRlOWM5MGYwODdlYmU0ZTg2OTllIiwidXNlcl9pZCI6Mn0.jXfuo5YqaV3pdhP5VrmJN_TOWxowmeaIe36OZYz1tCM",
}
files = {
    "csv_file": open("D:/cvd_prediction_api/heart.csv", "rb"),
}

# Send the request
response = requests.post(url, headers=headers, files=files)

# Get JSON response and its length
response_json = response.json()
l = len(response_json)

# Print status code and the length of the JSON response
print(response.status_code, l)

# Save the JSON response to a text file
with open("response_output.txt", "w") as file:
    # Saving in a pretty-printed format
    json.dump(response_json, file, indent=4)

'''
import os

# Get the value of the environment variable
gemini_key = os.getenv('API_KEY')

# Check if the variable exists and print it
if gemini_key:
    print(f"GEMINI_KEY: {gemini_key}")
else:
    print("GEMINI_KEY is not set")
'''
