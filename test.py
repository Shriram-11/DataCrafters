'''import requests

url = "http://127.0.0.1:8000/api/upload_csv/"
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzMwODE4NDIxLCJpYXQiOjE3MzA4MTY2MjEsImp0aSI6IjAwMDY4ZTdiYjRmNDQ4Zjc4ODY5YTA4MWFmYThkODEwIiwidXNlcl9pZCI6Mn0.W2kfL8oaIkVAgejvm5K5fHaVKpLVBijAmvplHsgbtD0",
}
files = {
    "csv_file": open("D:/cvd_prediction_api/heart.csv", "rb"),
}

response = requests.post(url, headers=headers, files=files)
l = (len(response.json()))
print(response.status_code, l)
'''
import os

# Get the value of the environment variable
gemini_key = os.getenv('API_KEY')

# Check if the variable exists and print it
if gemini_key:
    print(f"GEMINI_KEY: {gemini_key}")
else:
    print("GEMINI_KEY is not set")
