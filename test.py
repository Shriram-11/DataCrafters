import requests

url = "http://127.0.0.1:8000/api/upload_csv/"
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzMwNjI0MDczLCJpYXQiOjE3MzA2MjIyNzMsImp0aSI6IjcwYjM5NDhkMTE5ZDRmNGY4ZDQyZTU1N2Y0MjE4OWE2IiwidXNlcl9pZCI6Mn0.3M9sDc05Zi72wO0a1I6azNn88KQX-vBcqDYprjbBLpI",
}
files = {
    "csv_file": open("D:/cvd_prediction_api/heart.csv", "rb"),
}

response = requests.post(url, headers=headers, files=files)
print(response.status_code, response.json())
