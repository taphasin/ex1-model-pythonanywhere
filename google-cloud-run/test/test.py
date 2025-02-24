import requests

resp = requests.post("https://ex1-model-pythonanywhere-production.up.railway.app/", files={'file': open('three.png', 'rb')})

print(resp.json())
