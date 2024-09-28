import requests
BASE = 'http://127.0.0.1:5000/'
img_file_path = "image.jpg"
response = requests.post(
    url=BASE+'upload-image',
    files={'image': open(img_file_path,'rb')}
)
print(response.json())
