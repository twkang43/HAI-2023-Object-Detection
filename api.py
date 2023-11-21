#import importlib

#packages = ['fastapi', 'uvicorn']

#for package in packages:
#   try:
#      importlib.import_module(package)
#    except ImportError:
#        importlib.install(package)
#fastAPI와 uvicorn가 깔려있다면 코드 삭제해도 무방

import json
import io
import os
import uuid
import base64   
from fastapi import FastAPI, UploadFile



app = FastAPI()

UPLOAD_DIR = "./input_image"
filename = "new_image.jpg"

@app.post("/input_image")
async def save_image(file: UploadFile):
    
    content = await file.read()
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        fp.write(content)

    return {"filename":filename}




@app.post("/send_data")
async def send_image():

    outputs_folder = "outputs"
    files_in_folder = os.listdir(outputs_folder)

    image_files = [file for file in files_in_folder if file.endswith('.jpg')]
    text_files = [file for file in files_in_folder if file.endswith('.json')]
    
    
    image_path = os.path.join(outputs_folder, image_files[0]) if image_files else None
    list_path = os.path.join(outputs_folder, text_files[0]) if text_files else None

    if image_path:
        with open(image_path, "rb") as img_file:
            img_ = img_file.read()
            img_base64 = base64.b64encode(img_).decode('utf-8')

    if list_path:
        with open(list_path, "r") as list_file:
            ingredient_list = list_file.readlines()

    response_data = {
        "image": img_base64 if image_path else None,
        "ingredients": ingredient_list if list_path else None
    }

    return response_data