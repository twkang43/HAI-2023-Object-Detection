import os
import torch
import json
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from utils import predict_with_input

HOME = os.getcwd()
INPUT_PATH = os.path.join(os.getcwd(), "input_image")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Client로부터 이미지를 받은 후 해당 함수 실행
# Arguments : Object Detection을 수행할 이미지의 이름
# Outputs : bounding box가 그려진 이미지, 식재료 리스트
#
# -- Output Naming Convention --
# bbox가 그려진 이미지 : bbox_(image_name).jpg
# 식재료 리스트 : ingredients_(image_name).json
# -- 이때, image_name은 기존 이미지의 확장자(jpg, png 등)를 제외한 부분입니다 --
def detect_ingredients(image_name):
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(DEVICE)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    # id2label 읽기
    os.chdir("utils")
    with open("id2label.json", 'r') as json_file:
        id2label = json.load(json_file)
        id2label = {int(key): value for key, value in id2label.items()}
    os.chdir(HOME)

    predict = predict_with_input.InputPrediction(
        model=model, 
        processor=processor, 
        id2label=id2label,
        device=DEVICE
    )

    # input image 가져오기
    try:
        input_image = Image.open(os.path.join(INPUT_PATH, image_name))
    except Exception as e:
        print(f"ERROR! : {e}")
        return

    # input image에 대한 prediction
    prediction = predict.predict(input_image, threshold=0.2)

    # prediction 시각화 & 식재료 set 저장
    predict.draw_bbox(input_image, image_name, prediction)
    predict.save_ingredients_set(prediction, image_name)