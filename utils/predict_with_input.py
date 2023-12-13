import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

INPUT_PATH = os.path.join(os.getcwd(), "input_image")
OUTPUT_PATH = os.path.join(os.getcwd(), "outputs")

class InputPrediction():
    def __init__(self, model, processor, id2label, device):
        super(InputPrediction, self).__init__()
        self.model = model
        self.processor = processor
        self.id2label = id2label
        self.device = device
        self.font = ImageFont.truetype(os.path.join("fonts", "Pretendard-Bold.otf"), 15)

    def get_input(self):
        file_list = os.listdir(INPUT_PATH)
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
        image_files = [file for file in file_list if any(file.lower().endswith(ext) for ext in image_extensions)]
        image_name = image_files[-1]
            
        if 0 < len(image_files):
            # input_image 폴더 내의 마지막 이미지를 가져오기
            input_image = Image.open(os.path.join(INPUT_PATH, image_name))
            return input_image, image_name
        else:
            raise Exception("No input image found.")

    def predict(self, input_image, threshold):
        inputs = self.processor(images=input_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([input_image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

        return results

    def draw_bbox(self, input_image, image_name, prediction):
        draw = ImageDraw.Draw(input_image)

        for label, bbox in zip(prediction["labels"], prediction["boxes"]):
            label_str = self.id2label[label.item()]
            bbox = [round(i, 2) for i in bbox.tolist()]
            
            random_color = tuple(np.random.randint(0, 256, 3))

            # bounding box 그리기
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=random_color, width=2)

            # bounding box 위에 label 및 confidence 표시
            draw.text((bbox[0], bbox[1]-20), f"{label_str}", fill="white", font=self.font, anchor=None, spacing=4)

        image_name = "bbox_" + image_name
        image_path = os.path.join(OUTPUT_PATH, image_name)

        # 이미지가 이미 존재한다면 삭제
        if os.path.exists(image_path):
            os.remove(image_path)

        # 이미지 저장
        input_image.save(image_path)

    def save_ingredients_set(self, prediction, image_name):
        ingredients_str = []
        json_name = "ingredients_" + image_name[:-4] + ".json"

        for label in prediction["labels"]:
            ingredients_str.append(self.id2label[label.item()])

        # 재료의 집합을 저장
        ingredients_set = set(ingredients_str)
        json_path = os.path.join(OUTPUT_PATH, json_name)

        # JSON 파일이 이미 존재한다면 삭제
        if os.path.exists(json_path):
            os.remove(json_path)

        # JSON 파일 저장
        with open(json_path, 'w') as json_file:
            json.dump(list(ingredients_set), json_file, indent=4)