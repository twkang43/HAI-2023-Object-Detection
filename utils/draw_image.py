import os
import random
import torch
from PIL import Image, ImageDraw
import numpy as np
from transformers import DetrImageProcessor

TEST_DATASET_PATH = os.path.join("dataset", "test")

class DrawImage():
    def __init__(self, model, dataset):
        super(DrawImage, self).__init__()
        self.model = model
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.coco = dataset.coco

    def draw_image(self):
        image_ids = self.coco.getImgIds()
        image_id = random.choice(image_ids)
        print(f"Image # {image_id}")

        image_coco = self.coco.loadImgs(image_id)[0]
        image_gt = Image.open('%s/%s'%(TEST_DATASET_PATH, image_coco['file_name']))
        image_pred = Image.open('%s/%s'%(TEST_DATASET_PATH, image_coco['file_name']))

        if not os.path.exists("output_images"):
            os.mkdir("output_images")

        self.draw_ground_truth(image_gt, image_coco)
        self.draw_predict(image_pred)

    def draw_ground_truth(self, image, image_coco):
        draw = ImageDraw.Draw(image)

        annIds = self.coco.getAnnIds(imgIds=image_coco['id'])
        anns = self.coco.loadAnns(annIds)

        for ann in anns:
            bbox = ann["bbox"]
            random_color = tuple(np.random.randint(0, 256, 3))

            # bounding box 그리기
            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], outline=random_color, width=2)

            cat_name = self.coco.loadCats(ann["category_id"])[0]["name"]
            draw.text((bbox[0], bbox[1] - 5), cat_name, fill="white", font=None, anchor=None, spacing=4)

        # 이미지 저장
        image.save("output_images/ground_truth.png")

    def draw_predict(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)[0]

        draw = ImageDraw.Draw(image)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]

            print(
                f"Detected {self.model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
            
            # bounding box 좌표 얻기
            x, y, w, h = box
            x *= image.width
            y *= image.height
            w *= image.width
            h *= image.height

            # bounding box 그리기
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

            label_str = self.model.config.id2label[label.item()]
            confidence_str = f"{round(score.item(), 3)}"

            # bounding box 위에 label 및 confidence 표시
            draw.text((x, y), f"{label_str}: {confidence_str}", fill="red")

        # 이미지 저장
        image.save("output_images/predict.png")