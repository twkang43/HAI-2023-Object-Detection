import os
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

DATASET_PATH = os.path.join("dataset", "test")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DrawImage():
    def __init__(self, model, processor, dataset):
        super(DrawImage, self).__init__()
        self.model = model
        self.processor = processor
        self.coco = dataset.coco
        self.font = ImageFont.truetype(os.path.join("fonts", "Pretendard-Bold.otf"), 15)

    def draw_image(self):
        image_ids = self.coco.getImgIds()
        image_id = image_ids[np.random.randint(0, len(image_ids))]
        print(f"Image #{image_id}")

        image_coco = self.coco.loadImgs(image_id)[0]
        print(image_coco["file_name"])

        image_gt = Image.open(os.path.join(DATASET_PATH, image_coco["file_name"]))
        image_pred = Image.open(os.path.join(DATASET_PATH, image_coco["file_name"]))

        if not os.path.exists("output_images"):
            os.mkdir("output_images")

        self.draw_ground_truth(image_gt, image_id)
        self.draw_predict(image_pred)

    def draw_ground_truth(self, image, image_id):
        draw = ImageDraw.Draw(image)

        anns = self.coco.imgToAnns[image_id]
        cats = self.coco.cats
        id2label = {k: v["name"] for k,v in cats.items()}

        for ann in anns:
            bbox = ann["bbox"]
            class_idx = ann["category_id"]
            random_color = tuple(np.random.randint(0, 256, 3))

            # bounding box 그리기
            draw.rectangle([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], outline=random_color, width=2)
            draw.text((bbox[0], bbox[1]-20), id2label[class_idx], fill="white", font=self.font, spacing=4)

        image_path = os.path.join("output_images", "ground_truth.png")
        
        # 이미지가 이미 존재한다면 삭제
        if os.path.exists(image_path):
            os.remove(image_path)

        # 이미지 저장
        image.save(image_path)

    def draw_predict(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        id2label = {k: v["name"] for k,v in self.coco.cats.items()}
        draw = ImageDraw.Draw(image)

        for score, label, bbox in zip(results["scores"], results["labels"], results["boxes"]):
            label_str = id2label[label.item()]
            confidence_str = f"{round(score.item(), 3)}"
            bbox = [round(i, 2) for i in bbox.tolist()]

            print(
                f"Detected {label_str} with confidence "
                f"{round(score.item(), 3)} at location {bbox}"
            )
            
            random_color = tuple(np.random.randint(0, 256, 3))

            # bounding box 그리기
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=random_color, width=2)

            # bounding box 위에 label 및 confidence 표시
            draw.text((bbox[0], bbox[1]-20), f"{label_str}: {confidence_str}", fill="white", font=self.font, anchor=None, spacing=4)

        image_path = os.path.join("output_images", "predict.png")
        
        # 이미지가 이미 존재한다면 삭제
        if os.path.exists(image_path):
            os.remove(image_path)

        # 이미지 저장
        image.save(image_path)