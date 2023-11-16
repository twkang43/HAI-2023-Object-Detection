import os
import torchvision
from transformers import DetrImageProcessor
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io as io
import numpy as np

ANNOTATION_FILE_NAME = "_annotations.coco.json"

TRAIN_DATASET_PATH = os.path.join("dataset", "train")
VALID_DATASET_PATH = os.path.join("dataset", "valid")
TEST_DATASET_PATH = os.path.join("dataset", "test")

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_dir:str, processor, train=True):
        annotation_file = self.get_annotation_file_path(image_dir)
        super(CocoDetection, self).__init__(image_dir, annotation_file)
        self.processor = processor

        coco = COCO(annotation_file)
        self.show_random_img(coco, image_dir)

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {"image_id" : image_id, "annotations" : annotations}

        encoding = self.processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target
    
    def get_annotation_file_path(self, image_dir):
        return os.path.join(image_dir, ANNOTATION_FILE_NAME)
    
    def show_random_img(self, coco, image_dir):
        # Select one image at random
        imgIds = coco.getImgIds()
        img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

        # load and display image
        I = io.imread('%s/%s'%(image_dir,img["file_name"]))
        plt.axis("off")
        plt.imshow(I)

        annIds = coco.getAnnIds(imgIds=img["id"])
        anns = coco.loadAnns(annIds)
        
        # Draw bounding box and Display catNms
        for ann in anns:
            bbox = ann["bbox"]
            random_color = np.random.rand(3, )

            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor=random_color, facecolor="none")
            plt.gca().add_patch(rect)

            cat_name = coco.loadCats(ann["category_id"])[0]["name"]
            plt.text(bbox[0], bbox[1]-5, cat_name, color="white", backgroundcolor=random_color, fontsize=8)

        plt.title(image_dir)
        plt.show()

class CocoDataset():
    def __init__(self, batch_size):
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.batch_size = batch_size

        self.train_dataset = CocoDetection(
            image_dir=TRAIN_DATASET_PATH,
            processor=self.processor,
            train=True
        )

        self.val_dataset = CocoDetection(
            image_dir=VALID_DATASET_PATH,
            processor=self.processor,
            train=False
        )

        self.test_dataset = CocoDetection(
            image_dir=TEST_DATASET_PATH,
            processor=self.processor,
            train=False
        )
    
    def get_dataset(self):
        return self.train_dataset, self.val_dataset, self.test_dataset