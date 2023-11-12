import os
import torchvision

ANNOTATION_FILE_NAME = "_annotations.coco.json"

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_dir:str, processor, train=True):
        super(CocoDetection, self).__init__(image_dir, self.get_annotaion_file_path(image_dir))
        self.processor = processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        dic_annotations = {"imange_id" : image_id, "annotations" : annotations}

        encoding = self.processor(images=images, annotations=dic_annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][10]

        return pixel_values, target
    
    def get_annotaion_file_path(self, image_dir):
        return os.path.join(image_dir, ANNOTATION_FILE_NAME)