import torch

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(predictions) == 0:
            continue

        bboxes = prediction["boxes"]
        x_min, y_min, x_max, y_max = bboxes.unbind(1)
        bboxes = torch.stack((x_min, y_min, x_max-x_min, y_max-y_min), dim=1).tolist()

        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": bbox,
                    "score": scores[k]
                }
                for k, bbox in enumerate(bboxes)
            ]
        )

        return coco_results