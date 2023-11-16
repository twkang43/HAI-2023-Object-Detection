from PIL import Image, ImageDraw
import os

def visualize_results(img_path, predicted_bboxes, predicted_labels, target_bboxes, target_labels):
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for box, label in zip(predicted_bboxes, predicted_labels):
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"Pred : {label}", fill="red")

    for box, label in zip(target_bboxes, target_labels):
        draw.rectangle(box, outline="green", width=2)
        draw.text((box[0], box[1]), f"Ground Truth : {label}", fill="green")

    if not os.path.exists("output_img"):
        os.mkdir("output_img")

    image.save("output_img")