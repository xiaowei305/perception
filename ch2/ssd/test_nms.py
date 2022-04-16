from encoder import nms
import numpy as np
import random
import cv2
import torch


def draw_boxes(boxes):
    image = np.ones((200, 200), dtype=np.uint8) * 255
    for box in boxes:
        x1, y1, x2, y2 = (box * 200).astype(np.int32)
        cv2.rectangle(image, (x1, y1), (x2, y2), 0)

    cv2.imshow("image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    boxes = np.array([[0.2, 0.2, 0.5, 0.5], [0.7, 0.7, 0.9, 0.9]])
    new_boxes = []
    for i in range(10):
        box = random.choice(boxes)
        new_box = (np.random.random(4) - 0.5) * 0.1 + box
        new_boxes.append(new_box)

    draw_boxes(new_boxes)
    new_boxes = torch.tensor(new_boxes)
    scores = torch.tensor(np.random.random(len(new_boxes)))

    filtered_boxes, _ = nms(new_boxes, scores)
    filtered_boxes = filtered_boxes.numpy()
    draw_boxes(filtered_boxes)