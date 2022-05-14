import os
import json
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import ToTensor, Normalize, Compose


class TuSimple(data.Dataset):
    def __init__(self, root, input_size=(280, 640)):
        self.root = root
        ids = []

        for json_file in os.listdir(root):
            if not json_file.endswith('.json'):
                continue
            with open(os.path.join(root, json_file)) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    info = json.loads(line)
                    ids.append(info)
        self.ids = ids
        self.max_lane_num = max([x['lanes'] for x in self.ids])

        transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
        self.transform = transforms
        self.input_size = input_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, lanes).
        """
        img, lanes = self.get_image_label(index)
        w, h = img.size

        crop_y_start = 160
        img = img.crop((0, crop_y_start, w, h))
        lanes = [x - [0, crop_y_start] for x in lanes]

        w, h = img.size
        lanes = [x / [w, h] for x in lanes]

        input_h, input_w = self.input_size
        img = img.resize((input_w, input_h))
        img = self.transform(img)
        return img, {"lane": lanes}

    def __len__(self):
        return len(self.ids)

    def get_image_label(self, index):
        info = self.ids[index]
        img = Image.open(os.path.join(self.root, info['raw_file']))

        lanes = []
        for lane in info['lanes']:
            lanes.append(np.array(list(zip(lane, info['h_samples']))))
        return img, lanes
