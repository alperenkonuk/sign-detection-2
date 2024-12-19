import argparse
import time
from pathlib import Path

import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized


def detect(device):
    source, weights, imgszi, iou_thres, conf_thres = "inference/images/pist3.jpg", "weights/best.pt", 640, 0.45, 0.25

    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgszi, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                # Print results
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2
                    print(f'Class: {int(cls)}, x center: {x_center}, y center: {y_center}')
                    print('-')
                    print(f'Class: {int(cls)}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        detect(device)
