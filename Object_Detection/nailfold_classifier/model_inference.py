# coding:utf-8
import os
import time

import torch
import torchvision
from Object_Detection.nailfold_classifier.model_backbone import Backbone
from torch import nn
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from utils import transforms

classes = ["abnormal", "normal"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "./checkpoints/resnet18/model-resnet18-2-0.8260869565217391.pth"

val_transforms_list = [
    torchvision.transforms.Resize(size=(224, 224)),
    transforms.ZeroOneNormalize(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
val_transforms = torchvision.transforms.Compose(val_transforms_list)

backbone = Backbone(out_dimension=len(classes), model_name="resnet18")
model, _, _ = backbone.build_model()
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
model.to(device)
model.eval()

image_path = "./Object_Detection/data/mask_dataset/test/abnormal"
class SimpleFPS:
    """Minimal replacement for imutils.video.FPS."""

    def __init__(self):
        self._start = None
        self._end = None
        self._num_frames = 0

    def start(self):
        self._start = time.perf_counter()
        self._end = None
        self._num_frames = 0
        return self

    def stop(self):
        if self._start is not None and self._end is None:
            self._end = time.perf_counter()

    def update(self):
        if self._start is not None:
            self._num_frames += 1

    def elapsed(self):
        if self._start is None:
            return 0.0
        end_time = self._end if self._end is not None else time.perf_counter()
        return end_time - self._start

    def fps(self):
        elapsed_time = self.elapsed()
        if elapsed_time == 0:
            return 0.0
        return self._num_frames / elapsed_time


fps = SimpleFPS()
fps.start()

with torch.no_grad():
    for image_name in os.listdir(image_path):
        file_path = os.path.join(image_path, image_name)
        img = read_image(file_path, mode=ImageReadMode.RGB)
        img = img.to(device)
        img = img.unsqueeze(dim=0)
        img = val_transforms(img).to(device)

        res = model(img)
        cls_index = res.argmax(dim=1)
        cls_prob = nn.functional.softmax(res, dim=1)

        pred_prob = cls_prob[0][cls_index].item()
        pred_cls = classes[cls_index]

        print(pred_prob, pred_cls)

        fps.update()
fps.stop()
print("FPS: {}".format(fps.fps()))
print("time: {}".format(fps.elapsed()))
