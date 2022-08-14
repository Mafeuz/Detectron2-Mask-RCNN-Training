# ### Preparing the dataset
# 
# Prepare the dataset using Labelme annotation tool (for Instance segmentation) and LabelImg for object detection.
# 

# ### Installing Detectron2

# install dependencies: 
#!pip install pyyaml==5.1
#!pip install onnx==1.8.0

#!pip install pyyaml==5.1
#!pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

#!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

import torch
import torchvision
import cv2

print('Torch Ver:', torch.__version__)
print('Torchvision:', torchvision.__version__)

import os
import numpy as np
import glob
import time
import json
import random
import matplotlib.pyplot as plt

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer

#########################################################################################################################
# ### Register the data to Detectron2 config

def get_data_dicts(directory, img_size, classes):
    
    height, width = img_size
    
    dataset_dicts = []
    
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        
        filename = os.path.join(directory, img_anns["imagePath"])
        
        record["file_name"] = filename
        record["height"]    = height
        record["width"]     = width
      
        annos = img_anns["shapes"]
        objs = []
        
        for anno in annos:
            px = [a[0] for a in anno['points']] # x coord
            py = [a[1] for a in anno['points']] # y-coord
            poly = [(x, y) for x, y in zip(px, py)] # poly for segmentation
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            
            objs.append(obj)
            
        record["annotations"] = objs
        dataset_dicts.append(record)
        
    return dataset_dicts

#########################################################################################################################
def dataset_load(data_path, img_size, classes):
    
    for d in ["train", "test"]:

        DatasetCatalog.register(
            "category_" + d, 
            lambda d=d: get_data_dicts(data_path + d, img_size, classes)
        )

        MetadataCatalog.get("category_" + d).set(thing_classes=classes)

        microcontroller_metadata = MetadataCatalog.get("category_train")
    
    return microcontroller_metadata

#########################################################################################################################
# ### Training the Detectron2 Instance Segmentation Model

def det2_mask_build(data_path, img_size, classes, batch_size=2, lr=0.00025, workers=2, max_iter=1000, device='cuda:0'):

    microcontroller_metadata = dataset_load(data_path, img_size, classes)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN                = ("category_train",)
    cfg.DATASETS.TEST                 = ()
    cfg.DATALOADER.NUM_WORKERS        = workers
    cfg.MODEL.DEVICE                  = device
    cfg.MODEL.WEIGHTS                 = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH          = batch_size
    cfg.SOLVER.BASE_LR                = lr
    cfg.SOLVER.MAX_ITER               = max_iter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES   = len(classes) + 1
    cfg.INPUT.MIN_SIZE_TRAIN          = (800,)
    cfg.INPUT.MIN_SIZE_TEST           = 800
    cfg.INPUT.MAX_SIZE_TRAIN          = 1333
    cfg.INPUT.MAX_SIZE_TEST           = 1333
    
    return cfg, microcontroller_metadata

#########################################################################################################################
def det2_mask_train(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

#########################################################################################################################



