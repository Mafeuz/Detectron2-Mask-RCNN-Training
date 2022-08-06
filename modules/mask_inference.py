# install dependencies: 
#!pip install pyyaml==5.1
#!pip install onnx==1.8.0

#!pip install pyyaml==5.1
#!pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

#!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

import torch
import torchvision
import cv2
import random

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

##########################################################################################################################
'''
    Module for handling Detectron2 Mask-RCNN Inference

'''
##########################################################################################################################
def get_data_dicts(directory, classes):
    
    dataset_dicts = []
    
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        
        filename = os.path.join(directory, img_anns["imagePath"])
        
        record["file_name"] = filename
        record["height"] = 2000
        record["width"] = 2000
      
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

##########################################################################################################################
def dataset_load(classes):

    random_string = str(random.randint(0, 100000))
    
    for d in ['test']:

        DatasetCatalog.register(
            random_string + d, 
            lambda d=d: get_data_dicts(d, classes)
        )

        MetadataCatalog.get(random_string + d).set(thing_classes=classes)

        microcontroller_metadata = MetadataCatalog.get(random_string + 'test')
    
    return microcontroller_metadata

##########################################################################################################################
def load_maskrcnn(model_path, classes_list, conf_thresh):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes_list) + 1

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thresh 
    cfg.DATASETS.TEST = ("test", )
    predictor = DefaultPredictor(cfg)

    data_path = ''
    metadata = dataset_load(classes_list)
    
    return cfg, predictor, metadata

##########################################################################################################################
def det2_maskrcnn_inference(predictor, metadata, img_path, print_out=False, visualize=False, save_detection=False):
     
    img = cv2.imread(img_path)
    #img = cv2.resize(img, resize)

    start = time.time()
    outputs = predictor(img)
    end = time.time()

    results = outputs['instances'].to('cpu')
    classes = results.pred_classes.tolist()
    scores  = results.scores.tolist()
    masks   = results.pred_masks
    
    # Define list with each image outputs:
    img_results = []

    # Loop over detections:
    for j, score in enumerate(scores):
        
        if print_out:
            print('####################################### Output ###############################################')
            print('Processing Time:', (end-start)*1000)
            print(f'Detection {j} - Class: {classes[j]}, Score: {score}')
            print(f'Mask: {masks[j]}')

        # Append to list the results(convert mask tensor to array)
        img_results.append([j, classes[j], score, masks[j].detach().cpu().numpy()])
            
    if visualize or save_detection:
        v = Visualizer(img[:, :, ::-1],
                        metadata = metadata, 
                        scale=0.8, 
                        instance_mode=ColorMode.IMAGE_BW # removes the colors of unsegmented pixels
        )

        v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        
        img = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)

        if visualize:
            plt.figure(figsize = (14, 10))
            plt.imshow(img)
            plt.show()
        
        if save_detection:
            split    = img_path.split(os.sep)
            img_name = split[len(split)-1]
            cv2.imwrite('det_' + img_name, img)
            
    return img_results

##########################################################################################################################
