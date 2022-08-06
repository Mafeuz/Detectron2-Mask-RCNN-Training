# install dependencies: 
#!pip install pyyaml==5.1
#!pip install onnx==1.8.0

#!pip install pyyaml==5.1
#!pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

#!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

import cv2
import os

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
#from detectron2.utils.visualizer import ColorMode, Visualizer

##########################################################################################################################
'''
    Module for handling Detectron2 Mask-RCNN Inference

'''
##########################################################################################################################
def load_maskrcnn(model_path, classes_list, conf_thresh):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes_list) + 1

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thresh
    predictor = DefaultPredictor(cfg)

    return predictor

##########################################################################################################################
def det2_maskrcnn_inference(predictor, img_path):
     
    img = cv2.imread(img_path)

    outputs = predictor(img)

    results = outputs['instances'].to('cpu')
    classes = results.pred_classes.tolist()
    scores  = results.scores.tolist()
    masks   = results.pred_masks
    
    # Define list with each image outputs:
    img_results = []

    # Loop over detections:
    for j, score in enumerate(scores):
        
        # Append to list the results(convert mask tensor to array)
        img_results.append([j, classes[j], score, masks[j].detach().cpu().numpy()])
            
    return img_results

##########################################################################################################################
