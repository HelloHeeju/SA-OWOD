import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
from fvcore.common.file_io import PathManager
import itertools
import logging
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

__all__ = ["load_voc_instances", "register_pascal_voc"]

VOC_CLASS_NAMES_COCOFIED = [
    "airplane", "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant", "sofa", "tvmonitor"
]

UNK_CLASS = ["unknown"]

VOC_COCO_CLASS_NAMES = {}

VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

T2_CLASS_NAMES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

T4_CLASS_NAMES = [
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl"
]

VOC_COCO_CLASS_NAMES["SAOWOD"] = tuple(
    itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))


T1_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bus", "car",
    "cat", "cow", "dog", "horse", "motorbike", "sheep", "train",
    "elephant", "bear", "zebra", "giraffe", "truck", "person"
]

T2_CLASS_NAMES = [
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "chair", "diningtable",
    "pottedplant", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "microwave", "oven", "toaster", "sink",
    "refrigerator", "bed", "toilet", "sofa"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"
]

T4_CLASS_NAMES = [
    "laptop", "mouse", "remote", "keyboard", "cell phone", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "tvmonitor", "bottle"
]


def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]], cfg):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)
    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/", cfg.DATASETS.SETTING))
    dicts = []

    # PROB and CAT convert image id to int before iterating over image ids
    # RandBox uses COCO's loader, which implicitly converts image id to int
    ids = []
    id2fileids = {}
    for fileid in fileids:
        id = int(fileid.split('.')[0])
        ids.append(id)
        id2fileids[id] = fileid

    # filter instances
    if cfg.TEST.MASK == 1:
        allowed_class = list(range(0, cfg.TEST.PREV_INTRODUCED_CLS+cfg.TEST.CUR_INTRODUCED_CLS))
    else:
        allowed_class = list(range(cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.PREV_INTRODUCED_CLS+cfg.TEST.CUR_INTRODUCED_CLS))
        
    # Now lets load the additional unknown class annotations if exists
    if os.path.exists(cfg.DISCOVER_STORE_PATH) and ("test" not in split):
        print('Found Additional unknown annotations for split {}. Loading from: {}'.format(split, cfg.DISCOVER_STORE_PATH))
        # load the json and store the keys separately
        with open(cfg.DISCOVER_STORE_PATH, 'r') as file:
            unknown_annotations = json.load(file)
        unknown_file_ids = unknown_annotations.keys() # this is a list   
    else:
        unknown_file_ids = [] # empty             
    
    for id in ids:
        fileid = id2fileids[id]
    # for fileid in id2fileids.values():
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        try:
            with PathManager.open(anno_file) as f:
                tree = ET.parse(f)
        except:
            logger = logging.getLogger(__name__)
            logger.info('Not able to load: ' + anno_file + '. Continuing without aboarting...')
            continue

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls in VOC_CLASS_NAMES_COCOFIED:
                cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
            if cfg.TEST.MASK and ('test' not in split):
                if class_names.index(cls) not in allowed_class:
                    continue
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        
        # check if this file_id has additional unknown annotations
        if fileid in unknown_file_ids:
            additional_annot = unknown_annotations[fileid]
            for obj_idx in range(len(additional_annot["labels"])):
                bbox = additional_annot['bboxes'][obj_idx]
                instances.append(
                    {"category_id": class_names.index("unknown"), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
                )
        
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_pascal_voc(name, dirname, split, cfg, year=2007):
    # if "voc_coco" in name:
    #     class_names = VOC_COCO_CLASS_NAMES
    # else:
    #     class_names = tuple(VOC_CLASS_NAMES)
    class_names = VOC_COCO_CLASS_NAMES["SAOWOD"]
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names, cfg))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )
