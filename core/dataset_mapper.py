import copy
import logging
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

__all__ = ["DatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    # if is_train:
    #     tfm_gens.append(T.RandomFlip())
    # ResizeShortestEdge
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by our model.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if is_train:
            self.nor_gen = [
                T.RandomFlip(prob=0.5, horizontal=True), # weak
                T.RandomBrightness(0.5, 2.0), # weak
                T.RandomContrast(0.5, 2.0), # weak
                T.RandomSaturation(0.5, 2.0), # weak
            ]
            self.str_gen = [
                T.RandomFlip(prob=0.5, horizontal=True), # strong
                T.RandomBrightness(0.5, 2.0), # strong
                T.RandomContrast(0.5, 2.0), # strong
                T.RandomSaturation(0.5, 2.0), # strong
                T.RandomLighting(0.2), # strong
                T.RandomRotation(angle=[-30, 30], expand=False), # strong
            ]
        else:
            self.nor_gen = None
            self.str_gen = None
        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.nor_gen), str(self.str_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def identity_matrix(self):
        return np.eye(3, dtype=float)

    def flip_matrix(self, width):
        return np.array(
            [[-1.0, 0.0, width],
             [ 0.0, 1.0,   0.0],
             [ 0.0, 0.0,   1.0]], dtype=float
        )
    
    def rotation_matrix_3x3(self, rotation_transform):
        if rotation_transform is None:
            return self.identity_matrix()
        rm = rotation_transform.rm_coords  # 2x3 affine matrix
        return np.vstack([rm, [0, 0, 1]])
    
    def resize_matrix(self, old_h, old_w, new_h, new_w):
        scale_x = new_w / old_w
        scale_y = new_h / old_h
        return np.array(
            [[scale_x, 0.0, 0.0],
             [0.0, scale_y, 0.0],
             [0.0, 0.0, 1.0]], dtype=float
        )

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        
        image_raw, transforms_raw = T.apply_transform_gens(self.tfm_gens, image)

        if self.nor_gen is None:
            image_nor, transforms_nor = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image_nor, transforms_nor = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image_nor, transforms_nor = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.nor_gen + self.tfm_gens[-1:], image
                )
                
        if self.str_gen is None:
            image_str, transforms_str = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image_str, transforms_str = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image_str, transforms_str = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.str_gen + self.tfm_gens[-1:], image
                )

        image_raw_shape = image_raw.shape[:2]  # h, w
        image_nor_shape = image_nor.shape[:2]  # h, w
        image_str_shape = image_str.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image_raw"] = torch.as_tensor(np.ascontiguousarray(image_raw.transpose(2, 0, 1)))
        dataset_dict["image_nor"] = torch.as_tensor(np.ascontiguousarray(image_nor.transpose(2, 0, 1)))
        dataset_dict["image_str"] = torch.as_tensor(np.ascontiguousarray(image_str.transpose(2, 0, 1)))
        
        image_raw_matrix = self.identity_matrix()
        for t in transforms_raw.transforms:
            if isinstance(t, T.HFlipTransform):
                flip_mat = self.flip_matrix(t.width)
                image_raw_matrix = flip_mat @ image_raw_matrix
            elif isinstance(t, T.RotationTransform):
                rot_mat = self.rotation_matrix_3x3(t)
                image_raw_matrix = rot_mat @ image_raw_matrix
            elif isinstance(t, T.ResizeTransform):
                resize_mat = self.resize_matrix(t.h, t.w, t.new_h, t.new_w)
                image_raw_matrix = resize_mat @ image_raw_matrix
        dataset_dict["image_raw_matrix"] = image_raw_matrix
        
        image_nor_matrix = self.identity_matrix()
        for t in transforms_nor.transforms:
            if isinstance(t, T.HFlipTransform):
                flip_mat = self.flip_matrix(t.width)
                image_nor_matrix = flip_mat @ image_nor_matrix
            elif isinstance(t, T.RotationTransform):
                rot_mat = self.rotation_matrix_3x3(t)
                image_nor_matrix = rot_mat @ image_nor_matrix
            elif isinstance(t, T.ResizeTransform):
                resize_mat = self.resize_matrix(t.h, t.w, t.new_h, t.new_w)
                image_nor_matrix = resize_mat @ image_nor_matrix
        dataset_dict["image_nor_matrix"] = image_nor_matrix
        
        # str 이미지 변환 행렬 계산 (transforms가 적용되는 순서대로 행렬을 누적)
        image_str_matrix = self.identity_matrix()
        for t in transforms_str.transforms:
            if isinstance(t, T.HFlipTransform):
                flip_mat = self.flip_matrix(t.width)
                image_str_matrix = flip_mat @ image_str_matrix
            elif isinstance(t, T.RotationTransform):
                rot_mat = self.rotation_matrix_3x3(t)
                image_str_matrix = rot_mat @ image_str_matrix
            elif isinstance(t, T.ResizeTransform):
                resize_mat = self.resize_matrix(t.h, t.w, t.new_h, t.new_w)
                image_str_matrix = resize_mat @ image_str_matrix
        dataset_dict["image_str_matrix"] = image_str_matrix

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            annotations = dataset_dict.pop("annotations")
            # USER: Implement additional transformations if you have other types of data
            annos_raw = [
                utils.transform_instance_annotations(copy.deepcopy(obj), transforms_raw, image_raw_shape)
                for obj in annotations
                if obj.get("iscrowd", 0) == 0
            ]
            annos_nor = [
                utils.transform_instance_annotations(copy.deepcopy(obj), transforms_nor, image_nor_shape)
                for obj in annotations
                if obj.get("iscrowd", 0) == 0
            ]
            annos_str = [
                utils.transform_instance_annotations(copy.deepcopy(obj), transforms_str, image_str_shape)
                for obj in annotations
                if obj.get("iscrowd", 0) == 0
            ]
            instances_raw = utils.annotations_to_instances(annos_raw, image_raw_shape)
            instances_nor = utils.annotations_to_instances(annos_nor, image_nor_shape)
            instances_str = utils.annotations_to_instances(annos_str, image_str_shape)
            dataset_dict["instances_raw"] = utils.filter_empty_instances(instances_raw)
            dataset_dict["instances_nor"] = utils.filter_empty_instances(instances_nor)
            dataset_dict["instances_str"] = utils.filter_empty_instances(instances_str)
        return dataset_dict
