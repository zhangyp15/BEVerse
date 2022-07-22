# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset, output_to_nusc_box, lidar_nusc_box_to_global
from mmdet3d.datasets.pipelines import Compose
from .utils import VectorizedLocalMap, preprocess_map
from nuscenes.nuscenes import NuScenes
from .utils.geometry import invert_matrix_egopose_numpy, mat2pose_vec

import pdb


@DATASETS.register_module()
class MTLEgoNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 coordinate_system='ego',
                 filter_empty_gt=True,
                 test_mode=False,
                 grid_conf=None,
                 map_grid_conf=None,
                 receptive_field=1,
                 future_frames=0,
                 eval_version='detection_cvpr_2019',
                 filter_invalid_sample=False,
                 use_valid_flag=True):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            use_valid_flag=use_valid_flag,
            load_interval=load_interval,
            with_velocity=with_velocity,
            eval_version=eval_version,
        )

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        # whether test-set
        self.test_submission = 'test' in self.ann_file

        # temporal settings
        self.receptive_field = receptive_field
        self.n_future = future_frames
        self.sequence_length = receptive_field + future_frames
        self.filter_invalid_sample = filter_invalid_sample
        self.coordinate_system = coordinate_system

        # default, we use the LiDAR coordinate system as the BEV system
        assert self.coordinate_system in ['lidar', 'ego']

        # for vector maps
        self.map_dataroot = self.data_root
        
        map_xbound, map_ybound = grid_conf['xbound'], grid_conf['ybound']
        patch_h = map_ybound[1] - map_ybound[0]
        patch_w = map_xbound[1] - map_xbound[0]
        canvas_h = int(patch_h / map_ybound[2])
        canvas_w = int(patch_w / map_xbound[2])
        self.map_patch_size = (patch_h, patch_w)
        self.map_canvas_size = (canvas_h, canvas_w)

        # hdmap settings
        self.map_max_channel = 3
        self.map_thickness = 5
        self.map_angle_class = 36
        self.vector_map = VectorizedLocalMap(
            dataroot=self.map_dataroot,
            patch_size=self.map_patch_size,
            canvas_size=self.map_canvas_size,
        )

        # process infos so that they are sorted w.r.t. scenes & time_stamp
        self.data_infos.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        self._set_group_flag()

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_temporal_indices(self, index):
        current_scene_token = self.data_infos[index]['scene_token']

        # generate the past
        previous_indices = []
        for t in range(- self.receptive_field + 1, 0):
            index_t = index + t
            if index_t >= 0 and self.data_infos[index_t]['scene_token'] == current_scene_token:
                previous_indices.append(index_t)
            else:
                previous_indices.append(-1)  # for invalid indices

        # generate the future
        future_indices = []
        for t in range(1, self.n_future + 1):
            index_t = index + t
            if index_t < len(self.data_infos) and self.data_infos[index_t]['scene_token'] == current_scene_token:
                future_indices.append(index_t)
            else:
                future_indices.append(-1)

        return previous_indices, future_indices

    @staticmethod
    def get_egopose_from_info(info):
        # ego2global transformation (lidar_ego)
        e2g_trans_matrix = np.zeros((4, 4), dtype=np.float32)
        e2g_rot = info['ego2global_rotation']
        e2g_trans = info['ego2global_translation']
        e2g_trans_matrix[:3, :3] = pyquaternion.Quaternion(
            e2g_rot).rotation_matrix
        e2g_trans_matrix[:3, 3] = np.array(e2g_trans)
        e2g_trans_matrix[3, 3] = 1.0

        return e2g_trans_matrix

    def get_egomotions(self, indices):
        # get ego_motion for each frame
        future_egomotions = []
        for index in indices:
            cur_info = self.data_infos[index]
            ego_motion = np.eye(4, dtype=np.float32)
            next_frame = index + 1

            # 如何处理 invalid frame
            if index != -1 and next_frame < len(self.data_infos) and self.data_infos[next_frame]['scene_token'] == cur_info['scene_token']:
                next_info = self.data_infos[next_frame]
                # get ego2global transformation matrices
                cur_egopose = self.get_egopose_from_info(cur_info)
                next_egopose = self.get_egopose_from_info(next_info)

                # trans from cur to next
                ego_motion = invert_matrix_egopose_numpy(
                    next_egopose).dot(cur_egopose)
                ego_motion[3, :3] = 0.0
                ego_motion[3, 3] = 1.0

            # transformation between adjacent frames
            ego_motion = torch.Tensor(ego_motion).float()
            ego_motion = mat2pose_vec(ego_motion)
            future_egomotions.append(ego_motion)

        return torch.stack(future_egomotions, dim=0)

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        # when the labels for future frames are not complete, skip the sample
        if self.filter_invalid_sample and input_dict['has_invalid_frame'] is True:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        if self.filter_empty_gt and (example is None or
                                     ~(example['gt_labels_3d']._data != -1).any()):
            return None

        return example

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            data_root=self.data_root,
        )

        # info_keys = dict_keys(['lidar_path', 'scene_token', 'location', 'token', 'sweeps', 'cams', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts', 'valid_flag', 'instance_tokens'])

        # get temporal indices
        prev_indices, future_indices = self.get_temporal_indices(index)

        # ego motions are needed for all frames
        all_frames = prev_indices + [index] + future_indices
        # [num_seq, 6 DoF]
        future_egomotions = self.get_egomotions(all_frames)
        input_dict['future_egomotions'] = future_egomotions

        # whether invalid frame is present
        has_invalid_frame = -1 in all_frames
        input_dict['has_invalid_frame'] = has_invalid_frame
        input_dict['img_is_valid'] = np.array(all_frames) >= 0

        # for past frames, we need images, camera paramters, depth(optional)
        img_infos = []
        for prev_index in prev_indices:
            if prev_index >= 0:
                img_infos.append(self.data_infos[prev_index]['cams'])
            else:
                # get the information of current frame for invalid frames
                img_infos.append(info['cams'])

        # current frame
        img_infos.append(info['cams'])
        input_dict['img_info'] = img_infos

        input_dict['lidar2ego_rots'] = torch.tensor(pyquaternion.Quaternion(
            info['lidar2ego_rotation']).rotation_matrix)
        input_dict['lidar2ego_trans'] = torch.tensor(
            info['lidar2ego_translation'])

        # for future frames, we need detection labels
        if not self.test_submission:
            # generate detection labels for current + future frames
            label_frames = [index] + future_indices
            detection_ann_infos = []
            for label_frame in label_frames:
                if label_frame >= 0:
                    detection_ann_infos.append(
                        self.get_detection_ann_info(label_frame))
                else:
                    detection_ann_infos.append(None)

            input_dict['ann_info'] = detection_ann_infos
            # generate map labels only for the current frame
            input_dict['vectors'] = self.get_map_ann_info(info)

        return input_dict

    def get_detection_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # still need the gt_tokens
        gt_bboxes_3d = info["gt_boxes"]
        gt_names_3d = info["gt_names"]
        gt_instance_tokens = info["instance_tokens"]
        gt_valid_flag = info['valid_flag']
        gt_vis_tokens = info['visibility_tokens']

        if self.use_valid_flag:
            gt_valid_flag = info['valid_flag']
        else:
            gt_valid_flag = info['num_lidar_pts'] > 0

        # cls_name ==> cls_id
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity']
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        # convert 3DBoxes from LiDAR frame to Ego frame
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_rotation = pyquaternion.Quaternion(
            lidar2ego_rotation).rotation_matrix

        gt_bboxes_3d.rotate(lidar2ego_rotation.T)
        gt_bboxes_3d.translate(lidar2ego_translation)

        # # reverse
        # gt_bboxes_3d.translate([-x for x in lidar2ego_translation])
        # gt_bboxes_3d.rotate(np.linalg.inv(lidar2ego_rotation.T))

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            instance_tokens=gt_instance_tokens,
            gt_valid_flag=gt_valid_flag,
            gt_vis_tokens=gt_vis_tokens,
        )

        return anns_results

    def get_map_ann_info(self, info):
        # get the annotations of HD maps
        vectors = self.vector_map.gen_vectorized_samples(
            info['location'], info['ego2global_translation'], info['ego2global_rotation'])

        # type0 = 'divider'
        # type1 = 'pedestrain'
        # type2 = 'boundary'

        for vector in vectors:
            pts = vector['pts']
            vector['pts'] = np.concatenate(
                (pts, np.zeros((pts.shape[0], 1))), axis=1)

        return vectors

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return results_dict

    def show_results(self, results, out_dir, targets=None):
        # visualize the predictions & ground-truth
        pass

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)
