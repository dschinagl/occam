import copy
import torch
import open3d
import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from torch.utils.data import DataLoader

from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from scipy.spatial.transform import Rotation

from occam_utils.occam_datasets import BaseDataset, OccamInferenceDataset


class OccAM(object):
    """
    OccAM base class to store model, cfg and offer operations to preprocess the
    data and compute the attribution maps
    """

    def __init__(self, data_config, model_config, occam_config, class_names,
                 model_ckpt_path, nr_it, logger):
        """
        Parameters
        ----------
            data_config : EasyDict
               dataset cfg including data preprocessing properties (OpenPCDet)
            model_config : EasyDict
               object detection model definition (OpenPCDet)
            occam_config: EasyDict
                sampling properties for attribution map generation, see cfg file
            class_names :
                list of class names (OpenPCDet)
            model_ckpt_path: str
                path to pretrained model weights
            nr_it : int
                number of sub-sampling iterations; the higher, the more accurate
                are the resulting attribution maps
            logger: Logger
        """
        self.data_config = data_config
        self.model_config = model_config
        self.occam_config = occam_config
        self.class_names = class_names
        self.logger = logger
        self.nr_it = nr_it

        self.base_dataset = BaseDataset(data_config=self.data_config,
                                        class_names=self.class_names,
                                        occam_config=self.occam_config)

        self.model = build_network(model_cfg=self.model_config,
                                   num_class=len(self.class_names),
                                   dataset=self.base_dataset)
        self.model.load_params_from_file(filename=model_ckpt_path,
                                         logger=logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()

    def load_and_preprocess_pcl(self, source_file_path):
        """
        load given point cloud file and preprocess data according OpenPCDet
        data config using the base dataset

        Parameters
        ----------
        source_file_path : str
            path to point cloud to analyze (bin or npy)

        Returns
        -------
        pcl : ndarray (N, 4)
            preprocessed point cloud (x, y, z, intensity)
        """
        pcl = self.base_dataset.load_and_preprocess_pcl(source_file_path)
        return pcl

    def get_base_predictions(self, pcl):
        """
        get all K detections in full point cloud for which attribution maps will
        be determined

        Parameters
        ----------
        pcl : ndarray (N, 4)
            preprocessed point cloud (x, y, z, intensity)

        Returns
        -------
        base_det_boxes : ndarray (K, 7)
            bounding box parameters of detected objects
        base_det_labels : ndarray (K)
            labels of detected objects
        base_det_scores : ndarray (K)
            confidence scores for detected objects
        """
        input_dict = {
            'points': pcl
        }

        data_dict = self.base_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.base_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        with torch.no_grad():
            base_pred_dict, _ = self.model.forward(data_dict)

        base_det_boxes = base_pred_dict[0]['pred_boxes'].cpu().numpy()
        base_det_labels = base_pred_dict[0]['pred_labels'].cpu().numpy()
        base_det_scores = base_pred_dict[0]['pred_scores'].cpu().numpy()

        return base_det_boxes, base_det_labels, base_det_scores

    def merge_detections_in_batch(self, det_dicts):
        """
        In order to efficiently determine the confidence score for
        all detections in a batch they are merged.

        Parameters
        ----------
        det_dicts : list
            list of M dicts containing the detections in the M samples within
            the batch (pred boxes, pred scores, pred labels)

        Returns
        -------
        pert_det_boxes : ndarray (L, 7)
            bounding boxes of all L detections in the M samples
        pert_det_labels : ndarray (L)
            labels of all L detections in the M samples
        pert_det_scores : ndarray (L)
            scores of all L detections in the M samples
        batch_ids : ndarray (L)
            Mapping of the detections to the individual samples within the batch
        """
        batch_ids = []

        data_dict = defaultdict(list)
        for batch_id, cur_sample in enumerate(det_dicts):
            batch_ids.append(
                np.ones(cur_sample['pred_labels'].shape[0], dtype=int)
                * batch_id)

            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_ids = np.concatenate(batch_ids, axis=0)

        merged_dict = {}
        for key, val in data_dict.items():
            if key in ['pred_boxes', 'pred_scores', 'pred_labels']:
                merged_data = []
                for data in val:
                    data = data.cpu().numpy()
                    merged_data.append(data)
                merged_dict[key] = np.concatenate(merged_data, axis=0)

        pert_det_boxes = merged_dict['pred_boxes']
        pert_det_labels = merged_dict['pred_labels']
        pert_det_scores = merged_dict['pred_scores']
        return pert_det_boxes, pert_det_labels, pert_det_scores, batch_ids

    def compute_iou(self, base_boxes, pert_boxes):
        """
        3D IoU between base and perturbed detections
        """
        base_boxes = torch.from_numpy(base_boxes)
        pert_boxes = torch.from_numpy(pert_boxes)
        base_boxes, pert_boxes = base_boxes.cuda(), pert_boxes.cuda()
        iou = boxes_iou3d_gpu(base_boxes, pert_boxes)
        iou = iou.cpu().numpy()
        return iou

    def compute_translation_score(self, base_boxes, pert_boxes):
        """
        translation score (see paper for details)
        """
        translation_error = np.linalg.norm(
            base_boxes[:, :3][:, None, :] - pert_boxes[:, :3], axis=2)
        translation_score = 1 - translation_error
        translation_score[translation_score < 0] = 0
        return translation_score

    def compute_orientation_score(self, base_boxes, pert_boxes):
        """
        orientation score (see paper for details)
        """
        boxes_a = copy.deepcopy(base_boxes)
        boxes_b = copy.deepcopy(pert_boxes)

        boxes_a[:, 6] = boxes_a[:, 6] % (2 * math.pi)
        boxes_a[boxes_a[:, 6] > math.pi, 6] -= 2 * math.pi
        boxes_a[boxes_a[:, 6] < -math.pi, 6] += 2 * math.pi
        boxes_b[:, 6] = boxes_b[:, 6] % (2 * math.pi)
        boxes_b[boxes_b[:, 6] > math.pi, 6] -= 2 * math.pi
        boxes_b[boxes_b[:, 6] < -math.pi, 6] += 2 * math.pi
        orientation_error_ = np.abs(
            boxes_a[:, 6][:, None] - boxes_b[:, 6][None, :])
        orientation_error__ = 2 * math.pi - np.abs(
            boxes_a[:, 6][:, None] - boxes_b[:, 6][None, :])
        orientation_error = np.concatenate(
            (orientation_error_[:, :, None], orientation_error__[:, :, None]),
            axis=2)
        orientation_error = np.min(orientation_error, axis=2)
        orientation_score = 1 - orientation_error
        orientation_score[orientation_score < 0] = 0
        return orientation_score

    def compute_scale_score(self, base_boxes, pert_boxes):
        """
        scale score (see paper for details)
        """
        boxes_centered_a = copy.deepcopy(base_boxes)
        boxes_centered_b = copy.deepcopy(pert_boxes)
        boxes_centered_a[:, :3] = 0
        boxes_centered_a[:, 6] = 0
        boxes_centered_b[:, :3] = 0
        boxes_centered_b[:, 6] = 0
        scale_score = self.compute_iou(boxes_centered_a, boxes_centered_b)
        scale_score[scale_score < 0] = 0
        return scale_score

    def get_similarity_matrix(self, base_det_boxes, base_det_labels,
                              pert_det_boxes, pert_det_labels, pert_det_scores):
        """
        compute similarity score between the base detections in the full
        point cloud and the detections in the perturbed samples

        Parameters
        ----------
        base_det_boxes : (K, 7)
            bounding boxes of detected objects in full pcl
        base_det_labels : (K)
            class labels of detected objects in full pcl
        pert_det_boxes : ndarray (L, 7)
            bounding boxes of all L detections in the perturbed samples of the batch
        pert_det_labels : ndarray (L)
            labels of all L detections in the perturbed samples of the batch
        pert_det_scores : ndarray (L)
            scores of all L detections in the perturbed samples of the batch
        Returns
        -------
        sim_scores : ndarray (K, L)
            similarity score between all K detections in the full pcl and
            the L detections in the perturbed samples within the batch
        """
        # similarity score is only greater zero if boxes overlap
        s_overlap = self.compute_iou(base_det_boxes, pert_det_boxes) > 0
        s_overlap = s_overlap.astype(np.float32)

        # similarity score is only greater zero for boxes of same class
        s_class = base_det_labels[:, None] == pert_det_labels[None, :]
        s_class = s_class.astype(np.float32)

        # confidence score is directly used (see paper)
        s_conf = np.repeat(pert_det_scores[None, :], base_det_boxes.shape[0], axis=0)

        s_transl = self.compute_translation_score(base_det_boxes, pert_det_boxes)

        s_orient = self.compute_orientation_score(base_det_boxes, pert_det_boxes)

        s_score = self.compute_scale_score(base_det_boxes, pert_det_boxes)

        sim_scores = s_overlap * s_conf * s_transl * s_orient * s_score * s_class

        return sim_scores

    def compute_attribution_maps(self, pcl, base_det_boxes, base_det_labels,
                                 batch_size, num_workers):
        """
        attribution map computation for each base detection

        Parameters
        ----------
        pcl : ndarray (N, 4)
            preprocessed full point cloud (x, y, z, intensity)
        base_det_boxes : ndarray (K, 7)
            bounding boxes of detected objects in full pcl
        base_det_labels : ndarray (K)
            class labels of detected objects in full pcl
        batch_size : int
            batch_size during AM computation
        num_workers : int
            number of dataloader workers

        Returns
        -------
        attr_maps : ndarray (K, N)
            attribution scores for all K detected base objects and all N points
        """

        attr_maps = np.zeros((base_det_labels.shape[0], pcl.shape[0]))
        # count number of occurrences of each point in sampled pcl's
        sampling_map = np.zeros(pcl.shape[0])

        occam_inference_dataset = OccamInferenceDataset(
            data_config=self.data_config, class_names=self.class_names,
            occam_config=self.occam_config, pcl=pcl, nr_it=self.nr_it,
            logger=self.logger
        )

        dataloader = DataLoader(
            occam_inference_dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, shuffle=False,
            collate_fn=occam_inference_dataset.collate_batch, drop_last=False,
            sampler=None, timeout=0
        )

        progress_bar = tqdm.tqdm(
            total=self.nr_it, leave=True, desc='OccAM computation',
            dynamic_ncols=True)

        with torch.no_grad():
            for i, batch_dict in enumerate(dataloader):

                load_data_to_gpu(batch_dict)
                pert_pred_dicts, _ = self.model.forward(batch_dict)

                pert_det_boxes, pert_det_labels, pert_det_scores, batch_ids = \
                    self.merge_detections_in_batch(pert_pred_dicts)

                similarity_matrix = self.get_similarity_matrix(
                    base_det_boxes, base_det_labels,
                    pert_det_boxes, pert_det_labels, pert_det_scores)

                cur_batch_size = len(pert_pred_dicts)
                for j in range(cur_batch_size):
                    cur_mask = batch_dict['mask'][j, :].cpu().numpy()
                    sampling_map += cur_mask

                    batch_sample_mask = batch_ids == j
                    if np.sum(batch_sample_mask) > 0:
                        max_score = np.max(
                            similarity_matrix[:, batch_sample_mask], axis=1)
                        attr_maps += max_score[:, None] * cur_mask

                progress_bar.update(n=cur_batch_size)

        progress_bar.close()

        # normalize using occurrences
        attr_maps[:, sampling_map > 0] /= sampling_map[sampling_map > 0]

        return attr_maps

    def visualize_attr_map(self, points, box, attr_map, draw_origin=True):
        turbo_cmap = plt.get_cmap('turbo')
        attr_map_scaled = attr_map - attr_map.min()
        attr_map_scaled /= attr_map_scaled.max()
        color = turbo_cmap(attr_map_scaled)[:, :3]

        vis = open3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().point_size = 4.0
        vis.get_render_option().background_color = np.ones(3) * 0.25

        if draw_origin:
            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0])
            vis.add_geometry(axis_pcd)

        rot_mat = Rotation.from_rotvec([0, 0, box[6]]).as_matrix()
        bb = open3d.geometry.OrientedBoundingBox(box[:3], rot_mat, box[3:6])
        bb.color = (1.0, 0.0, 1.0)
        vis.add_geometry(bb)

        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        pts.colors = open3d.utility.Vector3dVector(color)
        vis.add_geometry(pts)

        vis.run()
        vis.destroy_window()
        
        
