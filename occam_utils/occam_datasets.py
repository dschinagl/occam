import numpy as np
import torch

from spconv.pytorch.utils import PointToVoxel
from scipy.spatial.transform import Rotation

from pcdet.datasets import DatasetTemplate


class BaseDataset(DatasetTemplate):
    """
    OpenPCDet dataset to load and preprocess the point cloud
    """
    def __init__(self, data_config, class_names, occam_config):
        """
        Parameters
        ----------
            data_config : EasyDict
               dataset cfg including data preprocessing properties (OpenPCDet)
            class_names :
                list of class names (OpenPCDet)
             occam_config: EasyDict
                sampling properties for attribution map generation, see cfg file
        """
        super().__init__(dataset_cfg=data_config, class_names=class_names,
                         training=False)
        self.occam_config = occam_config

    def load_and_preprocess_pcl(self, source_file_path):
        """
        load given point cloud file and preprocess data according OpenPCDet cfg

        Parameters
        ----------
        source_file_path : str
            path to point cloud to analyze (bin or npy)

        Returns
        -------
        pcl : ndarray (N, 4)
            preprocessed point cloud (x, y, z, intensity)
        """

        if source_file_path.split('.')[-1] == 'bin':
            points = np.fromfile(source_file_path, dtype=np.float32)
            points = points.reshape(-1, 4)
        elif source_file_path.split('.')[-1] == 'npy':
            points = np.load(source_file_path)
        else:
            raise NotImplementedError

        # FOV crop is usually done using the image
        if self.occam_config.FOV_CROP:
            angles = np.abs(np.degrees(np.arctan2(points[:, 1], points[:, 0])))
            mask = angles <= self.occam_config.FOV_ANGLE
            points = points[mask, :]

        input_dict = {
            'points': points
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        pcl = data_dict['points']
        return pcl


class OccamInferenceDataset(DatasetTemplate):
    """
    OpenPCDet dataset for occam inference; in each iteration a sub-sampled
    point cloud according occam config is generated
    """
    def __init__(self, data_config, class_names, occam_config, pcl, nr_it, logger):
        """
        Parameters
        ----------
            data_config : EasyDict
                dataset cfg including data preprocessing properties (OpenPCDet)
            class_names :
                list of class names (OpenPCDet)
            occam_config: EasyDict
                sampling properties for attribution map generation, see cfg file
            pcl : ndarray (N, 4)
                preprocessed full point cloud
            nr_it : int
                number of sub-sampling iterations
            logger : Logger
        """
        super().__init__(
            dataset_cfg=data_config, class_names=class_names, training=False,
            root_path=None, logger=logger
        )

        self.occam_config = occam_config
        self.pcl = pcl
        self.logger = logger
        self.nr_it = nr_it

        self.sampling_rand_rot = self.occam_config.SAMPLING.RANDOM_ROT
        self.sampling_vx_size = np.array(self.occam_config.SAMPLING.VOXEL_SIZE)
        self.lbda = self.occam_config.SAMPLING.LAMBDA # see paper
        self.sampling_density_coeff = np.array(
            self.occam_config.SAMPLING.DENSITY_DISTR_COEFF)
        self.sampling_range = self.get_sampling_range(
            rand_rot=self.sampling_rand_rot,
            pcl=self.pcl,
            vx_size=self.sampling_vx_size
        )

        self.voxel_generator = PointToVoxel(
            vsize_xyz=list(self.sampling_vx_size),
            coors_range_xyz=list(self.sampling_range),
            num_point_features=3,
            max_num_points_per_voxel=self.occam_config.SAMPLING.MAX_PTS_PER_VOXEL,
            max_num_voxels=self.occam_config.SAMPLING.MAX_VOXELS
        )

    def get_sampling_range(self, rand_rot, pcl, vx_size):
        """
        compute min/max sampling range for given random rotation

        Parameters
        ----------
        rand_rot : float
            max random rotation before sampling (+/-) in degrees
        pcl : ndarray (N, 4)
            full point cloud
        vx_size : ndarray (3)
            voxel size for sampling in x, y, z

        Returns
        -------
        sampling_range : ndarray (6)
            min/max sampling range for given rotation
        """
        rotmat_pos = Rotation.from_rotvec([0, 0, rand_rot], degrees=True)
        rotmat_neg = Rotation.from_rotvec([0, 0, -rand_rot], degrees=True)

        rot_pts = np.concatenate(
            (np.matmul(rotmat_pos.as_matrix(), pcl[:, :3].T),
             np.matmul(rotmat_neg.as_matrix(), pcl[:, :3].T)), axis=1)

        min_grid = np.floor(np.min(rot_pts, axis=1) / vx_size) * vx_size - vx_size
        max_grid = np.ceil(np.max(rot_pts, axis=1) / vx_size) * vx_size + vx_size

        sampling_range = np.concatenate((min_grid, max_grid))
        return sampling_range

    def __len__(self):
        return self.nr_it

    def __getitem__(self, index):
        if index == self.nr_it:
            raise IndexError

        # randomly rotate and translate full pcl
        rand_transl = np.random.rand(1, 3) * (self.sampling_vx_size[None, :])
        rand_transl -= self.sampling_vx_size[None, :] / 2

        rand_rot_ = np.random.rand(1) * self.sampling_rand_rot * 2 \
                    - self.sampling_rand_rot
        rand_rot_mat = Rotation.from_rotvec([0, 0, rand_rot_[0]], degrees=True)
        rand_rot_mat = rand_rot_mat.as_matrix()

        rand_rot_pcl = np.matmul(rand_rot_mat, self.pcl[:, :3].T).T
        rand_rot_transl_pcl = rand_rot_pcl + rand_transl
        rand_rot_transl_pcl = np.ascontiguousarray(rand_rot_transl_pcl)

        # voxelixe full pcl
        _, vx_coord, _, pt_vx_id = self.voxel_generator.generate_voxel_with_id(
            torch.from_numpy(rand_rot_transl_pcl))
        vx_coord, pt_vx_id = vx_coord.numpy(), pt_vx_id.numpy()
        vx_coord = vx_coord[:, [2, 1, 0]]

        # compute voxel center in original pcl
        vx_orig_coord = vx_coord * self.sampling_vx_size[None, :]
        vx_orig_coord += self.sampling_range[:3][None, :]
        vx_orig_coord += self.sampling_vx_size[None, :] / 2
        vx_orig_coord -= rand_transl
        vx_orig_coord = np.matmul(np.linalg.inv(rand_rot_mat), vx_orig_coord.T).T

        vx_dist = np.linalg.norm(vx_orig_coord, axis=1)
        vx_keep_prob = self.lbda * (
                np.power(vx_dist, 2) * self.sampling_density_coeff[0]
                + vx_dist * self.sampling_density_coeff[1]
                + self.sampling_density_coeff[2])

        vx_keep_ids = np.where(np.random.rand(vx_keep_prob.shape[0]) < vx_keep_prob)[0]
        pt_keep_mask = np.in1d(pt_vx_id, vx_keep_ids)

        input_dict = {
            'points': self.pcl[pt_keep_mask, :],
            'mask': pt_keep_mask
        }

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
