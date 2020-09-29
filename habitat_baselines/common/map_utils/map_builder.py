import numpy as np

import habitat_baselines.common.map_utils.depth_utils as du


class MapBuilder(object):
    def __init__(self, params):
        self.params = params
        frame_width = params['frame_width']
        frame_height = params['frame_height']
        fov = params['fov']
        self.camera_matrix = du.get_camera_matrix(
            frame_width,
            frame_height,
            fov)
        self.vision_range = params['vision_range']

        self.map_size_cm = params['map_size_cm']
        self.resolution = params['resolution']
        agent_min_z = params['agent_min_z']
        agent_max_z = params['agent_max_z']
        self.z_bins = [agent_min_z, agent_max_z]
        self.du_scale = params['du_scale']
        self.visualize = params['visualize']
        self.obs_threshold = params['obs_threshold']

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

        self.semantic_map = np.zeros(self.map.shape)

        self.agent_height = params['agent_height']
        self.agent_view_angle = params['agent_view_angle']
        return

    def update_map(self, depth, semantic, current_pose):
        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range * self.resolution] = np.NaN
        point_cloud = du.get_point_cloud_from_z(depth, 
                                                self.camera_matrix, 
                                                scale=self.du_scale)
        S = semantic[::self.du_scale, ::self.du_scale] # 128*128

        agent_view = du.transform_camera_view(point_cloud,
                                              self.agent_height,
                                              self.agent_view_angle)

        shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        agent_view_centered = du.transform_pose(agent_view, shift_loc)

        agent_view_flat = du.bin_points(
            agent_view_centered,
            self.vision_range,
            self.z_bins,
            self.resolution)

        agent_view_cropped = agent_view_flat[:, :, 1]

        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0

        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.0

        geocentric_pc = du.transform_pose(agent_view, current_pose)

        # print("geocentric_pc : ", geocentric_pc.shape) # 128*128*3

        geocentric_flat = du.bin_points(
            geocentric_pc,
            self.map.shape[0],
            self.z_bins, # 三层，在z_bins区间中的为1，否则为0或2
            self.resolution)

        semantic_flat = du.bin_semantic_points(
            geocentric_pc, # 128*128*3
            S, #128*128
            self.map.shape[0], 
            self.semantic_map_len,
            self.resolution
        )

        # print("semantic_flat :", semantic_flat.shape) # 480*480*21
        # print("geocentric_flat :", geocentric_flat.shape) # 480*480*3

        self.map = self.map + geocentric_flat
        self.semantic_map = self.semantic_map + semantic_flat
        # print("********semantic map : ", self.semantic_map.shape) # 480*480*21


        map_gt = self.map[:, :, 1] / self.obs_threshold # 取中间层，为在检测区间中的
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        map_se = self.semantic_map
        map_se[map_se >= 0.5] = 1.0
        map_se[map_se < 0.5] = 0.0

        explored_gt = self.map.sum(2)
        explored_gt[explored_gt > 1] = 1.0

        return agent_view_cropped, map_gt, agent_view_explored, explored_gt, map_se

    def get_st_pose(self, current_loc):
        loc = [- (current_loc[0] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               - (current_loc[1] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               90 - np.rad2deg(current_loc[2])]
        return loc

    def reset_map(self, map_size, object_len):
        self.map_size_cm = map_size
        self.semantic_map_len = object_len

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

        self.semantic_map = np.zeros((self.map.shape[0], 
                                    self.map.shape[1],     
                                    self.semantic_map_len), dtype=np.float32)

    def get_map(self):
        return self.map
