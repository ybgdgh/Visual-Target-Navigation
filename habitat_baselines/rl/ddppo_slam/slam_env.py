#!/usr/bin/env python3
import os

import habitat
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat import Config, Env, RLEnv, Dataset
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.environments import NavRLEnv

import quaternion
import skimage.morphology
import numpy as np
from torchvision import transforms
import torch
from torch.nn import functional as F
from PIL import Image

import matplotlib

import time

import matplotlib.pyplot as plt

from habitat_baselines.rl.ddppo_slam.map_utils.supervision import HabitatMaps
import habitat_baselines.rl.ddppo_slam.map_utils.pose as pu
import habitat_baselines.rl.ddppo_slam.map_utils.visualizations as vu
from habitat_baselines.rl.ddppo_slam.map_utils.model import get_grid
from habitat_baselines.rl.ddppo_slam.map_utils.map_builder import MapBuilder

def _preprocess_depth(depth):
    depth = depth[:, :, 0]*1
    mask2 = depth > 0.99
    depth[mask2] = 0.

    for i in range(depth.shape[1]):
        depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

    mask1 = depth == 0
    depth[mask1] = np.NaN
    depth = depth*1000.
    return depth



@baseline_registry.register_env(name="NavSLAMRLEnv")
class NavSLAMRLEnv(habitat.RLEnv):
    def __init__(self, config: Config):

        self.mapper = self.build_mapper()
        self.rank = config.NUM_PROCESSES

        self.print_images = 1

        self.figure, self.ax = plt.subplots(1,2, figsize=(6*16/9, 6),
                                                facecolor="whitesmoke",
                                                num="Thread {}".format(self.rank))
                                                
        self.episode_no = 0

        self.env_frame_width = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
        self.env_frame_height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT
        self.map_resolution = 5
        self.map_size_cm = 2400
        self.camera_height = 0.88
        self.du_scale = 2
        self.vision_range = 64
        self.visualize = 1
        self.obs_threshold = 1
        self.obstacle_boundary = 5

        self.collision_threshold = 0.20

        self.frame_width = 256
        self.frame_height = 256

        self.vis_type = 2 # '1: Show predicted map, 2: Show GT map'


        self.res = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((self.frame_height, self.frame_width),
                                      interpolation = Image.NEAREST)])

        super().__init__(config)

    
    def save_position(self):
        self.agent_state = self._env.sim.get_agent_state()
        self.trajectory_states.append([self.agent_state.position,
                                       self.agent_state.rotation])


    def reset(self):
        self.episode_no += 1
        self.timestep = 0
        self._previous_action = None
        self.trajectory_states = []

        # start_time = time.clock()
        # Get Ground Truth Map
        self.explorable_map = None
        obs = super().reset()
        # t1 = time.clock()
        # deta_t = str(t1 - start_time)
        # print("reset time: ", deta_t)

        # while self.explorable_map is None:
        full_map_size = self.map_size_cm//self.map_resolution
            # self.explorable_map = self._get_gt_map(full_map_size)

            # t2 = time.clock()
            # print("depth time: ", t2 - t1)


        # Preprocess observations
        # rgb = obs['rgb'].astype(np.uint8)
        # self.obs = rgb # For visualization
        # if self.frame_width != self.env_frame_width:
        #     rgb = np.asarray(self.res(rgb))
        # state = rgb.transpose(2, 0, 1)
        depth = _preprocess_depth(obs['depth'])

        

        # Initialize map and pose
        self.mapper.reset_map(self.map_size_cm)
        self.curr_loc = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.]
        self.curr_loc_gt = self.curr_loc
        self.last_loc_gt = self.curr_loc_gt
        self.last_loc = self.curr_loc
        self.last_sim_location = self.get_sim_location()


        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))

        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = \
            self.mapper.update_map(depth, mapper_gt_pose)
        

        # Initialize variables
        self.scene_name = self.habitat_env.sim.config.SCENE
        self.visited = np.zeros(self.map.shape)
        self.visited_vis = np.zeros(self.map.shape)
        self.visited_gt = np.zeros(self.map.shape)
        self.collison_map = np.zeros(self.map.shape)
        self.col_width = 1

        # Set info
        self.info = {
            'time': self.timestep,
            'fp_proj': fp_proj,
            'fp_explored': fp_explored,
            'sensor_pose': [0., 0., 0.],
            'pose_err': [0., 0., 0.],
        }

        self.save_position()

        return obs, self.info


    def build_mapper(self):
        params = {}
        params['frame_width'] = 256
        params['frame_height'] = 256
        params['fov'] =  79
        params['resolution'] = 5
        params['map_size_cm'] = 1200
        params['agent_min_z'] = 25
        params['agent_max_z'] = 150
        params['agent_height'] = 0.88 * 100
        params['agent_view_angle'] = 0
        params['du_scale'] = 2
        params['vision_range'] = 64
        params['visualize'] = 0
        params['obs_threshold'] = 1
        self.selem = skimage.morphology.disk(5 /
                                             5)
        mapper = MapBuilder(params)
        return mapper



    def _get_gt_map(self, full_map_size):
        self.scene_name = self.habitat_env.sim.config.SCENE
        logger.error('Computing map for %s', self.scene_name)

        # Get map in habitat simulator coordinates
        self.map_obj = HabitatMaps(self.habitat_env)
        if self.map_obj.size[0] < 1 or self.map_obj.size[1] < 1:
            logger.error("Invalid map: {}/{}".format(
                            self.scene_name, self.episode_no))
            return None

        agent_y = self._env.sim.get_agent_state().position.tolist()[1]*100.
        sim_map = self.map_obj.get_map(agent_y, -50., 50.0)

        sim_map[sim_map > 0] = 1.

        # Transform the map to align with the agent
        min_x, min_y = self.map_obj.origin/100.0
        x, y, o = self.get_sim_location()
        x, y = -x - min_x, -y - min_y
        range_x, range_y = self.map_obj.max/100. - self.map_obj.origin/100.

        map_size = sim_map.shape
        scale = 2.
        grid_size = int(scale*max(map_size))
        grid_map = np.zeros((grid_size, grid_size))

        grid_map[(grid_size - map_size[0])//2:
                 (grid_size - map_size[0])//2 + map_size[0],
                 (grid_size - map_size[1])//2:
                 (grid_size - map_size[1])//2 + map_size[1]] = sim_map

        if map_size[0] > map_size[1]:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale) \
                             * map_size[1] * 1. / map_size[0],
                    (y - range_y/2.) * 2. / (range_y * scale),
                    180.0 + np.rad2deg(o)
                ]])

        else:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale),
                    (y - range_y/2.) * 2. / (range_y * scale) \
                            * map_size[0] * 1. / map_size[1],
                    180.0 + np.rad2deg(o)
                ]])

        rot_mat, trans_mat = get_grid(st, (1, 1,
            grid_size, grid_size), torch.device("cpu"))

        grid_map = torch.from_numpy(grid_map).float()
        grid_map = grid_map.unsqueeze(0).unsqueeze(0)
        translated = F.grid_sample(grid_map, trans_mat)
        rotated = F.grid_sample(translated, rot_mat)

        episode_map = torch.zeros((full_map_size, full_map_size)).float()
        if full_map_size > grid_size:
            episode_map[(full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size,
                        (full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size] = \
                                rotated[0,0]
        else:
            episode_map = rotated[0,0,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size]



        episode_map = episode_map.numpy()
        episode_map[episode_map > 0] = 1.

        return episode_map


    def step(self, action):

        self.timestep += 1

        self.last_loc = np.copy(self.curr_loc)
        self.last_loc_gt = np.copy(self.curr_loc_gt)

        self._previous_action = action

        obs, rew, done, info = super().step(action)

        # # Preprocess observations
        rgb = obs['rgb'].astype(np.uint8)
        self.obs = rgb # For visualization
        # if self.frame_width != self.env_frame_width:
        #     rgb = np.asarray(self.res(rgb))

        # state = rgb.transpose(2, 0, 1)

        depth = _preprocess_depth(obs['depth'])

        # print("*********** obs['depth']: ", obs['depth'])
        # print("*********** depth: ", depth)

        # Get base sensor and ground-truth pose
        dx_gt, dy_gt, do_gt = self.get_gt_pose_change()

        self.curr_loc_gt = pu.get_new_pose(self.curr_loc_gt,
                               (dx_gt, dy_gt, do_gt))


        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))

        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = \
                self.mapper.update_map(depth, mapper_gt_pose)

        x1, y1, t1 = self.last_loc
        x2, y2, t2 = self.curr_loc
        if abs(x1 - x2)< 0.05 and abs(y1 - y2) < 0.05:
            self.col_width += 2
            self.col_width = min(self.col_width, 9)
        else:
            self.col_width = 1

        dist = pu.get_l2_distance(x1, x2, y1, y2)
        if dist < self.collision_threshold: #Collision
            length = 2
            width = self.col_width
            buf = 3
            for i in range(length):
                for j in range(width):
                    wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + \
                                    (j-width//2) * np.sin(np.deg2rad(t1)))
                    wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - \
                                    (j-width//2) * np.cos(np.deg2rad(t1)))
                    r, c = wy, wx
                    r, c = int(r*100/self.map_resolution), \
                            int(c*100/self.map_resolution)
                    [r, c] = pu.threshold_poses([r, c],
                                self.collison_map.shape)
                    self.collison_map[r,c] = 1

        # Set info
        self.info['time'] = self.timestep
        self.info['fp_proj'] = fp_proj
        self.info['fp_explored']= fp_explored

        self.save_position()

        return obs, rew, done, info


    def get_sim_location(self):
        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o


    def get_gt_pose_change(self):
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do


    def map_visualize(self):
        
        # Get last loc ground truth pose
        last_start_x, last_start_y = self.last_loc_gt[0], self.last_loc_gt[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0/self.map_resolution),
                      int(c * 100.0/self.map_resolution)]
        last_start = pu.threshold_poses(last_start, self.visited_gt.shape)

        # Get ground truth pose
        start_x_gt, start_y_gt, start_o_gt = self.curr_loc_gt
        r, c = start_y_gt, start_x_gt
        start_gt = [int(r * 100.0/self.map_resolution),
                    int(c * 100.0/self.map_resolution)]
        start_gt = pu.threshold_poses(start_gt, self.visited_gt.shape)

        # dump_dir = "mytask/dump"
        # ep_dir = '{}/episodes/{}/{}/'.format(
        #                     dump_dir, self.rank+1, self.episode_no)
        # if not os.path.exists(ep_dir):
        #     os.makedirs(ep_dir)

        vis_grid = vu.get_colored_map(self.map,
                        self.collison_map,
                        self.visited_gt,
                        self.visited_gt,
                        (int(last_start_x), int(last_start_y)),
                        self.explored_map,
                        self.explorable_map,
                        self.map*self.explored_map)
        vis_grid = np.flipud(vis_grid)
        # vu.visualize(self.figure, self.ax, self.obs, vis_grid[:,:,::-1],
        #             (start_x_gt, start_y_gt, start_o_gt),
        #             (start_x_gt, start_y_gt, start_o_gt),
        #             dump_dir, self.rank, self.episode_no,
        #             self.timestep, self.visualize,
        #             self.print_images, self.vis_type)

        return vis_grid[:,:,::-1]


