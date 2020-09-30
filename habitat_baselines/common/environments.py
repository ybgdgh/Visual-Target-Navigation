#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
import math
import habitat
# from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry

# NavSLAMRLEnv
import os
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat import Config, Env, RLEnv, Dataset

from collections import Counter
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

from habitat_baselines.common.map_utils.supervision import HabitatMaps
import habitat_baselines.common.map_utils.pose as pu
import habitat_baselines.common.map_utils.visualizations as vu
from habitat_baselines.common.map_utils.model import get_grid
from habitat_baselines.common.map_utils.map_builder import MapBuilder
from habitat_baselines.common.map_utils.fmm_planner import FMMPlanner


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    print("env_name :", env_name)
    print("env_type :", baseline_registry.get_env(env_name))
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        self.dataset = dataset

        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]

        self.scene = self.habitat_env.sim.semantic_annotations()
        self.object_len = len(self.scene.objects)

        for obj in self.scene.objects:
            if obj is not None:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()}, index:{obj.category.index()}"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )

        if "semantic" in observations:
            observations["semantic"] = self._preprocess_semantic(observations["semantic"])
        # print("category_to_task_category_id: ", self.dataset.category_to_task_category_id)
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        obs, rew, done, info = super().step(*args, **kwargs)

        if "semantic" in obs:
            obs["semantic"] = self._preprocess_semantic(obs["semantic"])
        return obs, rew, done, info

    def _preprocess_semantic(self, semantic):
        # print("*********semantic type: ", type(semantic))
        se = list(set(semantic.ravel()))
        # print(se) # []
        for i in range(len(se)):
            if self.scene.objects[se[i]] is not None and self.scene.objects[se[i]].category.name() in self.dataset.category_to_task_category_id:
                # print(self.scene.objects[se[i]].id) 
                # print(self.scene.objects[se[i]].category.index()) 
                # print(type(self.scene.objects[se[i]].category.index()) ) # int
                semantic[semantic==se[i]] = self.dataset.category_to_task_category_id[self.scene.objects[se[i]].category.name()]
                # print(self.scene.objects[se[i]+1].category.name())
            else :
                semantic[
                    semantic==se[i]
                    ] = 0
        semantic = semantic.astype(np.uint8)
        se = list(set(semantic.ravel()))
        # print("semantic: ", se) # []

        return semantic

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()



@baseline_registry.register_env(name="NavSLAMRLEnv")
class NavSLAMRLEnv(NavRLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):

        self.rank = config.NUM_PROCESSES

        self.print_images = 1

        self.figure, self.ax = plt.subplots(1,3, figsize=(6*16/9, 6),
                                                facecolor="whitesmoke",
                                                num="Thread {}".format(self.rank))
                                                
        self.episode_no = 0

        self.env_frame_width = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
        self.env_frame_height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT
        self.hfov = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV
        self.map_resolution = config.RL.SLAMDDPPO.map_resolution
        self.map_size_cm = config.RL.SLAMDDPPO.map_size_cm
        self.agent_min_z = config.RL.SLAMDDPPO.agent_min_z
        self.agent_max_z = config.RL.SLAMDDPPO.agent_max_z
        self.camera_height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION[1]
        self.agent_view_angle = config.RL.SLAMDDPPO.agent_view_angle
        self.du_scale = config.RL.SLAMDDPPO.du_scale
        self.vision_range = config.RL.SLAMDDPPO.vision_range
        self.visualize = config.RL.SLAMDDPPO.visualize
        self.obs_threshold = config.RL.SLAMDDPPO.obs_threshold
        self.obstacle_boundary = config.RL.SLAMDDPPO.obstacle_boundary

        self.collision_threshold = config.RL.SLAMDDPPO.collision_threshold
        self.vis_type = config.RL.SLAMDDPPO.vis_type # '1: Show predicted map, 2: Show GT map'

        # self.res = transforms.Compose([transforms.ToPILImage(),
        #             transforms.Resize((self.frame_height, self.frame_width),
        #                               interpolation = Image.NEAREST)])

        # 创建地图
        self.mapper = self.build_mapper()
        self.full_map_size = self.map_size_cm//self.map_resolution

        super().__init__(config, dataset)

    
    def save_position(self):
        self.agent_state = self._env.sim.get_agent_state()
        self.trajectory_states.append([self.agent_state.position,
                                       self.agent_state.rotation])


    def reset(self):
        self.episode_no += 1
        self.timestep = 0
        self._previous_action = None
        self.trajectory_states = []
        # print("reset")
        # start_time = time.clock()
        # Get Ground Truth Map
        self.explorable_map = None
        obs = super().reset()
        # t1 = time.clock()
        # deta_t = str(t1 - start_time)
        # print("reset time: ", deta_t)

        # while self.explorable_map is None:
        #     self.explorable_map = self._get_gt_map(self.full_map_size)

            # t2 = time.clock()
            # print("depth time: ", t2 - t1)
        rgb = obs['rgb'].astype(np.uint8)
        self.obs = rgb # For visualization
        depth = _preprocess_depth(obs['depth'])
        semantic = obs['semantic']
        self.semantic = semantic
        self.object_ind = obs["objectgoal"]
        self.object_name = list(self.dataset.category_to_task_category_id.keys())[list(self.dataset.category_to_task_category_id.values()).index(self.object_ind)]
        # print(self.object_name)

        # se = list(set(semantic.ravel()))
        # print(se)

        # scene = self.habitat_env.sim.semantic_annotations()
        # self.object_len = len(scene.objects)
        # print("object number: ", self.object_len)
        

        # Initialize map and pose
        self.mapper.reset_map(self.map_size_cm, 21)
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
        fp_proj, self.map, fp_explored, self.explored_map, self.semantic_map = \
            self.mapper.update_map(depth, semantic, mapper_gt_pose)
        
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

        # obs["semantic_map"] = self.semantic_map # 480*480*21
        # obs["collusion_map"] = self.map # 480*480
        # obs["explored_map"] = self.explored_map
        map_copy = self.map.copy()
        explored_map_copy = self.explored_map.copy()
        input_map = map_copy[:,:,np.newaxis]
        input_explored_map = explored_map_copy[:,:,np.newaxis]
        # print("semantic: ", self.semantic_map.shape, self.map.shape)
        map_sum = np.concatenate([input_map, self.semantic_map, input_explored_map], axis=2) 
        obs["map_sum"] = map_sum.astype(np.uint8)
        obs["curr_pose"] = np.array(
                            [(self.curr_loc_gt[1]*100 / self.map_resolution),
                            (self.curr_loc_gt[0]*100 / self.map_resolution)]).astype(np.float32)
        return obs


    def build_mapper(self):
        params = {}
        params['frame_width'] = self.env_frame_width
        params['frame_height'] = self.env_frame_height
        params['fov'] =  self.hfov
        params['resolution'] = self.map_resolution
        params['map_size_cm'] = self.map_size_cm
        params['agent_min_z'] = self.agent_min_z
        params['agent_max_z'] = self.agent_max_z
        params['agent_height'] = self.camera_height * 100
        params['agent_view_angle'] = self.agent_view_angle
        params['du_scale'] = self.du_scale
        params['vision_range'] = self.vision_range
        params['visualize'] = self.visualize
        params['obs_threshold'] = self.obs_threshold
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


    def step(self, *args, **kwargs):

        self.timestep += 1

        self.last_loc = np.copy(self.curr_loc)
        self.last_loc_gt = np.copy(self.curr_loc_gt)

        self._previous_action = kwargs["action"]

        obs, rew, done, info = super().step(*args, **kwargs)

        # # Preprocess observations
        rgb = obs['rgb'].astype(np.uint8)
        self.obs = rgb # For visualization
        # if self.frame_width != self.env_frame_width:
        #     rgb = np.asarray(self.res(rgb))

        # state = rgb.transpose(2, 0, 1)

        depth = _preprocess_depth(obs['depth'])
        semantic = obs['semantic']
        self.semantic = semantic
        # self.object_ind = obs["objectgoal"]
        # print("object_ind: ", self.object_ind)
        # se = list(set(semantic.ravel()))
        # print(se)
        
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
        fp_proj, self.map, fp_explored, self.explored_map, self.semantic_map = \
                self.mapper.update_map(depth, semantic, mapper_gt_pose)

        # print("semantic count: ", Counter(semantic.ravel()))
        # for i in range(self.semantic_map.shape[2]):
        #     se_map = list(set(np.array(self.semantic_map[:,:,i].ravel())))
        #     if len(se_map) > 1:
        #         print("self.semantic_map: ", i, Counter(np.array(self.semantic_map[:,:,i].ravel()))) # []

        # print("self._previous_action", self._previous_action)
        if self._previous_action["action"] == 1:
            x1, y1, t1 = self.last_loc_gt
            x2, y2, t2 = self.curr_loc_gt
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
                        # print("collision map: ", r, c)

        # Set info
        self.info['time'] = self.timestep
        self.info['fp_proj'] = fp_proj
        self.info['fp_explored']= fp_explored

        self.save_position()

        map_copy = self.map.copy()
        explored_map_copy = self.explored_map.copy()
        input_map = map_copy[:,:,np.newaxis]
        input_explored_map = explored_map_copy[:,:,np.newaxis]
        # print("semantic: ", self.semantic_map.shape, self.map.shape)
        map_sum = np.concatenate([input_map, input_explored_map, self.semantic_map], axis=2) 
        # print("semantic: ", type(map_sum[0]))
        obs["map_sum"] = map_sum.astype(np.uint8)
        obs["curr_pose"] = np.array(
                            [(self.curr_loc_gt[1]*100.0 / self.map_resolution),
                            (self.curr_loc_gt[0]*100.0 / self.map_resolution)]).astype(np.float32)

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

    def get_local_actions(self, global_goal):
                    
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
        
        planning_window = [0, 480, 0, 480]

        # global_goal=kwargs["goal"]
        # print("global_goals type: ", type(global_goal))
        # print("global_goals type: ", global_goal)

        # semantic map goal
        # goal_list=[]
        Find_flag = False
        object_map = self.semantic_map[:,:,self.object_ind[0]-1]
        if len(object_map[object_map!=0]) > 5:
            print("map_objectid: ", self.object_ind[0],
                "num: ", len(object_map[object_map!=0]))
            goal_list = np.array(np.array(object_map.nonzero()).T)
            # print("goal_list: ", goal_list.shape)

            goal_err = np.abs(goal_list - start_gt).sum(1)
            # print("goal_err: ", goal_err)

            index = np.argmin(goal_err, axis=0)
            # print("index: ", index)

            global_goal = torch.from_numpy(goal_list[index])

            Find_flag = True
            print("global_goal: ", global_goal)
            

        goal = pu.threshold_poses(global_goal, self.map.shape)
        # print("self.map: ", self.map.shape)
        # print("self.explored_map: ", self.explored_map.shape)
        # print("start_gt: ", start_gt)
        # print("goal: ", type(goal))

        # Get short-term goal
        stg, replan = self._get_stg(self.map, self.explored_map, start_gt, np.copy(goal), planning_window)

        # print("stg: ", stg)

        # Find GT action
        gt_action = self._get_gt_action(1 - self.explored_map, 
                                        start_gt,
                                        [int(stg[0]), int(stg[1])],
                                        np.copy(goal),
                                        planning_window, start_o_gt, 
                                        Find_flag, replan)
        
        
        dump_dir = "habitat_baselines/dump"
        ep_dir = '{}/episodes/{}/{}/'.format(
                            dump_dir, self.rank+1, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        vis_grid = vu.get_colored_map(self.map,
                        self.collison_map,
                        self.visited_gt,
                        self.visited_gt,
                        goal.int(),
                        self.explored_map,
                        self.explorable_map,
                        self.map*self.explored_map,
                        self.semantic_map)
        vis_grid = np.flipud(vis_grid)
        vu.visualize(self.figure, self.ax, self.obs, self.semantic, vis_grid[:,:,::-1],
                    (start_x_gt, start_y_gt, start_o_gt),
                    (start_x_gt, start_y_gt, start_o_gt),
                    dump_dir, self.rank, self.episode_no,
                    self.timestep, self.visualize,
                    self.print_images, self.object_name, gt_action)

        return gt_action

    

    def _get_stg(self, grid, explored, start, goal, planning_window):

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(20., dist)
        x1 = max(1, int(x1 - buf))
        x2 = min(grid.shape[0]-1, int(x2 + buf))
        y1 = max(1, int(y1 - buf))
        y2 = min(grid.shape[1]-1, int(y2 + buf))

        rows = explored.sum(1)
        rows[rows>0] = 1
        ex1 = np.argmax(rows)
        ex2 = len(rows) - np.argmax(np.flip(rows))

        cols = explored.sum(0)
        cols[cols>0] = 1
        ey1 = np.argmax(cols)
        ey2 = len(cols) - np.argmax(np.flip(cols))

        ex1 = min(int(start[0]) - 2, ex1)
        ex2 = max(int(start[0]) + 2, ex2)
        ey1 = min(int(start[1]) - 2, ey1)
        ey2 = max(int(start[1]) + 2, ey2)

        x1 = max(x1, ex1)
        x2 = min(x2, ex2)
        y1 = max(y1, ey1)
        y2 = min(y2, ey2)

        # print("grid: ", grid.shape) # 480*480*1
        # print(gx1, gx2, gy1, gy2, x1, x2, y1, y2)

        traversible = skimage.morphology.binary_dilation(
                        grid[x1:x2, y1:y2],
                        self.selem) != True
        traversible[self.collison_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                    int(start[1]-y1)-1:int(start[1]-y1)+2] = 1
        # print("traversible: ", traversible.shape)

        if goal[0]-2 > x1 and goal[0]+3 < x2\
            and goal[1]-2 > y1 and goal[1]+3 < y2:
            traversible[int(goal[0]-x1)-2:int(goal[0]-x1)+3,
                    int(goal[1]-y1)-2:int(goal[1]-y1)+3] = 1
        else:
            goal[0] = min(max(x1, goal[0]), x2)
            goal[1] = min(max(y1, goal[1]), y2)

        def add_boundary(mat):
            h, w = mat.shape
            new_mat = np.ones((h+2,w+2))
            new_mat[1:h+1,1:w+1] = mat
            return new_mat

        traversible = add_boundary(traversible)

        planner = FMMPlanner(traversible, 360//10)

        reachable = planner.set_goal([goal[1]-y1+1, goal[0]-x1+1])

        stg_x, stg_y = start[0] - x1 + 1, start[1] - y1 + 1
        for i in range(1):
            stg_x, stg_y, replan = planner.get_short_term_goal([stg_x, stg_y])
        if replan:
            stg_x, stg_y = start[0], start[1]
        else:
            stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), replan


    def _get_gt_action(self, grid, start, goal, g_goal, planning_window, start_o, Find_flag, replan):

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(5., dist)
        x1 = max(0, int(x1 - buf))
        x2 = min(grid.shape[0], int(x2 + buf))
        y1 = max(0, int(y1 - buf))
        y2 = min(grid.shape[1], int(y2 + buf))
        # print("grid: ", grid.shape)
        # print(gx1, gx2, gy1, gy2, x1, x2, y1, y2)
        path_found = False
        goal_r = 0
        while not path_found:
            traversible = skimage.morphology.binary_dilation(
                            grid[gx1:gx2, gy1:gy2][x1:x2, y1:y2],
                            self.selem) != True
            # print(grid[gx1:gx2, gy1:gy2].shape, grid[gx1:gx2, gy1:gy2][x1:x2, y1:y2].shape)
            traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
            traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                        int(start[1]-y1)-1:int(start[1]-y1)+2] = 1
            # print("traversible: ", traversible.shape)
            traversible[int(goal[0]-x1)-goal_r:int(goal[0]-x1)+goal_r+1,
                        int(goal[1]-y1)-goal_r:int(goal[1]-y1)+goal_r+1] = 1
            scale = 1
            planner = FMMPlanner(traversible, 360//10, scale)
            # print("traversible: ", traversible.shape)
            reachable = planner.set_goal([goal[1]-y1, goal[0]-x1])

            stg_x_gt, stg_y_gt = start[0] - x1, start[1] - y1
            for i in range(1):
                stg_x_gt, stg_y_gt, replan = \
                        planner.get_short_term_goal([stg_x_gt, stg_y_gt])

            if replan and buf < 100.:
                buf = 2*buf
                x1 = max(0, int(x1 - buf))
                x2 = min(grid.shape[0], int(x2 + buf))
                y1 = max(0, int(y1 - buf))
                y2 = min(grid.shape[1], int(y2 + buf))
            elif replan and goal_r < 50:
                goal_r += 1
            else:
                path_found = True

        stg_x_gt, stg_y_gt = stg_x_gt + x1, stg_y_gt + y1
        angle_st_goal = math.degrees(math.atan2(stg_x_gt - start[0],
                                                stg_y_gt - start[1]))
        angle_agent = (start_o)%360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal)%360.0
        if relative_angle > 180:
            relative_angle -= 360

        g_dist = pu.get_l2_distance(g_goal[0], start[0], g_goal[1], start[1])

        if Find_flag:
            print("distance: ", g_dist)
        if (g_dist < 6.0 and Find_flag) or replan:
            gt_action = 0

        elif relative_angle > 15.:
            gt_action = 3
        elif relative_angle < -15.:
            gt_action = 2
        else:
            gt_action = 1

        return gt_action



def _preprocess_depth(depth):
    # print("depth: ", depth)
    # print("depth: ", depth.shape) # 256*256*1
    depth = depth[:, :, 0]*1
    mask2 = depth > 0.99
    depth[mask2] = 0.

    for i in range(depth.shape[1]):
        depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

    mask1 = depth == 0
    depth[mask1] = np.NaN
    depth = depth*1000.
    return depth

