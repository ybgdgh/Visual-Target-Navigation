# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for processing depth images.
"""
from argparse import Namespace

import numpy as np

import habitat_baselines.common.map_utils.rotation_utils as ru


def get_camera_matrix(width, height, fov):
    """Returns a camera matrix from image size and fov."""
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
    camera_matrix = {'xc': xc, 'zc': zc, 'f': f}
    camera_matrix = Namespace(**camera_matrix)
    return camera_matrix


def get_point_cloud_from_z(Y, camera_matrix, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    x, z = np.meshgrid(np.arange(Y.shape[-1]),
                       np.arange(Y.shape[-2] - 1, -1, -1))

    # print("Y :", Y.shape) # 256*256
    # print("x: ", x)
    # print("z: ", z)
    for i in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)

    # print("x: ", x)
    # print("x: ", x.shape) # 256*256
    # print("z: ", z)
    # print("z: ", z.shape)
    X = (x[::scale, ::scale] - camera_matrix.xc) * Y[::scale, ::scale] / camera_matrix.f
    Z = (z[::scale, ::scale] - camera_matrix.zc) * Y[::scale, ::scale] / camera_matrix.f
    

    # print("x: ", x[::scale, ::scale])
    # print("x: ", x[::scale, ::scale].shape) # 18*128
    # print("X: ", X.shape) # 128*128
    # print("Z: ", Z.shape) # 128*128
    XYZ = np.concatenate((X[..., np.newaxis], Y[::scale, ::scale][..., np.newaxis],
                          Z[..., np.newaxis]), axis=X.ndim)

    # print("XYZ: ", XYZ)
    # print("XYZ: ", XYZ.shape) # 128*128*3
    # print("XYZS: ", XYZS) # 
    # print("XYZS: ", XYZS.shape) # 128*128*4
    return XYZ


def transform_camera_view(XYZ, sensor_height, camera_elevation_degree):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([1., 0., 0.], angle=np.deg2rad(camera_elevation_degree))
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ


def transform_pose(XYZ, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([0., 0., 1.], angle=current_pose[2] - np.pi / 2.)
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[:, :, 0] = XYZ[:, :, 0] + current_pose[0]
    XYZ[:, :, 1] = XYZ[:, :, 1] + current_pose[1]
    return XYZ


def bin_points(XYZ_cms, map_size, z_bins, xy_resolution):
    """Bins points into xy-z bins
    XYZ_cms is ... x H x W x3
    Outputs is ... x map_size x map_size x (len(z_bins)+1)
    """
    sh = XYZ_cms.shape
    # print("XYZ_cms: ",XYZ_cms.shape) #128*128*3
    XYZ_cms = XYZ_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
    # print("XYZ_cms: ",XYZ_cms)
    # print("XYZ_cms: ",XYZ_cms.shape) #1*128*128*3
    # print("z_bins: ",z_bins) #1*128*128*3

    n_z_bins = len(z_bins) + 1
    counts = []
    isvalid = []
    for XYZ_cm in XYZ_cms:
        # print("XYZ_cm:", XYZ_cm.shape) # 128*128*3
        isnotnan = np.logical_not(np.isnan(XYZ_cm[:, :, 0]))
        X_bin = np.round(XYZ_cm[:, :, 0] / xy_resolution).astype(np.int32)
        Y_bin = np.round(XYZ_cm[:, :, 1] / xy_resolution).astype(np.int32)
        Z_bin = np.digitize(XYZ_cm[:, :, 2], bins=z_bins).astype(np.int32) # 三层，在z_bins区间中的为1，否则为0或2

        # print("Z_bin: {}".format(Z_bin))
        # print("Z_bin: {}".format(Z_bin.shape))

        isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0, Y_bin < map_size,
                            Z_bin >= 0, Z_bin < n_z_bins, isnotnan])
        isvalid = np.all(isvalid, axis=0)

        # print("isvalid: ", isvalid)
        # print("isvalid: ", isvalid.shape) #128*128

        ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin 
        ind[np.logical_not(isvalid)] = 0
        # print("ind: ", ind)
        # print("ind: ", ind.shape) # 128*128
        count = np.bincount(ind.ravel(), isvalid.ravel().astype(np.int32),
                            minlength=map_size * map_size * n_z_bins)
        # print("counts: ", count.shape) # 691200

        counts = np.reshape(count, [map_size, map_size, n_z_bins])
        # print("counts: ", counts.shape) # 480*480*3


    counts = counts.reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])
    # print("counts: ", counts.shape) # 480*480*3
    # print("counts: ", counts) # 480*480*3

    return counts



def bin_semantic_points(XYZ_cms, semantic, map_size, semantic_map_len, xy_resolution):
    """Bins points into xy-z bins
    XYZ_cms is ... x H x W x3
    semantic is ... x H x W
    Outputs is ... x map_size x map_size x len
    """
    # print("XYZ_cms: ",XYZ_cms.shape) #128*128*3
    # XYZ_cms = XYZ_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
    # print("XYZ_cms: ",XYZ_cms)
    # print("XYZ_cms: ",XYZ_cms.shape) #1*128*128*3
    # print("z_bins: ",z_bins) 


    # print("XYZ_cm:", XYZ_cm.shape) # 128*128*3
    isnotnan = np.logical_not(np.isnan(XYZ_cms[:, :, 0]))
    X_bin = np.round(XYZ_cms[:, :, 0] / xy_resolution).astype(np.int32)
    Y_bin = np.round(XYZ_cms[:, :, 1] / xy_resolution).astype(np.int32)
    Z_bin = semantic.astype(np.int32)

    # print("Z_bin: {}".format(X_bin))
    # print("X_bin: {}".format(X_bin.shape)) #128*128
    # print("Z_bin: {}".format(Z_bin.shape)) #128*128

    isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0, Y_bin < map_size,
                        Z_bin > 0, Z_bin < semantic_map_len, isnotnan])
    isvalid = np.all(isvalid, axis=0)

    # print("isvalid: ", isvalid)
    # print("isvalid: ", isvalid.shape) #128*128

    ind = (Y_bin * map_size + X_bin) * semantic_map_len + Z_bin 
    ind[np.logical_not(isvalid)] = 0
    # print("ind: ", ind)
    # print("ind: ", ind.shape) # 128*128
    count = np.bincount(ind.ravel(), isvalid.ravel().astype(np.int32),
                        minlength=map_size * map_size * semantic_map_len)
    # print("counts: ", count.shape) # 11980800

    counts = np.reshape(count, [map_size, map_size, semantic_map_len])
    # print("counts: ", counts.shape) # 480*480*semantic_map_len

    return counts
