##!/usr/bin/env python3
# @File: geometry.py
# @Description: Utility functions for geometry operations.
# @CreatedTime: 2024/07/09
# @Author: Yueyuan Li

import math

import numpy as np


def normalize_angle(angle):
    """Normalize angle to be in the range (-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def normalize_degree(angle):
    """Normalize angle to be in the range (-180, 180]"""
    return (angle + 180) % 360 - 180


def get_relative_position(pose_origin, pose_to_transform):
    """Transform a global position to a local position.

    Args:
        pose_origin: A 4x4 matrix records the rotation and coordinates of the origin of the local coordinate system.
        pose_to_transform: A 4x4 matrix records the rotation and coordinates of the global position to be transformed.
    """
    relative_position = pose_to_transform[:3, 3] - pose_origin[:3, 3]
    rotation = pose_origin[:3, :3].T
    relative_position = rotation @ relative_position

    return relative_position


def get_transform_2D(data: np.ndarray, translation: np.ndarray, yaw: float):
    """Transform 2D points into a new coordinate system.

    Args:
        data (np.ndarray): 2D points (N,2)
        translation (np.ndarray): Translation in meters (2,)
        yaw (float): Yaw angle in radians
    """

    rotation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ]
    )

    transformed_data = (data - translation) @ rotation_matrix

    return transformed_data


def get_transform_3D(data: np.ndarray, translation: np.ndarray, yaw: float):
    """Transform 3D points into a new coordinate system.

    Args:
        data (np.ndarray): 3D points (N,3)
        translation (np.ndarray): Translation in meters (3,)
        yaw (float): Yaw angle in radians
    """

    rotation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    transformed_data = (data - translation) @ rotation_matrix

    return transformed_data


def preprocess_compass(compass):
    """Checks the compass for Nans and rotates it into the default CARLA coordinate
    system with range [-pi,pi].

    Args:
        compass (float): Compass value provided by the IMU, in radian
    """
    if math.isnan(compass):  # simulation bug
        compass = 0.0

    # The minus 90.0 degree is because the compass sensor uses a different
    # coordinate system then CARLA. Check the coordinate_sytems.txt file
    compass = normalize_angle(compass - np.deg2rad(90.0))

    return compass
