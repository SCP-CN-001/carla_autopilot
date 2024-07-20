##!/usr/bin/env python3
# @File: geometry.py
# @Description: Utility functions for geometry operations.
# @CreatedTime: 2024/07/09
# @Author: Yueyuan Li


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


def get_transform_2D(data: np.ndarray, translation: np.ndarray, angle: float):
    """Transform 2D points into a new coordinate system.

    Args:
        data (np.ndarray): 2D points (N,2)
        translation (np.ndarray): Translation in meters (2,)
        angle (float): Angle in radians
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rotation_matrix = np.array(
        [
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle],
        ]
    )

    transformed_data = (data - translation) @ rotation_matrix

    return transformed_data


def get_transform_3D(data: np.ndarray, translation: np.ndarray, angle: float):
    """Transform 3D points into a new coordinate system.

    Args:
        data (np.ndarray): 3D points (N,3)
        translation (np.ndarray): Translation in meters (3,)
        angle (float): Angle in radians
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rotation_matrix = np.array(
        [
            [cos_angle, -sin_angle, 0.0],
            [sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    transformed_data = (data - translation) @ rotation_matrix

    return transformed_data


def get_angle_by_position(current_position, current_angle, target_position):
    """
    Calculate the angle (in radians) from the current position and heading to a target position.

    Args:
        current_position (list): A list of (x, y) coordinates representing the current position.
        current_angle (float): The current heading angle in radians.
        target_position (tuple or list): A tuple or list of (x, y) coordinates representing the target position.

    Returns:
        float: The angle (in radians) from the current position and heading to the target position.
    """
    cos_angle = np.cos(current_angle)
    sin_angle = np.sin(current_angle)

    # Calculate the vector from the current position to the target position
    delta_position = target_position - current_position

    # Calculate the dot product of the position delta vector and the current heading vector
    target_x = cos_angle * delta_position[0] + sin_angle * delta_position[1]
    target_y = -sin_angle * delta_position[0] + cos_angle * delta_position[1]

    # Calculate the angle (in radians) from the current heading to the target position
    angle_radians = -np.arctan2(-target_y, target_x)

    return angle_radians
