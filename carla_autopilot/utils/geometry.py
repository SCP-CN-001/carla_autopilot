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


def get_world_to_ego_to_image_coords(
    world_coords, ego_transform, camera_transform, fov, img_size
):
    """
    Transform world coordinates to image coordinates.

    Args:
        world_coords (np.ndarray): World coordinates (N,3)
        ego_transform (dict): Ego vehicle transform (x, y, z, roll, pitch, yaw)
        camera_transform (dict): Camera transform (x, y, z, roll, pitch, yaw)
        fov (float): Field of view in degrees
        img_size (tuple): Image size (width, height)

    Returns:
        np.ndarray: Image coordinates (N,2)
    """

    def rotation_matrix(roll, pitch, yaw):
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )

        R_y = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )

        R_z = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        # R = R_z @ R_x @ R_y
        R = R_z @ R_y @ R_x
        return R

    def precompute_camera_parameters(camera_transform, fov, img_size):
        """
        Precompute the rotation matrix, translation vector, and intrinsic matrix of the camera.

        Args:
            camera_transform (dict): Camera transform (x, y, z, roll, pitch, yaw)
            fov (float): Field of view in degrees
            img_size (tuple): Image size (width, height)

        Returns:
            np.ndarray: Camera rotation matrix (3,3)
            np.ndarray: Camera translation vector (3,1)
            np.ndarray: Intrinsic matrix (3,3)
        """
        # the rotation matrix of the camera
        camera_R = rotation_matrix(
            camera_transform["roll"], camera_transform["pitch"], camera_transform["yaw"]
        )

        # the translation vector of the camera
        # noticed that the z coordinate is negated
        camera_T = np.array(
            [camera_transform["x"], camera_transform["y"], -camera_transform["z"]]
        ).reshape((3, 1))

        # the intrinsic matrix
        fov_rad = np.deg2rad(fov)
        fx = img_size[0] / (2 * np.tan(fov_rad / 2))
        fy = fx
        cx = img_size[0] / 2
        cy = img_size[1] / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        return camera_R, camera_T, K

    def compute_world_to_camera_matrix(ego_transform, camera_R, camera_T):
        """
        Compute the transformation matrix from world to camera coordinates.

        Args:
            ego_transform (dict): Ego vehicle transform (x, y, z, roll, pitch, yaw)
            camera_R (np.ndarray): Camera rotation matrix (3,3)
            camera_T (np.ndarray): Camera translation vector (3,1)

        Returns:
            np.ndarray: Transformation matrix from world to camera coordinates (4,4)
            np.ndarray: Transformation matrix from world to ego vehicle coordinates (4,4)
        """
        ego_R = rotation_matrix(
            ego_transform["roll"], ego_transform["pitch"], ego_transform["yaw"]
        )

        ego_T = np.array(
            [ego_transform["x"], ego_transform["y"], ego_transform["z"]]
        ).reshape((3, 1))

        world_to_ego = np.eye(4)
        world_to_ego[:3, :3] = ego_R.T
        world_to_ego[:3, 3] = (-ego_R.T @ ego_T).flatten()

        ego_to_camera = np.eye(4)
        ego_to_camera[:3, :3] = camera_R.T
        ego_to_camera[:3, 3] = (-camera_R.T @ camera_T).flatten()

        world_to_camera = ego_to_camera @ world_to_ego

        return world_to_camera

    def project_waypoints_to_image(waypoints, world_to_camera, K):
        """
        Project waypoints to image coordinates.

        Args:
            waypoints (np.ndarray): Waypoints in world coordinates (N,3)
            world_to_camera (np.ndarray): Transformation matrix from world to camera coordinates (4,4)
            K (np.ndarray): Intrinsic matrix (3,3)

        Returns:
            np.ndarray: Image coordinates (N,2)
        """

        num_points = waypoints.shape[0]
        waypoints_h = np.hstack((waypoints, np.ones((num_points, 1)))).T  # 4xN

        waypoints_camera = world_to_camera @ waypoints_h  # 4xN

        # Remove points behind the camera
        # in_front = waypoints_camera[2, :] > 0
        # waypoints_camera = waypoints_camera[:, in_front]

        # change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # the z coordinate is not negated below so that we do not need to flip the image later
        waypoints_camera = np.array(
            [
                waypoints_camera[1, :],
                waypoints_camera[2, :],
                waypoints_camera[0, :],
                waypoints_camera[3, :],
            ]
        )

        # Project to image coordinates
        projections = K @ waypoints_camera[:3, :]  # 3xN
        projections /= projections[2, :]
        projected_points = projections[:2, :].T  # Nx2

        return projected_points

    # Compute camera transform
    camera_R, camera_T, K = precompute_camera_parameters(
        camera_transform, fov, img_size
    )

    # Compute world to camera matrix
    world_to_camera = compute_world_to_camera_matrix(ego_transform, camera_R, camera_T)

    # Project waypoints to image coordinates
    image_coords = project_waypoints_to_image(world_coords, world_to_camera, K)

    return image_coords
