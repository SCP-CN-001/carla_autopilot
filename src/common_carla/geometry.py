##!/usr/bin/env python3
# @File: geometry.py
# @Description: Include some tool function that operate geometric computation with carla instances.
# @CreatedTime: 2024/07/15
# @Author: Yueyuan Li

import carla
import numpy as np


def dot_product(vector1, vector2, vector_class=carla.Vector3D):
    """Calculate the dot product of two vectors.

    Returns:
            float: The dot product of the two vectors.
    """
    if vector_class == carla.Vector3D:
        return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z
    elif vector_class == carla.Vector2D:
        return vector1.x * vector2.x + vector1.y * vector2.y

    raise NotImplementedError("Only support carla.Vector3D and carla.Vector2D.")


def cross_product(vector1, vector2, vector_class=carla.Vector3D):
    """Calculate the cross product of two vectors.

    Returns:
            carla.Vector3D: The cross product of the two vectors.
    """
    if vector_class == carla.Vector3D:
        x = vector1.y * vector2.z - vector1.z * vector2.y
        y = vector1.z * vector2.x - vector1.x * vector2.z
        z = vector1.x * vector2.y - vector1.y * vector2.x

        return carla.Vector3D(x=x, y=y, z=z)
    elif vector_class == carla.Vector2D:
        return vector1.x * vector2.y - vector1.y * vector2.x

    raise NotImplementedError("Only support carla.Vector3D and carla.Vector2D.")


def get_vector_length(vector, vector_class: carla.Vector3D) -> float:
    """Calculate the length of a vector.

    Returns:
        float: The length of the vector.
    """
    if vector_class == carla.Vector3D:
        return np.sqrt(vector.x**2 + vector.y**2 + vector.z**2)
    elif vector_class == carla.Vector2D:
        return np.sqrt(vector.x**2 + vector.y**2)

    raise NotImplementedError("Only support carla.Vector3D and carla.Vector2D.")


def get_rotation_matrix(carla_rotation: carla.Rotation) -> np.ndarray:
    """Transform rpy in carla.Rotation to rotation matrix in np.array"""
    roll = np.deg2rad(carla_rotation.roll)
    pitch = np.deg2rad(carla_rotation.pitch)
    yaw = np.deg2rad(carla_rotation.yaw)

    yaw_matrix = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    pitch_matrix = np.array(
        [
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    roll_matrix = np.array(
        [[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]]
    )

    rotation_matrix = yaw_matrix.dot(pitch_matrix).dot(roll_matrix)
    return rotation_matrix


def vector_global_to_local(
    vector: carla.Vector3D, carla_rotation: carla.Rotation
) -> carla.Vector3D:
    """Transform a carla.Vector3D to local coordinate system.

    Args:
        vector (carla.Vector3D): Vector in global coordinate (world, actor)
        carla_rotation (carla.Rotation): Rotation in global coordinate (world, actor)

    Returns:
        carla.Vector3D: Vector in local coordinate.
    """
    rotation = get_rotation_matrix(carla_rotation)
    vector_np = np.array([[vector.x], [vector.y], [vector.z]])
    vector_local_np = rotation.T.dot(vector_np)
    vector_local = carla.Vector3D(
        x=vector_local_np[0, 0], y=vector_local_np[1, 0], z=vector_local_np[2, 0]
    )

    return vector_local


def location_global_to_local(
    location: carla.Location, transform: carla.Transform
) -> carla.Location:
    """Transform a carla.Location to local coordinate system.

    Args:
        location (carla.Location): Location in global coordinate (world, actor)
        transform (carla.Transform): Transform in global coordinate (world, actor)

    Returns:
        carla.Location: Location in local coordinate.
    """
    vector_global = carla.Vector3D(
        x=location.x - transform.x,
        y=location.y - transform.y,
        z=location.z - transform.z,
    )
    vector_local = vector_global_to_local(vector_global, transform.rotation)
    location_local = carla.Location(
        x=vector_local.x, y=vector_local.y, z=vector_local.z
    )

    return location_local
