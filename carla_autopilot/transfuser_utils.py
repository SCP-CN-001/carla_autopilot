"""
Some utility functions e.g. for normalizing angles
Functions for detecting red lights are adapted from scenario runners
atomic_criteria.py
"""
import itertools
import math
from collections import deque
from copy import deepcopy

import carla
import cv2
import numpy as np
import shapely
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon
from torch import nn


def rotate_point(point, angle):
    """
    rotate a given point by a given angle
    """
    x_ = (
        math.cos(math.radians(angle)) * point.x
        - math.sin(math.radians(angle)) * point.y
    )
    y_ = (
        math.sin(math.radians(angle)) * point.x
        + math.cos(math.radians(angle)) * point.y
    )
    return carla.Vector3D(x_, y_, point.z)


def get_traffic_light_waypoints(traffic_light, carla_map):
    """
    get area of a given traffic light
    """
    base_transform = traffic_light.get_transform()
    base_loc = traffic_light.get_location()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)

    # Discretize the trigger box into points
    area_ext = traffic_light.trigger_volume.extent
    x_values = np.arange(
        -0.9 * area_ext.x, 0.9 * area_ext.x, 1.0
    )  # 0.9 to avoid crossing to adjacent lanes

    area = []
    for x in x_values:
        point = rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
        point_location = area_loc + carla.Location(x=point.x, y=point.y)
        area.append(point_location)

    # Get the waypoints of these points, removing duplicates
    ini_wps = []
    for pt in area:
        wpx = carla_map.get_waypoint(pt)
        # As x_values are arranged in order, only the last one has to be checked
        if (
            not ini_wps
            or ini_wps[-1].road_id != wpx.road_id
            or ini_wps[-1].lane_id != wpx.lane_id
        ):
            ini_wps.append(wpx)

    # Advance them until the intersection
    wps = []
    eu_wps = []
    for wpx in ini_wps:
        distance_to_light = base_loc.distance(wpx.transform.location)
        eu_wps.append(wpx)
        next_distance_to_light = distance_to_light + 1.0
        while not wpx.is_intersection:
            next_wp = wpx.next(0.5)[0]
            next_distance_to_light = base_loc.distance(next_wp.transform.location)
            if (
                next_wp
                and not next_wp.is_intersection
                and next_distance_to_light <= distance_to_light
            ):
                eu_wps.append(next_wp)
                distance_to_light = next_distance_to_light
                wpx = next_wp
            else:
                break

        if not next_distance_to_light <= distance_to_light and len(eu_wps) >= 4:
            wps.append(eu_wps[-4])
        else:
            wps.append(wpx)

    return area_loc, wps


class CarlaActorDummy:
    """
    Actor dummy structure used to simulate a CARLA actor for data augmentation
    """

    world = None
    bounding_box = None
    transform = None
    id = None

    def __init__(
        self, world, bounding_box, transform, id
    ):  # pylint: disable=locally-disabled, redefined-builtin
        self.world = world
        self.bounding_box = bounding_box
        self.transform = transform
        self.id = id

    def get_world(self):
        return self.world

    def get_transform(self):
        return self.transform

    def get_bounding_box(self):
        return self.bounding_box
