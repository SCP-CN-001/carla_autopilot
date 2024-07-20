##!/usr/bin/env python3
# @File: route_planner.py
# @Description: This file implements the class used for data generation for the high level commands and the target waypoints.
# @CreatedTime: 2024/07/12
# @Author: Yueyuan Li, PDM-Lite

import math
import warnings
import xml.etree.ElementTree as ET
from collections import deque
from copy import deepcopy

import carla
import numpy as np
from agents.navigation.global_route_planner import GlobalRoutePlanner


class RoutePlanner:
    """
    Gets the next waypoint along a path.

    Adapted from: https://github.com/SCP-CN-001/carla_autopilot/blob/main/src/planners/route_planner.py.

    Only improve the code style and remove the unused code.
    """

    def __init__(self, configs):
        self.saved_route = deque()
        self.route = deque()
        self.saved_route_distances = deque()
        self.route_distances = deque()

        self.min_distance = configs.min_distance
        self.max_distance = configs.max_distance
        self.is_last = False

        self.mean = np.array(configs.mean)
        self.scale = np.array(configs.scale)

    def set_route(self, global_plan, gps=False, carla_map=None):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                warnings.warn("deprecated", DeprecationWarning)
                # GPS uses a different coordinate system than CARLA.
                # This converts from GPS -> CARLA (90° rotation)
                pos = np.array([pos["lat"], pos["lon"], pos["z"]])
                pos = (pos - self.mean) * self.scale
            else:
                # important to use the z variable, otherwise there are some rare bugs at
                # carla.map.get_waypoint(carla.Location) and the wrong wp is returned
                pos = np.array([pos.location.x, pos.location.y, pos.location.z])
                pos -= self.mean

            self.route.append((pos, cmd))

        if carla_map is not None:
            for _ in range(50):
                loc = carla.Location(
                    x=self.route[-1][0][0],
                    y=self.route[-1][0][1],
                    z=self.route[-1][0][2],
                )
                next_loc = carla_map.get_waypoint(loc).next(1)[0].transform.location
                next_loc = np.array([next_loc.x, next_loc.y, next_loc.z])
                self.route.append((next_loc, self.route[-1][1]))

        # We do the calculations in the beginning once so that we don't have
        # to do them every time in run_step
        self.route_distances.append(0.0)
        for i in range(1, len(self.route)):
            diff = self.route[i][0] - self.route[i - 1][0]
            distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5
            self.route_distances.append(distance)

    def run_step(self, gps):
        if len(self.route) <= 2:
            self.is_last = True
            return self.route

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += self.route_distances[i]

            diff = self.route[i][0] - gps
            distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5

            if farthest_in_range < distance <= self.min_distance:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
                self.route_distances.popleft()

        return self.route

    def save(self):
        # because self.route saves objects of traffic lights and traffic signs a deep copy is not possible
        self.saved_route = []
        for (
            loc,
            cmd,
            d_traffic,
            traffic,
            d_stop,
            stop,
            speed_limit,
            corrected_speed_limit,
        ) in self.route:
            self.saved_route.append(
                (
                    np.copy(loc),
                    cmd,
                    d_traffic,
                    traffic,
                    d_stop,
                    stop,
                    speed_limit,
                    corrected_speed_limit,
                )
            )

        self.saved_route = deque(self.saved_route)
        self.saved_route_distances = deepcopy(self.route_distances)

    def load(self):
        self.route = self.saved_route
        self.route_distances = self.saved_route_distances
        self.is_last = False
        self.route = self.saved_route
        self.route_distances = self.saved_route_distances
        self.is_last = False


def interpolate_trajectory(
    world_map, waypoints_trajectory, hop_resolution=1.0, max_len=400
):
    """
    Given some raw keypoints interpolate a full dense trajectory to be used by the user.

    Args:
        world: A reference to the CARLA world so we can use the planner
        waypoints_trajectory: The current coarse trajectory
        hop_resolution: The resolution, how dense is the provided trajectory going to be made.

    Returns:
        The full interpolated route both in GPS coordinates and also in its original form.
    """

    grp = GlobalRoutePlanner(world_map, hop_resolution)
    # Obtain route plan
    lat_ref, lon_ref = _get_latlon_ref(world_map)

    route = []
    gps_route = []

    for i in range(len(waypoints_trajectory) - 1):
        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        if waypoint.x != waypoint_next.x or waypoint.y != waypoint_next.y:
            interpolated_trace = grp.trace_route(waypoint, waypoint_next)
            if len(interpolated_trace) > max_len:
                waypoints_trajectory[i + 1] = waypoints_trajectory[i]
            else:
                for wp, connection in interpolated_trace:
                    route.append((wp.transform, connection))
                    gps_coord = _location_to_gps(
                        lat_ref, lon_ref, wp.transform.location
                    )
                    gps_route.append((gps_coord, connection))

    return gps_route, route


def extrapolate_waypoint_route(waypoint_route, route_points):
    # guard against inplace mutation
    route = deepcopy(waypoint_route)

    # determine length of route before extrapolation
    remaining_waypoints = len(route)

    # we start at the end of the unextrapolated route and move linearly
    heading_vector = route[-1][0] - route[-2][0]
    heading_vector = heading_vector / (np.linalg.norm(heading_vector) + 1.0e-7)

    # we extrapolate 2 meters ahead for each point and skip the first two
    extrapolation = []
    for i in range(2, route_points + 2):
        next_wp = route[-1][0] + i * 2 * heading_vector
        extrapolation.append((next_wp, route[-1][1]))
    route.extend(extrapolation)

    # the waypoint_planner does not pop the last (few) waypoints in its
    # route. We manually pop those when they are the only remaining points
    # in the original route and only pass the extrapolation to the cost.
    if remaining_waypoints == 2:
        route.popleft()
        route.popleft()
    elif remaining_waypoints == 1:
        route.popleft()
    return route


def location_route_to_gps(route, lat_ref, lon_ref):
    """
    Locate each waypoint of the route into gps, (lat long ) representations.
    """
    gps_route = []

    for transform, connection in route:
        gps_point = _location_to_gps(lat_ref, lon_ref, transform.location)
        gps_route.append((gps_point, connection))

    return gps_route


def _get_latlon_ref(world_map):
    """
    Convert from waypoints world coordinates to CARLA GPS coordinates.

    Returns:
        tuple: The latitude and longitude reference for the current map.
    """
    xodr = world_map.to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    # default reference
    lat_ref = 42.0
    lon_ref = 2.0

    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for geo_ref in header.iter("geoReference"):
                if geo_ref.text:
                    str_list = geo_ref.text.split(" ")
                    for item in str_list:
                        if "+lat_0" in item:
                            lat_ref = float(item.split("=")[1])
                        if "+lon_0" in item:
                            lon_ref = float(item.split("=")[1])
    return lat_ref, lon_ref


def _location_to_gps(lat_ref, lon_ref, location):
    """
    Convert from world coordinates to GPS coordinates

    Args:
        lat_ref: Latitude reference for the current map
        lon_ref: Longitude reference for the current map
        location: Location to translate

    Returns:
        dict: A dictionary with lat, lon and height
    """

    EARTH_RADIUS_EQUA = 6378137.0
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = (
        scale
        * EARTH_RADIUS_EQUA
        * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    )
    mx += location.x
    my -= location.y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    z = location.z

    return {"lat": lat, "lon": lon, "z": z}
