##!/usr/bin/env python3
# @File: route.py
# @Description: Utility functions for route in CARLA.
# @CreatedTime: 2024/07/26
# @Author: Yueyuan Li, PDM-Lite Team

import carla
import numpy as np
from agents.navigation.local_planner import RoadOption
from shapely.geometry import Polygon
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard_custom.scenarios.cheater import Cheater


def get_route_polygon(planner, max_distance, offset=0.0) -> Polygon:
    point_list = []

    ego_vehicle = CarlaDataProvider.get_hero_actor()
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location

    extent_y = ego_vehicle.bounding_box.extent.y
    right_extent = extent_y + offset
    left_extent = -extent_y + offset
    right_vector = ego_transform.get_right_vector()

    right_location = carla.Location(
        right_extent * right_vector.x, right_extent * right_vector.y
    )
    left_location = carla.Location(
        left_extent * right_vector.x, left_extent * right_vector.y
    )
    pt1 = ego_location + right_location
    pt2 = ego_location + left_location

    point_list.append([[pt1.x, pt1.y, pt1.z], [pt2.x, pt2.y, pt2.z]])

    for waypoint, _ in planner.get_plan():
        if ego_location.distance(waypoint.transform.location) > max_distance:
            break

        right_vector = waypoint.transform.get_right_vector()
        pt1 = waypoint.transform.location + right_location
        pt2 = waypoint.transform.location + left_location
        point_list.append([[pt1.x, pt1.y, pt1.z], [pt2.x, pt2.y, pt2.z]])

    # Two points don't create a polygon, nothing to check
    if len(point_list) < 2:
        return None

    return Polygon(point_list)


def sort_scenarios_by_distance(ego_location):
    """
    Sorts the active scenarios based on the distance from the ego vehicle.
    """
    distances = []

    # Calculate the distance of each scenario's first actor from the ego vehicle
    for _, scenario_data in Cheater.active_scenarios:
        first_actor = scenario_data[0]
        distances.append(ego_location.distance(first_actor.get_location()))

    # Sort the scenarios based on the calculated distances
    indices = np.argsort(distances)
    Cheater.active_scenarios = [Cheater.active_scenarios[i] for i in indices]


def compute_min_time_for_distance(
    distance,
    target_speed,
    ego_speed,
    time_step=0.05,
    params_min_time_cover_distance=[0.00904221, 0.00733342, -0.03744807, 0.0235038],
):
    """
    Computes the minimum time the ego vehicle needs to travel a given distance.

    Args:
        distance (float): The distance to be traveled.
        target_speed (float): The target speed of the ego vehicle.
        ego_speed (float): The current speed of the ego vehicle.
        time_step (float): The time step to be used for the computation.
        params_min_time_cover_distance (list): Parameters to calculate how much the ego agent needs to cover a given distance. Default values are taken from the kinematic bicycle model

    Returns:
        float: The minimum time needed to travel the given distance.
    """
    min_time_needed = 0.0
    remaining_distance = distance
    current_speed = ego_speed

    # Iterate over time steps until the distance is covered
    while True:
        # Takes less than a tick to cover remaining_distance with current_speed
        if remaining_distance - current_speed * time_step < 0:
            break

        remaining_distance -= current_speed * time_step
        min_time_needed += time_step

        # Values from kinematic bicycle model
        normalized_speed = current_speed / 120.0
        speed_change = np.clip(
            params_min_time_cover_distance[0]
            + params_min_time_cover_distance[1] * normalized_speed
            + params_min_time_cover_distance[2] * normalized_speed**2
            + params_min_time_cover_distance[3] * normalized_speed**3,
            0.0,
            np.inf,
        )
        current_speed = np.clip(
            120 * (normalized_speed + speed_change), 0, target_speed
        )

    # Add remaining time at the current speed
    min_time_needed += remaining_distance / current_speed

    return min_time_needed


def get_previous_road_lane_ids(start_wp, distance_retrieve_id):
    """
    Retrieves the previous road and lane IDs for a given starting waypoint.

    Args:
        start_wp (carla.Waypoint): The starting waypoint.
        distance_retrieve_id (int): Distance to check for road_id/lane_id for RouteObstacle scenarios

    Returns:
        list: A list of tuples containing road IDs and lane IDs.
    """
    current_wp = start_wp
    previous_road_lane_ids = [(current_wp.road_id, current_wp.lane_id)]

    # Traverse backwards up to 100 waypoints to find previous lane IDs
    for _ in range(distance_retrieve_id):
        previous_wps = current_wp.previous(1)

        # Check if the road ends and no previous route waypoints exist
        if len(previous_wps) == 0:
            break
        current_wp = previous_wps[0]

        if (
            current_wp.road_id,
            current_wp.lane_id,
        ) not in previous_road_lane_ids:
            previous_road_lane_ids.append((current_wp.road_id, current_wp.lane_id))

    return previous_road_lane_ids


def is_near_lane_change(
    vehicle,
    route_points,
    route_index,
    commands,
    safety_distance=10.0,
    min_point_lookahead=200,
    min_point_previous=150,
    points_per_meter=10,
):
    """
    Check if the vehicle is near a lane change maneuver.

    Args:
        vehicle (_type_): _description_
        route_points (np.ndarray): An array of locations representing the planned route.
        safety_distance (float): Safety distance to be added to emergency braking distance.
        min_point_lookahead (int): Minimum number of points to look ahead when checking for lane change.
        min_point_previous (int): Minimum number of previous points to consider when checking for lane change.
        points_per_meter (int): Points sampled per meter when interpolating route.

    Returns:
        bool: True if the vehicle is near a lane change maneuver, False otherwise.
    """
    vehicle_velocity = vehicle.get_velocity().length()

    # Calculate the braking distance based on the ego velocity
    braking_distance = (((vehicle_velocity * 3.6) / 10.0) ** 2 / 2.0) + safety_distance

    # Determine the number of waypoints to look ahead based on the braking distance
    look_ahead_points = max(
        min_point_lookahead,
        min(route_points.shape[0], points_per_meter * int(braking_distance)),
    )

    from_index = max(0, route_index - min_point_previous)
    to_index = min(len(commands) - 1, route_index + look_ahead_points)
    # Iterate over the points around the current position, checking for lane change commands
    for i in range(from_index, to_index, 1):
        if commands[i] in (
            RoadOption.CHANGELANELEFT,
            RoadOption.CHANGELANERIGHT,
        ):
            return True

    return False


def is_overtaking_path_clear(
    world_map,
    ego_vehicle,
    vehicle_list,
    target_speed,
    route_points,
    from_index,
    to_index,
    previous_road_lane_ids,
    safety_distance_check_path_free,
    safety_time_check_path_free,
    min_speed=50.0 / 3.6,
):
    """
    Checks if the path between two route indices is clear for the ego vehicle to overtake.

    Args:
        vehicle_list (list): A list of all vehicles in the simulation.
        target_speed (float): The target speed of the ego vehicle.
        from_index (int): The starting route index.
        to_index (int): The ending route index.
        previous_road_lane_ids (list): A list of tuples containing previous road IDs and lane IDs.
        min_speed (float, optional): The minimum speed to consider for overtaking. Defaults to 50/3.6 km/h.

    Returns:
        bool: True if the path is clear for overtaking, False otherwise.
    """
    ego_location = ego_vehicle.get_location()
    ego_bbox = ego_vehicle.bounding_box
    ego_speed = ego_vehicle.get_velocity().length()

    # 10 m safety distance, overtake with max. 50 km/h
    from_location = route_points[from_index]
    from_location = carla.Location(from_location[0], from_location[1], from_location[2])

    to_location = route_points[to_index]
    to_location = carla.Location(to_location[0], to_location[1], to_location[2])

    # Compute the distance and time needed for the ego vehicle to overtake
    ego_distance = (
        to_location.distance(ego_location)
        + ego_bbox.extent.x * 2
        + safety_distance_check_path_free
    )

    ego_time = compute_min_time_for_distance(
        ego_distance, min(min_speed, target_speed), ego_speed
    )

    path_clear = True
    for vehicle in vehicle_list:
        # Sort out ego vehicle
        if vehicle.id == ego_vehicle.id:
            continue

        vehicle_location = vehicle.get_location()
        vehicle_wp = world_map.get_waypoint(vehicle_location)

        # Check if the vehicle is on the previous lane IDs
        if (vehicle_wp.road_id, vehicle_wp.lane_id) in previous_road_lane_ids:
            diff_vector = vehicle_location - ego_location
            dot_product = (
                ego_vehicle.get_transform().get_forward_vector().dot(diff_vector)
            )
            # Skip if the vehicle is not relevant, because its not on the overtaking path and behind
            # the ego vehicle
            if dot_product < 0:
                continue

            diff_vector_2 = to_location - vehicle_location
            dot_product_2 = (
                vehicle.get_transform().get_forward_vector().dot(diff_vector_2)
            )
            # The overtaking path is blocked by vehicle
            if dot_product_2 < 0:
                path_clear = False
                break

            other_vehicle_distance = (
                to_location.distance(vehicle_location) - vehicle.bounding_box.extent.x
            )
            other_vehicle_time = other_vehicle_distance / max(
                1.0, vehicle.get_velocity().length()
            )

            # Add 200 ms safety margin
            # Vehicle needs less time to arrive at to_location than the ego vehicle
            if other_vehicle_time < ego_time + safety_time_check_path_free:
                path_clear = False
                break

    return path_clear
