##!/usr/bin/env python3
# @File: actor.py
# @Description: Utility functions for actors in CARLA.
# @CreatedTime: 2024/07/26
# @Author: Yueyuan Li, PDM-Lite Team

import logging

import carla
import numpy as np
from agents.navigation.local_planner import LocalPlanner
from agents.tools.misc import compute_distance, is_within_distance
from shapely.geometry import Polygon

from src.common_carla.route import get_route_polygon


def get_forward_speed(actor):
    """
    Calculate the forward speed of the actor.

    Returns:
        float: The forward speed of the vehicle in m/s.
    """

    velocity = actor.get_velocity()
    transform = actor.get_transform()

    velocity_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)

    vector_direction = np.array(
        [
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ]
    )

    forward_speed = np.dot(velocity_np, vector_direction)

    return forward_speed


def get_horizontal_distance(actor1: carla.Actor, actor2: carla.Actor):
    """
    Calculate the horizontal distance between two actors (ignoring the z-coordinate).

    Returns:
        float: The horizontal distance between the two actors.
    """
    location1 = actor1.get_location()
    location2 = actor2.get_location()

    # Compute the distance vector (ignoring the z-coordinate)
    distance = carla.Vector3D(
        location1.x - location2.x, location1.y - location2.y, 0
    ).length()

    return distance


def get_nearby_objects(ego_actor, all_actors, search_radius):
    """
    Find actors, who's trigger boxes are within a specified radius around the ego vehicle.

    Args:
        ego_actor (carla.Actor): The position of the ego actor.
        all_actors (list): A list of all actors.
        search_radius (float): The radius (in meters) around the ego vehicle to search for nearby actors.

    Returns:
        list: A list of actors within the specified search radius.
    """
    nearby_objects = []

    for actor in all_actors:
        try:
            trigger_box_global_pos = actor.get_transform().transform(
                actor.trigger_volume.location
            )
        except:
            logging.error(
                "Warning! Error caught in get_nearby_objects. (probably AttributeError: actor.trigger_volume)"
            )
            logging.error("Skipping this object.")
            continue

        # Convert the vector to a carla.Location for distance calculation
        trigger_box_global_pos = carla.Location(
            x=trigger_box_global_pos.x,
            y=trigger_box_global_pos.y,
            z=trigger_box_global_pos.z,
        )

        # Check if the actor's trigger volume is within the search radius
        if trigger_box_global_pos.distance(ego_actor.get_location()) < search_radius:
            nearby_objects.append(actor)

    return nearby_objects


def detect_obstacle(
    ego_actor,
    world_map,
    planner: LocalPlanner,
    list_actor: list,
    max_distance: float,
    angle_range=[0, 90],
    lane_offset: float = 0.0,
    use_bbox: bool = False,
):
    ego_transform = ego_actor.get_transform()
    ego_location = ego_transform.location
    ego_wp = world_map.get_waypoint(ego_location, lane_tpe=carla.LaneType.Any)

    # Get the right offset
    if ego_wp.lane_id < 0 and lane_offset != 0:
        lane_offset = -lane_offset

    # Get the transform of the front of the ego
    ego_front_transform = ego_transform
    ego_front_transform.location += carla.Location(
        ego_actor.bounding_box.extent.x * ego_transform.get_forward_vector(),
    )

    opposite_invasion = (
        ego_actor.bounding_box.extent.y + abs(lane_offset) > ego_wp.lane_width / 2
    )
    use_bbox = use_bbox or opposite_invasion or ego_wp.is_junction

    # Get the route bounding box
    route_polygon = get_route_polygon(planner, max_distance, offset=lane_offset)

    for target_actor in list_actor:
        if target_actor.id == ego_actor.id:
            continue

        target_transform = target_actor.get_transform()
        if target_transform.location.distance(ego_location) > max_distance:
            continue
        target_wp = world_map.get_waypoint(
            target_transform.location, lane_type=carla.LaneType.Any
        )

        # General approach for junctions and vehicles invading other lanes due to the offset

        if use_bbox or target_wp.is_junction:
            target_bbox = target_actor.bounding_box
            target_vtx = target_bbox.get_world_vertices(target_transform)
            list_target = [[v.x, v.y, v.z] for v in target_vtx]
            target_polygon = Polygon(list_target)

            if route_polygon.intersects(target_polygon):
                distance_to_obstacle = compute_distance(
                    target_actor.get_location(), ego_location
                )
                return True, target_actor.id, distance_to_obstacle

        # Simplified approach, using only the plan waypoints (similar to TM)
        else:
            if (
                target_wp.road_id != ego_wp.road_id
                or target_wp.lane_id != ego_wp.lane_id
            ):
                next_wp = planner.get_incoming_waypoint_and_direction(steps=3)[0]
                if not next_wp:
                    continue
                if (
                    target_wp.road_id != next_wp.road_id
                    or target_wp.lane_id != next_wp.lane_id
                ):
                    continue

            target_forward_vector = target_transform.get_forward_vector()
            target_extent = target_actor.bounding_box.extent.x
            target_rear_transform = target_transform
            target_rear_transform.location -= carla.Location(
                x=target_extent * target_forward_vector.x,
                y=target_extent * target_forward_vector.y,
            )

            if is_within_distance(
                target_rear_transform, ego_front_transform, max_distance, angle_range
            ):
                distance_to_obstacle = compute_distance(
                    target_actor.get_location(), ego_location
                )
                return True, target_actor.id, distance_to_obstacle

    return False, None, -1
