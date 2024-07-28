##!/usr/bin/env python3
# @File: stop_sign.py
# @Description: Check if the vehicle is affected by a stop sign.
# @CreatedTime: 2024/07/26
# @Author: Yueyuan Li, PDM-Lite Team

import carla

from src.common_carla.bounding_box import is_point_in_bbox
from src.common_carla.geometry import dot_product, get_vector_length


class StopSignDetector:
    """Check if the vehicle is affected by a stop sign.

    Adopted from:
    - https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/team_code/birds_eye_view/run_stop_sign.py

    """

    def __init__(
        self,
        carla_world,
        proximity_threshold=50.0,
        speed_threshold=0.1,
        waypoint_step=0.25,
    ):
        self.proximity_threshold = proximity_threshold
        self.speed_threshold = speed_threshold
        self.waypoint_step = waypoint_step

        self.world_map = carla_world.get_map()

        all_actors = carla_world.get_actors()
        self.stop_sign_list = []
        for actor in all_actors:
            if "traffic.stop" in actor.type_id:
                self.stop_sign_list.append(actor)

        self.target_stop_sign = None
        self.is_stopped = False
        self.is_affected = False

    def is_near_stop_sign(self, vehicle, stop_sign, multi_step=80):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        vehicle_location = vehicle.get_transform().location
        stop_sign_transform = stop_sign.get_transform()
        stop_sign_location = stop_sign_transform.location

        # A fast and coarse check
        if stop_sign_location.distance(vehicle_location) > self.proximity_threshold:
            return affected

        # A slower and accurate check based on waypoint's horizon and geometric test
        trigger_volume_transformed = stop_sign_transform.transform(
            stop_sign.trigger_volume.location
        )

        # Get the list of waypoints ahead of the vehicle
        location_list = [vehicle_location]
        wp = self.world_map.get_waypoint(vehicle_location)
        for _ in range(multi_step):
            next_wps = wp.next(self.waypoint_step)
            if not next_wps:
                break
            wp = next_wps[0]
            if not wp:
                break

            location_list.append(wp.transform.location)

        # Check if the any of the actor wps is inside the stop's bounding box.
        # Using more than one waypoint removes issues with small trigger volumes and backwards movement
        stop_sign_extent = stop_sign.trigger_volume.extent
        stop_sign_extent.x = max(0.5, stop_sign_extent.x)
        stop_sign_extent.y = max(0.5, stop_sign_extent.y)

        for location in location_list:
            if is_point_in_bbox(location, trigger_volume_transformed, stop_sign_extent):
                affected = True
                break

        return affected

    def scan_stop_sign(self, vehicle):
        """Find the stop sign that affects the vehicle."""
        target_stop_sign = None
        vehicle_transform = vehicle.get_transform()
        vehicle_direction = vehicle_transform.get_forward_vector()

        wp = self.world_map.get_waypoint(vehicle_transform.location)
        wp_direction = wp.transform.get_forward_vector()

        dot_vehicle_wp = dot_product(wp_direction, vehicle_direction)

        if dot_vehicle_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in self.stop_sign_list:
                if self.is_near_stop_sign(vehicle, stop_sign):
                    # this stop sign is affecting the vehicle
                    target_stop_sign = stop_sign
                    break

        return target_stop_sign

    def tick(self, vehicle):
        vehicle_location = vehicle.get_location()

        if self.target_stop_sign is None:
            # Find a stop sign that affects the vehicle
            self.target_stop_sign = self.scan_stop_sign(vehicle)
        else:
            if not self.is_stopped:
                # When the vehicle is affected by the stop sign, check if the vehicle has stopped.
                current_speed = get_vector_length(
                    vehicle.get_velocity(), carla.Vector2D
                )
                stop_sign_location = self.target_stop_sign.get_transform().transform(
                    self.target_stop_sign.trigger_volume.location
                )
                if (
                    current_speed < self.speed_threshold
                    and stop_sign_location.distance(vehicle_location) < 4
                ):
                    self.is_stopped = True

            if not self.is_affected:
                stop_sign_transform = self.target_stop_sign.get_transform()
                trigger_volume_transformed = stop_sign_transform.transform(
                    self.target_stop_sign.trigger_volume.location
                )

                # Check if the any of the actor wps is inside the stop's bounding box.
                # Using more than one waypoint removes issues with small trigger volumes and backwards movement
                stop_sign_extent = self.target_stop_sign.trigger_volume.extent
                stop_sign_extent.x = max(0.5, stop_sign_extent.x)
                stop_sign_extent.y = max(0.5, stop_sign_extent.y)

                if is_point_in_bbox(
                    vehicle_location, trigger_volume_transformed, stop_sign_extent
                ):
                    self.is_affected = True

            if not self.is_near_stop_sign(vehicle, self.target_stop_sign):
                # Check if the vehicle is still affected by the stop sign.
                # If not, reset the stop sign.
                self.target_stop_sign = None
                self.is_stopped = False
                self.is_affected = False
