##!/usr/bin/env python3
# @File: expert_agent.py
# @Description: An expert agent for data collection in CARLA Leaderboard 2.0.
# @CreatedTime: 2024/07/08
# @Author: Yueyuan Li, PDM-Lite


import logging
from collections import deque

import carla
import numpy as np
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from scipy.integrate import RK45
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from src.common_carla.traffic_light import get_before_traffic_light_waypoints
from src.controllers import LateralPIDController, LongitudinalLinearRegressionController
from src.planners.privileged_route_planner import PrivilegedRoutePlanner
from src.planners.route_planner import RoutePlanner
from src.utils.kinematic_bicycle_model import KinematicBicycleModel


def get_entry_point():
    return "ExpertAgent"


class ExpertAgent(AutonomousAgent):
    """An expert agent for data collection in CARLA Leaderboard 2.0. This agent has access to the ground truth in the simulator directly.

    Adopted from:
    """

    def setup(self, configs):
        self.configs = configs
        self.record = configs.record
        self.track = configs.track
        self.save_path = configs.save_path
        self.route_index = configs.route_index
        self.generate_data = configs.generate_data
        self.make_histogram = configs.make_histogram
        self.tp_stats = configs.tp_stats
        self.visualize = configs.visualize

        # Dynamics models
        self.ego_physics_model = KinematicBicycleModel(configs)
        self.vehicle_physics_model = KinematicBicycleModel(configs)

        self.world_map = CarlaDataProvider.get_map()
        self.ego_vehicle = CarlaDataProvider.get_hero_actor()
        self.world = CarlaDataProvider.get_world()

        # Check if the vehicle starts from a parking spot
        distance_to_road = self.global_plan_world_coord[0][0].location.distance(
            self.ego_vehicle.get_location()
        )

        # The first waypoint starts at the lane center, hence it will be more than 2 m away from the center of the ego vehicle at the beginning if the route starts with a parking exit scenario.
        starts_with_parking_exit = distance_to_road > 2

        # Setup planners
        self.waypoint_planner = PrivilegedRoutePlanner(configs.privileged_route_planner)
        self.waypoint_planner.setup_route(
            self.global_plan_world_coord,
            self.world,
            self.world_map,
            starts_with_parking_exit,
            self.ego_vehicle.get_location(),
        )
        self.waypoint_planner.save()

        self.command_planner = RoutePlanner(
            configs.route_planner.min_distance,
            configs.route_planner.max_distance,
        )
        self.command_planner.set_route(self._global_plan_world_coord)

        # Navigation command buffer, needed because the correct command comes from the last cleared waypoint
        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.next_commands = deque(maxlen=2)
        self.next_commands.append(4)
        self.next_commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

        # Setup controllers
        self.longitudinal_controller = LongitudinalLinearRegressionController(
            configs.longitudinal_controller
        )
        self.lateral_controller = LateralPIDController(self.config)

        # Initialize controls
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.target_speed = self.config.target_speed_fast

        self.augmentation_translation = 0
        self.augmentation_rotation = 0

        # Angle to the next waypoint, normalized in [-1, 1] corresponding to [-90, 90]
        self.angle = 0.0

        self._reset_flags()
        self.junction = False
        self.aim_wp = None  # Waypoint the expert is steering towards
        self.remaining_route = None  # Remaining route
        self.remaining_route_original = None  # Remaining original route
        self.was_at_stop_sign = False
        self.cleared_stop_sign = False
        self.visible_walker_ids = []
        self.walker_past_pos = {}  # Position of walker in the last frame

        self.vehicle_lights = (
            carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        )
        # Set up logging

        # Preprocess traffic lights
        all_actors = self.world.get_actors()
        for actor in all_actors:
            if "traffic_light" in actor.type_id:
                center, waypoints = get_before_traffic_light_waypoints(
                    actor, self.world_map
                )
                self.list_traffic_lights.append((actor, center, waypoints))

        # Remove bugged 2-wheelers
        # https://github.com/carla-simulator/carla/issues/3670
        for actor in all_actors:
            if "vehicle" in actor.type_id:
                extent = actor.bounding_box.extent
                if extent.x < 0.001 or extent.y < 0.001 or extent.z < 0.001:
                    actor.destroy()

    def sensors(self):
        return

    def run_step(self, input_data):
        return

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self.global_plan_gps = global_plan_gps
        self.global_plan_world_coord = global_plan_world_coord

    def _reset_flags(self):
        self.traffic_light_hazard = False
        self.stop_sign_hazard = False
        self.vehicle_hazard = False
        self.vehicle_affecting_id = None
        self.walker_hazard = False

    def _get_speed_idm(
        self,
        ego_speed,
        desired_speed,
        leading_actor_length,
        leading_actor_speed,
        distance_to_leading_actor,
        s0=4.0,
        T=0.5,
    ):
        """Compute the target speed for the ego vehicle using the Intelligent Driver Model (IDM).

        Args:

            s0 (float, optional): The minimum desired net distance.
            T (float, optional): The desired time headway.

        Returns:
            float: The target speed for the ego vehicle.
        """
        a = self.configs.idm.maximum_acceleration  # Maximum acceleration [m/s²]
        b = (
            self.configs.idm.comfortable_braking_deceleration_high_speed
            if ego_speed > self.configs.idm.comfortable_braking_deceleration_threshold
            else self.configs.idm.comfortable_braking_deceleration_low_speed
        )  # Comfortable deceleration [m/s²]
        delta = self.configs.idm.acceleration_exponent  # Acceleration exponent

        t_bound = self.configs.idm.t_bound

        def idm_equations(t, x):
            """
            Differential equations for the Intelligent Driver Model.

            Args:
                t (float): Time.
                x (list): State variables [position, speed].

            Returns:
                list: Derivatives of the state variables.
            """
            ego_position, ego_speed = x

            speed_diff = ego_speed - leading_actor_speed
            s_star = s0 + ego_speed * T + ego_speed * speed_diff / 2.0 / np.sqrt(a * b)
            # The maximum is needed to avoid numerical un-stability
            s = max(
                0.1,
                distance_to_leading_actor
                + t * leading_actor_speed
                - ego_position
                - leading_actor_length,
            )
            dvdt = a * (1.0 - (ego_speed / desired_speed) ** delta - (s_star / s) ** 2)

            return [ego_speed, dvdt]

        # Set the initial conditions
        y0 = [0.0, ego_speed]

        # Integrate the differential equations using RK45
        rk45 = RK45(fun=idm_equations, t0=0.0, y0=y0, t_bound=t_bound)
        while rk45.status == "running":
            rk45.step()

        # The target speed is the final speed obtained from the integration
        target_speed = rk45.y[1]

        # Clip the target speed to non-negative values
        return np.clip(target_speed, 0, np.inf)

    def _get_speed_considering_leading_vehicle(
        self,
        initial_target_speed,
        predicted_bounding_boxes,
        near_lane_change,
        leading_vehicle_ids,
        rear_vehicle_ids,
        speed_reduced_by_obj,
        plant,
    ):
        """Compute the target speed for the ego vehicle considering the leading vehicle.

        Args:
            initial_target_speed (float): The initial target speed for the ego vehicle.
            predicted_bounding_boxes (dict): A dictionary mapping actor IDs to lists of predicted bounding boxes.
            near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.
            leading_vehicle_ids (list): A list of IDs for vehicles in front of the ego vehicle.
            rear_vehicle_ids (list): A list of IDs for vehicles behind the ego vehicle.
            speed_reduced_by_obj (list or None): A list containing [reduced speed, object type, object ID, distance] for the object that caused the most speed reduction, or None if no speed reduction.
            plant (bool): Whether to use plant.

        Returns:
            float: The target speed for the ego vehicle considering the leading vehicle.
        """
        target_speed_wrt_leading_vehicle = initial_target_speed
        ego_vehicle_location = self.ego_vehicle.get_location()

        if not plant:
            for vehicle_id, _ in predicted_bounding_boxes.items():
                if vehicle_id in leading_vehicle_ids and not near_lane_change:
                    # Vehicle is in front of the ego vehicle
                    ego_speed = self.ego_vehicle.get_velocity().length()
                    vehicle = self.world.get_actor(vehicle_id)
                    other_speed = vehicle.get_velocity().length()
                    distance_to_vehicle = ego_vehicle_location.distance(
                        vehicle.get_location()
                    )

                    # Compute the target speed using the IDM
                    target_speed_wrt_leading_vehicle = min(
                        target_speed_wrt_leading_vehicle,
                        self._get_speed_idm(
                            ego_speed,
                            initial_target_speed,
                            vehicle.bounding_box.extent.x * 2,
                            other_speed,
                            distance_to_vehicle,
                            s0=self.config.idm_leading_vehicle_minimum_distance,
                            T=self.config.idm_leading_vehicle_time_headway,
                        ),
                    )

                    # Update the object causing the most speed reduction
                    if (
                        speed_reduced_by_obj is None
                        or speed_reduced_by_obj[0] > target_speed_wrt_leading_vehicle
                    ):
                        speed_reduced_by_obj = [
                            target_speed_wrt_leading_vehicle,
                            vehicle.type_id,
                            vehicle.id,
                            distance_to_vehicle,
                        ]

            if self.visualize:
                for vehicle_id in predicted_bounding_boxes.keys():
                    # check if vehicle is in front of the ego vehicle
                    if (
                        vehicle_id in leading_vehicle_ids and not near_lane_change
                    ) or vehicle_id in rear_vehicle_ids:
                        vehicle = self.world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(
                            pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0
                        )
                        self.world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=self.config.leading_vehicle_color,
                            life_time=self.config.draw_life_time,
                        )

        return target_speed_wrt_leading_vehicle, speed_reduced_by_obj

    def _get_speed_considering_all_actors(
        self,
        initial_target_speed,
        ego_bounding_boxes,
        predicted_bounding_boxes,
        near_lane_change,
        leading_vehicle_ids,
        rear_vehicle_ids,
        speed_reduced_by_obj,
        nearby_walkers,
        nearby_walkers_ids,
    ):
        """Compute the target speeds for the ego vehicle considering all actors (vehicles, bicycles,
        and pedestrians) by checking for intersecting bounding boxes.

        Args:
            initial_target_speed (float): The initial target speed for the ego vehicle.
            ego_bounding_boxes (list): A list of bounding boxes for the ego vehicle at different future frames.
            predicted_bounding_boxes (dict): A dictionary mapping actor IDs to lists of predicted bounding boxes.
            near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.
            leading_vehicle_ids (list): A list of IDs for vehicles in front of the ego vehicle.
            rear_vehicle_ids (list): A list of IDs for vehicles behind the ego vehicle.
            speed_reduced_by_obj (list or None): A list containing [reduced speed, object type, object ID, distance] for the object that caused the most speed reduction, or None if no speed reduction.
            nearby_walkers (dict): A list of predicted bounding boxes of nearby pedestrians.
            nearby_walkers_ids (list): A list of IDs for nearby pedestrians.
        """
        target_speed_bicycle = initial_target_speed
        target_speed_pedestrian = initial_target_speed
        target_speed_vehicle = initial_target_speed
        ego_vehicle_location = self.ego_vehicle.get_location()
        hazard_color = self.configs.ego_vehicle_forecasted_bbs_hazard_color
        normal_color = self.configs.ego_vehicle_forecasted_bbs_normal_color
        color = normal_color

        # Iterate over the ego vehicle's bounding boxes and predicted bounding boxes of other actors
        for i, ego_bounding_box in enumerate(ego_bounding_boxes):
            for vehicle_id, bounding_boxes in predicted_bounding_boxes.items():
                # Skip leading and rear vehicles if not near a lane change
                if vehicle_id in leading_vehicle_ids and not near_lane_change:
                    continue
                elif vehicle_id in rear_vehicle_ids and not near_lane_change:
                    continue
                else:
                    # Check if the ego bounding box intersects with the predicted bounding box of the actor
                    intersects_with_ego = self.check_obb_intersection(
                        ego_bounding_box, bounding_boxes[i]
                    )
                    ego_speed = self.ego_vehicle.get_velocity().length()

                    if intersects_with_ego:
                        blocking_actor = self.world.get_actor(vehicle_id)

                        # Handle the case when the blocking actor is a bicycle
                        if (
                            "base_type" in blocking_actor.attributes
                            and blocking_actor.attributes["base_type"] == "bicycle"
                        ):
                            other_speed = blocking_actor.get_velocity().length()
                            distance_to_actor = ego_vehicle_location.distance(
                                blocking_actor.get_location()
                            )

                            # Compute the target speed for bicycles using the IDM
                            target_speed_bicycle = min(
                                target_speed_bicycle,
                                self._get_speed_idm(
                                    ego_speed,
                                    initial_target_speed,
                                    blocking_actor.bounding_box.extent.x * 2,
                                    other_speed,
                                    distance_to_actor,
                                    s0=self.config.idm_bicycle_minimum_distance,
                                    T=self.config.idm_bicycle_desired_time_headway,
                                ),
                            )

                            # Update the object causing the most speed reduction
                            if (
                                speed_reduced_by_obj is None
                                or speed_reduced_by_obj[0] > target_speed_bicycle
                            ):
                                speed_reduced_by_obj = [
                                    target_speed_bicycle,
                                    blocking_actor.type_id,
                                    blocking_actor.id,
                                    distance_to_actor,
                                ]

                        # Handle the case when the blocking actor is not a bicycle
                        else:
                            self.vehicle_hazard = True  # Set the vehicle hazard flag
                            self.vehicle_affecting_id = vehicle_id  # Store the ID of the vehicle causing the hazard
                            color = hazard_color  # Change the following colors from green to red (no hazard to hazard)
                            target_speed_vehicle = 0
                            distance_to_actor = blocking_actor.get_location().distance(
                                ego_vehicle_location
                            )

                            # Update the object causing the most speed reduction
                            if (
                                speed_reduced_by_obj is None
                                or speed_reduced_by_obj[0] > target_speed_vehicle
                            ):
                                speed_reduced_by_obj = [
                                    target_speed_vehicle,
                                    blocking_actor.type_id,
                                    blocking_actor.id,
                                    distance_to_actor,
                                ]

            # Iterate over nearby pedestrians and check for intersections with the ego bounding box
            for pedestrian_bb, pedestrian_id in zip(nearby_walkers, nearby_walkers_ids):
                if self.check_obb_intersection(ego_bounding_box, pedestrian_bb[i]):
                    color = hazard_color
                    ego_speed = self.ego_vehicle.get_velocity().length()
                    blocking_actor = self.world.get_actor(pedestrian_id)
                    distance_to_actor = ego_vehicle_location.distance(
                        blocking_actor.get_location()
                    )

                    # Compute the target speed for pedestrians using the IDM
                    target_speed_pedestrian = min(
                        target_speed_pedestrian,
                        self._get_speed_idm(
                            ego_speed,
                            initial_target_speed,
                            0.5 + self.ego_vehicle.bounding_box.extent.x,
                            0.0,
                            distance_to_actor,
                            s0=self.config.idm_pedestrian_minimum_distance,
                            T=self.config.idm_pedestrian_desired_time_headway,
                        ),
                    )

                    # Update the object causing the most speed reduction
                    if (
                        speed_reduced_by_obj is None
                        or speed_reduced_by_obj[0] > target_speed_pedestrian
                    ):
                        speed_reduced_by_obj = [
                            target_speed_pedestrian,
                            blocking_actor.type_id,
                            blocking_actor.id,
                            distance_to_actor,
                        ]

            if self.visualize:
                self.world.debug.draw_box(
                    box=ego_bounding_box,
                    rotation=ego_bounding_box.rotation,
                    thickness=0.1,
                    color=color,
                    life_time=self.config.draw_life_time,
                )

        return (
            target_speed_bicycle,
            target_speed_pedestrian,
            target_speed_vehicle,
            speed_reduced_by_obj,
        )

    def _get_speed_affected_by_traffic_light(self):
        close_traffic_lights = []
