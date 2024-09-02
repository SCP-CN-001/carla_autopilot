##!/usr/bin/env python3
# @File: expert_agent.py
# @Description: An expert agent for data collection in CARLA Leaderboard 2.0.
# @CreatedTime: 2024/07/08
# @Author: Yueyuan Li, PDM-Lite Team

import logging
import math
from collections import deque

import carla
import numpy as np
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from leaderboard.envs.sensor_interface import SensorInterface
from omegaconf import OmegaConf
from scipy.integrate import RK45
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from src.common_carla.actor import get_horizontal_distance, get_nearby_objects
from src.common_carla.bounding_box import check_obb_intersection
from src.common_carla.route import (
    get_previous_road_lane_ids,
    is_near_lane_change,
    is_overtaking_path_clear,
    sort_scenarios_by_distance,
)
from src.common_carla.traffic_light import get_before_traffic_light_waypoints
from src.controllers import LateralPIDController, LongitudinalLinearRegressionController

# from src.leaderboard_custom.scenarios.cheater import Cheater
from src.planners.privileged_route_planner import PrivilegedRoutePlanner
from src.planners.route_planner import RoutePlanner
from src.utils.geometry import get_angle_by_position, normalize_angle
from src.utils.kinematic_bicycle_model import KinematicBicycleModel


def get_entry_point():
    return "ExpertAgent"


class ExpertAgent(AutonomousAgent):
    """
    An expert agent for data collection in CARLA Leaderboard 2.0. This agent has access to the ground truth in the simulator directly.

    Adopted from: https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/team_code/autopilot.py
    """

    def __init__(self, carla_host, carla_port, debug=False):
        self.debug = debug == 1

        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()

        self.wallclock_t0 = None
        logging.info("Expert agent initialized.")

    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.sensor_interface.get_data(GameTime.get_frame())

        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()
        sim_ratio = 0 if wallclock_diff == 0 else timestamp / wallclock_diff

        print(
            "=== [Agent] -- Wallclock = {} -- System time = {} -- Game time = {} -- Ratio = {}x".format(
                str(wallclock)[:-3],
                format(wallclock_diff, ".3f"),
                format(timestamp, ".3f"),
                format(sim_ratio, ".3f"),
            )
        )

        control = self.run_step(input_data)
        control.manual_gear_shift = False

        return control

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self.origin_global_plan_gps = global_plan_gps
        self.origin_global_plan_world_coord = global_plan_world_coord

        logging.info(f"Sparse waypoints: {len(self._global_plan)}")
        logging.info(f"Dense waypoints: {len(self.origin_global_plan_gps)}")

    def parse_config(self, path_config: str):
        def decode_expression(x):
            return eval(x)

        OmegaConf.register_new_resolver("eval", decode_expression, use_cache=False)
        configs = OmegaConf.load(path_config)

        for key, value in configs.items():
            if key not in [
                "kinematic_bicycle_model",
                "privileged_route_planner",
                "route_planner",
                "longitudinal_linear_regression_controller",
                "lateral_pid_controller",
            ]:
                setattr(self, key, value)

        return configs

    def setup(self, path_to_conf_file: str):
        """
        Setup the attributes of the expert agent.

        Args:
            path_to_conf_file (str): The absolute path to the configuration file.
        """
        self.configs = self.parse_config(path_to_conf_file)
        self.step = -1

        # Dynamics models
        self.ego_physics_model = KinematicBicycleModel(
            self.configs.kinematic_bicycle_model
        )
        self.vehicle_physics_model = KinematicBicycleModel(
            self.configs.kinematic_bicycle_model
        )

        self.world = CarlaDataProvider.get_world()
        self.world_map = CarlaDataProvider.get_map()
        self.ego_vehicle = CarlaDataProvider.get_hero_actor()
        self.control = None

        # Check if the vehicle starts from a parking spot
        distance_road = self.origin_global_plan_world_coord[0][0].location.distance(
            self.ego_vehicle.get_location()
        )

        # The first waypoint starts at the lane center, hence it will be more than 2 m away from the center of the ego vehicle at the beginning if the route starts with a parking exit scenario.
        starts_with_parking_exit = distance_road > 2

        # Setup planners
        self.waypoint_planner = PrivilegedRoutePlanner(
            self.configs.privileged_route_planner
        )
        self.waypoint_planner.setup_route(
            self.origin_global_plan_world_coord,
            self.world,
            self.world_map,
            starts_with_parking_exit,
            self.ego_vehicle.get_location(),
        )
        self.waypoint_planner.save()

        self.command_planner = RoutePlanner(self.configs.route_planner)
        self.command_planner.set_route(self._global_plan_world_coord)

        # Navigation command buffer, needed because the correct command comes from the last cleared waypoint
        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.next_commands = deque(maxlen=2)
        self.next_commands.append(4)
        self.next_commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

        # To avoid failing the ActorBlockedTest, the agent has to move at least 0.1 m/s every 179 ticks
        self.ego_blocked_for_ticks = 0

        # Setup controllers
        self.longitudinal_controller = LongitudinalLinearRegressionController(
            self.configs.longitudinal_linear_regression_controller
        )
        self.lateral_controller = LateralPIDController(
            self.configs.lateral_pid_controller
        )
        self.lateral_pid_controller_speed_scale = (
            self.configs.lateral_pid_controller.speed_scale
        )
        self.lateral_pid_controller_speed_offset = (
            self.configs.lateral_pid_controller.speed_offset
        )
        self.lateral_pid_controller_default_lookahead = (
            self.configs.lateral_pid_controller.default_lookahead
        )
        self.lateral_pid_controller_max_lookahead_distance = (
            self.configs.lateral_pid_controller.max_lookahead_distance
        )

        # Initialize controls
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.target_speed = self.configs.target_speed_fast
        self.target_wp = None  # Waypoint the expert is steering towards

        self.augmentation_translation = 0
        self.augmentation_rotation = 0

        # Angle to the next waypoint, normalized in [-1, 1] corresponding to [-90, 90]
        self.angle = 0.0

        self._reset_flags()
        self.traffic_light_list = []
        self.is_junction = False
        self.remaining_route = None  # Remaining route
        self.remaining_route_original = None  # Remaining original route
        self.cleared_stop_sign = False

        # The bounding box of the traffic light that may affect the ego vehicle
        self.traffic_light_loc = []
        # The bounding box of the stop sign that may affect the ego vehicle
        self.stop_sign_bbox = []

        self.vehicle_lights = (
            carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        )

        # Preprocess traffic lights
        all_actors = self.world.get_actors()
        for actor in all_actors:
            if "traffic_light" in actor.type_id:
                center, waypoints = get_before_traffic_light_waypoints(
                    actor, self.world_map
                )
                self.traffic_light_list.append((actor, center, waypoints))

        # Remove bugged 2-wheelers
        # https://github.com/carla-simulator/carla/issues/3670
        for actor in all_actors:
            if "vehicle" in actor.type_id:
                extent = actor.bounding_box.extent
                if extent.x < 0.001 or extent.y < 0.001 or extent.z < 0.001:
                    actor.destroy()

        logging.info("Expert agent setup completed.")

    def sensors(self):
        sensors = [
            {"type": "sensor.opendrive_map", "reading_frequency": 1e-6, "id": "hd_map"},
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
        ]

        return sensors

    def get_ego_state(self, input_data):
        # Get the vehicle's speed from its velocity vector
        speed = self.ego_vehicle.get_velocity().length()

        # Checks the compass for Nans and rotates it into the default CARLA coordinate system with range [-pi,pi].
        compass = input_data["imu"][1][-1]
        if math.isnan(compass):
            compass = 0.0
        # The minus 90.0 degree is because the compass sensor uses a different
        # coordinate system then CARLA. Check the coordinate_sytems.txt file
        compass = normalize_angle(compass - np.deg2rad(90.0))

        # Get the vehicle's position from its location
        position = self.ego_vehicle.get_location()
        gps = np.array([position.x, position.y, position.z])

        return gps, speed, compass

    def run_step(self, input_data):
        """
        Run a single step of the agent's control loop.

        Args:
            input_data (dict): Input data for the current step.

        Returns:
            carla.VehicleControl: The control commands for the current step.
        """
        self.step += 1
        self.traffic_light_loc = []
        self.stop_sign_bbox = []

        # Get the control commands and driving data for the current step
        control = self.get_control(input_data)
        self.control = control  # for visiting the control command anywhere in the class

        return control

    def get_control(self, input_data):
        """
        Compute the control commands and save the driving data for the current frame.

        Args:
            input_data (dict): Input data for the current frame.

        Returns:
            tuple: A tuple containing the control commands (steer, throttle, brake) and the driving data.
        """
        ego_gps, ego_speed, ego_compass = self.get_ego_state(input_data)

        # Waypoint planning and route generation
        (
            route_np,
            route_wp,
            _,
            distance_next_traffic_light,
            next_traffic_light,
            distance_next_stop_sign,
            next_stop_sign,
            speed_limit,
        ) = self.waypoint_planner.run_step(ego_gps)

        # Extract relevant route information
        self.remaining_route = route_np[int(self.distance_first_checkpoint) :][
            :: self.points_per_meter
        ]
        self.remaining_route_original = self.waypoint_planner.original_route_points[
            self.waypoint_planner.route_index :
        ][int(self.distance_first_checkpoint) :][:: self.points_per_meter]

        target_speed = speed_limit * self.factor_target_speed_limit

        # Reduce target speed if there is a junction ahead
        for i in range(min(self.max_lookahead_near_junction, len(route_wp))):
            if route_wp[i].is_junction:
                target_speed = min(target_speed, self.max_speed_junction)
                break

        # Get the list of vehicles in the scene
        actor_list = self.world.get_actors()
        vehicle_list = list(actor_list.filter("*vehicle*"))

        # Manage route obstacle scenarios and adjust target speed
        (
            target_speed_route_obstacle,
            keep_driving,
            speed_reduced_by_obj,
        ) = self._manage_route_obstacle_scenarios(
            ego_speed, target_speed, route_wp, vehicle_list, route_np
        )

        # In case the agent overtakes an obstacle, keep driving in case the opposite lane is free instead of using idm
        # and the kinematic bicycle model forecasts
        if keep_driving:
            brake, target_speed = False, target_speed_route_obstacle
        else:
            (
                brake,
                target_speed,
                speed_reduced_by_obj,
            ) = self._get_speed_brake_and_target(
                target_speed,
                actor_list,
                vehicle_list,
                route_np,
                distance_next_traffic_light,
                next_traffic_light,
                distance_next_stop_sign,
                next_stop_sign,
                speed_reduced_by_obj,
            )

        target_speed = min(target_speed, target_speed_route_obstacle)

        # Determine if the ego vehicle is at a junction
        ego_waypoint = self.world_map.get_waypoint(self.ego_vehicle.get_location())
        self.is_junction = ego_waypoint.is_junction

        # Compute throttle and brake control
        throttle, control_brake = self.longitudinal_controller.get_throttle_and_brake(
            brake, target_speed, ego_speed
        )

        # Compute steering control
        steer = self._get_steer(route_np, ego_gps, ego_compass, ego_speed)

        # Create the control command
        control = carla.VehicleControl()
        control.steer = steer + self.steer_noise * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake or control_brake)

        # Apply brake if the vehicle is stopped to prevent rolling back
        if control.throttle == 0 and ego_speed < self.min_speed_prevent_rolling_back:
            control.brake = 1

        # Apply throttle if the vehicle is blocked for too long
        ego_velocity = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if ego_velocity < 0.1:
            self.ego_blocked_for_ticks += 1
        else:
            self.ego_blocked_for_ticks = 0

        if self.ego_blocked_for_ticks >= self.max_blocked_ticks:
            control.throttle = 1
            control.brake = 0

        # Save control commands and target speed
        self.steer = control.steer
        self.throttle = control.throttle
        self.brake = control.brake
        self.target_speed = target_speed

        # Update speed histogram if enabled
        # if self.make_histogram:
        #     self.speed_histogram.append((self.target_speed * 3.6) if not brake else 0.0)

        # Get the target and next target points from the command planner
        command_route = self.command_planner.run_step(ego_gps)
        if len(command_route) > 2:
            target_point, far_command = command_route[1]
            next_target_point, next_far_command = command_route[2]
        elif len(command_route) > 1:
            target_point, far_command = command_route[1]
            next_target_point, next_far_command = command_route[1]
        else:
            target_point, far_command = command_route[0]
            next_target_point, next_far_command = command_route[0]

        # Update command history and save driving datas
        if (target_point != self.target_point_prev).all():
            self.target_point_prev = target_point
            self.commands.append(far_command.value)
            self.next_commands.append(next_far_command.value)

        return control

    def _reset_flags(self):
        self.traffic_light_hazard = False
        self.stop_sign_hazard = False
        self.stop_sign_close = False
        self.vehicle_hazard = False
        self.vehicle_affecting_id = None
        self.walker_hazard = False
        self.walker_close = False

    def _manage_route_obstacle_scenarios(
        self, ego_speed, target_speed, route_waypoints, vehicle_list, route_points
    ):
        """This method handles various obstacle and scenario situations that may arise during navigation.

        It adjusts the target speed, modifies the route, and determines if the ego vehicle should keep driving or wait.

        The method supports different scenario types including:
        - InvadingTurn
        - Accident
        - ConstructionObstacle
        - ParkedObstacle
        - AccidentTwoWays
        - ConstructionObstacleTwoWays
        - ParkedObstacleTwoWays
        - VehicleOpensDoorTwoWays
        - HazardAtSideLaneTwoWays
        - HazardAtSideLane
        - YieldToEmergencyVehicle.

        Args:
            target_speed (float): The current target speed of the ego vehicle.
            route_waypoints (list): A list of waypoints representing the current route.
            vehicle_list (list): A list of all vehicles in the simulation.
            route_points (numpy.ndarray): A numpy array containing the current route points.

        Returns:
            tuple: A tuple containing the updated target speed, a boolean indicating whether to keep driving,
                and a list containing information about a potential decreased target speed due to an object.
        """
        keep_driving = False
        speed_reduced_by_obj = [
            target_speed,
            None,
            None,
            None,
        ]  # [target_speed, type, id, distance]

        # Remove scenarios that ended with a scenario timeout
        # active_scenarios = Cheater.active_scenarios.copy()
        active_scenarios = CarlaDataProvider.active_scenarios.copy()
        for i, (scenario_type, scenario_data) in enumerate(active_scenarios):
            first_actor, last_actor = scenario_data[:2]
            if not first_actor.is_alive or (
                last_actor is not None and not last_actor.is_alive
            ):
                # Cheater.active_scenarios.remove(active_scenarios[i])
                CarlaDataProvider.active_scenarios.remove(active_scenarios[i])

        # Only continue if there are some active scenarios available
        # if len(Cheater.active_scenarios) != 0:
        if len(CarlaDataProvider.active_scenarios) != 0:
            ego_location = self.ego_vehicle.get_location()

            # Sort the scenarios by distance if there is more than one active scenario
            # if len(Cheater.active_scenarios) != 1:
            if len(CarlaDataProvider.active_scenarios) != 1:
                sort_scenarios_by_distance(ego_location)

            # scenario_type, scenario_data = Cheater.active_scenarios[0]
            scenario_type, scenario_data = CarlaDataProvider.active_scenarios[0]

            if scenario_type == "InvadingTurn":
                first_cone, last_cone, offset = scenario_data

                closest_distance = first_cone.get_location().distance(ego_location)

                if closest_distance < self.distance_process_scenario:
                    self.waypoint_planner.shift_route_for_invading_turn(
                        first_cone, last_cone, offset
                    )
                    # Cheater.active_scenarios = Cheater.active_scenarios[1:]
                    CarlaDataProvider.active_scenarios = (
                        CarlaDataProvider.active_scenarios[1:]
                    )

            elif scenario_type in [
                "Accident",
                "ConstructionObstacle",
                "ParkedObstacle",
            ]:
                first_actor, last_actor, direction = scenario_data[:3]

                horizontal_distance = get_horizontal_distance(
                    self.ego_vehicle, first_actor
                )

                # Shift the route around the obstacles
                if horizontal_distance < self.distance_process_scenario:
                    transition_length = {
                        "Accident": self.distance_smooth_transition,
                        "ConstructionObstacle": self.distance_smooth_transition_construction_obstacle,
                        "ParkedObstacle": self.distance_smooth_transition,
                    }[scenario_type]
                    _, _ = self.waypoint_planner.shift_route_around_actors(
                        first_actor, last_actor, direction, transition_length
                    )
                    # Cheater.active_scenarios = Cheater.active_scenarios[1:]
                    CarlaDataProvider.active_scenarios = (
                        CarlaDataProvider.active_scenarios[1:]
                    )

            elif scenario_type in [
                "AccidentTwoWays",
                "ConstructionObstacleTwoWays",
                "ParkedObstacleTwoWays",
                "VehicleOpensDoorTwoWays",
            ]:
                (
                    first_actor,
                    last_actor,
                    direction,
                    changed_route,
                    from_index,
                    to_index,
                    path_clear,
                ) = scenario_data

                # change the route if the ego is close enough to the obstacle
                horizontal_distance = get_horizontal_distance(
                    self.ego_vehicle, first_actor
                )

                # Shift the route around the obstacles
                if (
                    horizontal_distance < self.distance_process_scenario
                    and not changed_route
                ):
                    transition_length = {
                        "AccidentTwoWays": self.distance_smooth_transition_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.distance_smooth_transition_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.distance_smooth_transition_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.distance_smooth_transition_vehicle_opens_door_two_ways,
                    }[scenario_type]
                    add_before_length = {
                        "AccidentTwoWays": self.distance_before_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.distance_before_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.distance_before_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.distance_before_vehicle_opens_door_two_ways,
                    }[scenario_type]
                    add_after_length = {
                        "AccidentTwoWays": self.distance_after_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.distance_after_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.distance_after_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.distance_after_vehicle_opens_door_two_ways,
                    }[scenario_type]
                    factor = {
                        "AccidentTwoWays": self.factor_accident_two_ways,
                        "ConstructionObstacleTwoWays": self.factor_construction_obstacle_two_ways,
                        "ParkedObstacleTwoWays": self.factor_parked_obstacle_two_ways,
                        "VehicleOpensDoorTwoWays": self.factor_vehicle_opens_door_two_ways,
                    }[scenario_type]

                    (
                        from_index,
                        to_index,
                    ) = self.waypoint_planner.shift_route_around_actors(
                        first_actor,
                        last_actor,
                        direction,
                        transition_length,
                        lane_transition_factor=factor,
                        extra_length_before=add_before_length,
                        extra_length_after=add_after_length,
                    )

                    changed_route = True
                    scenario_data[3] = changed_route
                    scenario_data[4] = from_index
                    scenario_data[5] = to_index

                # Check if the ego can overtake the obstacle
                if (
                    changed_route
                    and from_index - self.waypoint_planner.route_index
                    < self.distance_overtake_two_way_scenario
                    and not path_clear
                ):
                    # Get previous roads and lanes of the target lane
                    target_lane = (
                        route_waypoints[0].get_left_lane()
                        if direction == "right"
                        else route_waypoints[0].get_right_lane()
                    )
                    if target_lane is None:
                        return target_speed, keep_driving, speed_reduced_by_obj
                    previous_road_lane_ids = get_previous_road_lane_ids(
                        target_lane, self.distance_retrieve_id
                    )

                    overtake_speed = (
                        self.speed_overtake_vehicle_opens_door_two_ways
                        if scenario_type == "VehicleOpensDoorTwoWays"
                        else self.speed_overtake
                    )
                    path_clear = is_overtaking_path_clear(
                        self.world_map,
                        self.ego_vehicle,
                        vehicle_list,
                        target_speed,
                        self.waypoint_planner.route_points,
                        from_index,
                        to_index,
                        previous_road_lane_ids,
                        self.safety_distance_check_path_free,
                        self.safety_time_check_path_free,
                        min_speed=overtake_speed,
                    )

                    scenario_data[6] = path_clear

                # If the overtaking path is clear, keep driving; otherwise, wait behind the obstacle
                if path_clear:
                    if (
                        self.waypoint_planner.route_index
                        >= to_index - self.distance_delete_scenario_in_two_ways
                    ):
                        # Cheater.active_scenarios = Cheater.active_scenarios[1:]
                        CarlaDataProvider.active_scenarios = (
                            CarlaDataProvider.active_scenarios[1:]
                        )
                    target_speed = {
                        "AccidentTwoWays": self.speed_overtake,
                        "ConstructionObstacleTwoWays": self.speed_overtake,
                        "ParkedObstacleTwoWays": self.speed_overtake,
                        "VehicleOpensDoorTwoWays": self.speed_overtake_vehicle_opens_door_two_ways,
                    }[scenario_type]
                    keep_driving = True
                else:
                    distance_leading_actor = (
                        float(from_index + 15 - self.waypoint_planner.route_index)
                        / self.points_per_meter
                    )
                    target_speed = self._get_speed_idm(
                        ego_speed,
                        target_speed,
                        self.ego_vehicle.bounding_box.extent.x,
                        0,
                        distance_leading_actor,
                        s0=self.idm_min_distance_two_way_scenario,
                        T=self.idm_time_two_way_scenario,
                    )

                    # Update the object causing the most speed reduction
                    if (
                        speed_reduced_by_obj is None
                        or speed_reduced_by_obj[0] > target_speed
                    ):
                        speed_reduced_by_obj = [
                            target_speed,
                            first_actor.type_id,
                            first_actor.id,
                            distance_leading_actor,
                        ]

            elif scenario_type == "HazardAtSideLaneTwoWays":
                (
                    first_actor,
                    last_actor,
                    changed_route,
                    from_index,
                    to_index,
                    path_clear,
                ) = scenario_data

                horizontal_distance = get_horizontal_distance(
                    self.ego_vehicle, first_actor
                )

                if (
                    horizontal_distance
                    < self.distance_process_hazard_side_lane_two_ways
                    and not changed_route
                ):
                    to_index = self.waypoint_planner.get_closest_route_index(
                        self.waypoint_planner.route_index, last_actor.get_location()
                    )

                    # Assume the bicycles don't drive more than 7.5 m during the overtaking process
                    to_index += 135
                    from_index = self.waypoint_planner.route_index

                    starting_wp = route_waypoints[0].get_left_lane()
                    previous_road_lane_ids = get_previous_road_lane_ids(
                        starting_wp, self.distance_retrieve_id
                    )
                    path_clear = is_overtaking_path_clear(
                        self.world_map,
                        self.ego_vehicle,
                        vehicle_list,
                        target_speed,
                        self.waypoint_planner.route_points,
                        from_index,
                        to_index,
                        previous_road_lane_ids,
                        self.safety_distance_check_path_free,
                        self.safety_time_check_path_free,
                        min_speed=self.speed_overtake,
                    )

                    if path_clear:
                        transition_length = self.distance_smooth_transition
                        self.waypoint_planner.shift_route_smoothly(
                            from_index, to_index, True, transition_length
                        )

                        changed_route = True
                        scenario_data[2] = changed_route
                        scenario_data[3] = from_index
                        scenario_data[4] = to_index
                        scenario_data[5] = path_clear

                # the overtaking path is clear
                if path_clear:
                    # Check if the overtaking is done
                    if self.waypoint_planner.route_index >= to_index:
                        # Cheater.active_scenarios = Cheater.active_scenarios[1:]
                        CarlaDataProvider.active_scenarios = (
                            CarlaDataProvider.active_scenarios[1:]
                        )
                    # Overtake with max. 50 km/h
                    target_speed, keep_driving = (
                        self.speed_overtake,
                        True,
                    )

            elif scenario_type == "HazardAtSideLane":
                (
                    first_actor,
                    last_actor,
                    changed_first_part_of_route,
                    from_index,
                    to_index,
                    path_clear,
                ) = scenario_data

                horizontal_distance = get_horizontal_distance(
                    self.ego_vehicle, last_actor
                )

                if (
                    horizontal_distance < self.distance_process_hazard_side_lane
                    and not changed_first_part_of_route
                ):
                    transition_length = self.distance_smooth_transition
                    (
                        from_index,
                        to_index,
                    ) = self.waypoint_planner.shift_route_around_actors(
                        first_actor, last_actor, "right", transition_length
                    )

                    to_index -= transition_length
                    changed_first_part_of_route = True
                    scenario_data[2] = changed_first_part_of_route
                    scenario_data[3] = from_index
                    scenario_data[4] = to_index

                if changed_first_part_of_route:
                    to_idx_ = self.waypoint_planner.extend_lane_shift_transition_for_hazard_at_side_lane(
                        last_actor, to_index
                    )
                    to_index = to_idx_
                    scenario_data[4] = to_index

                if self.waypoint_planner.route_index > to_index:
                    # Cheater.active_scenarios = Cheater.active_scenarios[1:]
                    CarlaDataProvider.active_scenarios = (
                        CarlaDataProvider.active_scenarios[1:]
                    )

            elif scenario_type == "YieldToEmergencyVehicle":
                (
                    emergency_veh,
                    _,
                    changed_route,
                    from_index,
                    to_index,
                    to_left,
                ) = scenario_data

                horizontal_distance = get_horizontal_distance(
                    self.ego_vehicle, emergency_veh
                )

                if (
                    horizontal_distance < self.distance_process_scenario
                    and not changed_route
                ):
                    # Assume the emergency vehicle doesn't drive more than 20 m during the overtaking process
                    from_index = (
                        self.waypoint_planner.route_index + 30 * self.points_per_meter
                    )
                    to_index = (
                        from_index
                        + int(2 * self.points_per_meter) * self.points_per_meter
                    )

                    transition_length = self.distance_smooth_transition
                    to_left = (
                        self.waypoint_planner.route_waypoints[from_index].lane_change
                        != carla.LaneChange.Right
                    )
                    self.waypoint_planner.shift_route_smoothly(
                        from_index, to_index, to_left, transition_length
                    )

                    changed_route = True
                    to_index -= transition_length
                    scenario_data[2] = changed_route
                    scenario_data[3] = from_index
                    scenario_data[4] = to_index
                    scenario_data[5] = to_left

                if changed_route:
                    to_idx_ = self.waypoint_planner.extend_lane_shift_transition_for_yield_to_emergency_vehicle(
                        to_left, to_index
                    )
                    to_index = to_idx_
                    scenario_data[4] = to_index

                    # Check if the emergency vehicle is in front of the ego vehicle
                    diff = emergency_veh.get_location() - ego_location
                    dot_res = (
                        self.ego_vehicle.get_transform().get_forward_vector().dot(diff)
                    )
                    if dot_res > 0:
                        # Cheater.active_scenarios = Cheater.active_scenarios[1:]
                        CarlaDataProvider.active_scenarios = (
                            CarlaDataProvider.active_scenarios[1:]
                        )

        if self.debug:
            for i in range(
                min(
                    route_points.shape[0] - 1,
                    self.distance_draw_future_route,
                )
            ):
                location = route_points[i]
                location = carla.Location(location[0], location[1], location[2] + 0.1)
                self.world.debug.draw_point(
                    location=location,
                    size=0.05,
                    color=self.color_future_route,
                    life_time=self.draw_life_time,
                )

        return target_speed, keep_driving, speed_reduced_by_obj

    def _get_speed_idm(
        self,
        ego_speed,
        target_speed,
        leading_actor_length,
        leading_actor_speed,
        distance_leading_actor,
        s0=4.0,
        T=0.5,
    ):
        """Compute the target speed for the ego vehicle using the Intelligent Driver Model (IDM).

        Args:
            ego_speed (float): The current speed of the ego vehicle.
            target_speed (float): The desired target speed for the ego vehicle.
            leading_actor_length (float): The length of the leading vehicle or obstacle.
            leading_actor_speed (float): The speed of the leading vehicle or obstacle.
            distance_leading_actor (float): The distance to the leading vehicle or obstacle.
            s0 (float, optional): The minimum desired net distance.
            T (float, optional): The desired time headway.

        Returns:
            float: The target speed for the ego vehicle.
        """
        a = self.idm_max_acceleration  # Maximum acceleration [m/s²]
        b = (
            self.idm_comfortable_braking_deceleration_high_speed
            if ego_speed > self.idm_comfortable_braking_threshold_speed
            else self.idm_comfortable_braking_deceleration_low_speed
        )  # Comfortable deceleration [m/s²]
        delta = self.idm_acceleration_exponent  # Acceleration exponent

        t_bound = self.idm_time_boundary

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
            # The maximum is needed to avoid numerical instability
            s = max(
                0.1,
                distance_leading_actor
                + t * leading_actor_speed
                - ego_position
                - leading_actor_length,
            )
            dvdt = a * (1.0 - (ego_speed / target_speed) ** delta - (s_star / s) ** 2)

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

    def _get_speed_wrt_leading_vehicle(
        self,
        ego_location,
        target_speed,
        predicted_bboxes,
        near_lane_change,
        leading_vehicle_ids,
        rear_vehicle_ids,
        speed_reduced_by_obj,
    ):
        """Compute the target speed for the ego vehicle considering the leading vehicle.

        Args:
            ego_location (carla.Location): The location of the ego vehicle.
            target_speed (float): The initial target speed for the ego vehicle.
            predicted_bboxes (dict): A dictionary mapping actor IDs to lists of predicted bounding boxes.
            near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.
            leading_vehicle_ids (list): A list of IDs for vehicles in front of the ego vehicle.
            rear_vehicle_ids (list): A list of IDs for vehicles behind the ego vehicle.
            speed_reduced_by_obj (list or None): A list containing [reduced speed, object type, object ID, distance] for the object that caused the most speed reduction, or None if no speed reduction.

        Returns:
            float: The target speed for the ego vehicle considering the leading vehicle.
        """
        target_speed_wrt_leading_vehicle = target_speed

        for vehicle_id, _ in predicted_bboxes.items():
            if vehicle_id in leading_vehicle_ids and not near_lane_change:
                # Vehicle is in front of the ego vehicle
                ego_speed = self.ego_vehicle.get_velocity().length()
                vehicle = self.world.get_actor(vehicle_id)
                other_speed = vehicle.get_velocity().length()
                distance_vehicle = ego_location.distance(vehicle.get_location())

                # Compute the target speed using the IDM
                target_speed_wrt_leading_vehicle = min(
                    target_speed_wrt_leading_vehicle,
                    self._get_speed_idm(
                        ego_speed,
                        target_speed,
                        vehicle.bounding_box.extent.x * 2,
                        other_speed,
                        distance_vehicle,
                        s0=self.idm_min_distance_leading_vehicle,
                        T=self.idm_time_leading_vehicle,
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
                        distance_vehicle,
                    ]

        if self.debug:
            for vehicle_id in predicted_bboxes.keys():
                # check if vehicle is in front of the ego vehicle
                if vehicle_id in leading_vehicle_ids and not near_lane_change:
                    extent = vehicle.bounding_box.extent
                    bbox = carla.BoundingBox(vehicle.get_location(), extent)
                    bbox.rotation = carla.Rotation(
                        pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0
                    )
                    self.world.debug.draw_box(
                        box=bbox,
                        rotation=bbox.rotation,
                        thickness=0.5,
                        color=self.color_bbox_leading_vehicle,
                        life_time=self.draw_life_time,
                    )
                elif vehicle_id in rear_vehicle_ids:
                    vehicle = self.world.get_actor(vehicle_id)
                    extent = vehicle.bounding_box.extent
                    bbox = carla.BoundingBox(vehicle.get_location(), extent)
                    bbox.rotation = carla.Rotation(
                        pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0
                    )
                    self.world.debug.draw_box(
                        box=bbox,
                        rotation=bbox.rotation,
                        thickness=0.5,
                        color=self.color_bbox_rear_vehicle,
                        life_time=self.draw_life_time,
                    )

        return target_speed_wrt_leading_vehicle, speed_reduced_by_obj

    def _get_speed_wrt_all_actors(
        self,
        target_speed,
        ego_bboxes,
        predicted_bboxes,
        near_lane_change,
        leading_vehicle_ids,
        rear_vehicle_ids,
        speed_reduced_by_obj,
        nearby_walkers,
        nearby_walkers_ids,
    ):
        """Compute the target speeds for the ego vehicle considering all actors (vehicles, bicycles,  and walkers) by checking for intersecting bounding boxes.

        Args:
            target_speed (float): The initial target speed for the ego vehicle.
            ego_bboxes (list): A list of bounding boxes for the ego vehicle at different future frames.
            predicted_bboxes (dict): A dictionary mapping actor IDs to lists of predicted bounding boxes.
            near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.
            leading_vehicle_ids (list): A list of IDs for vehicles in front of the ego vehicle.
            rear_vehicle_ids (list): A list of IDs for vehicles behind the ego vehicle.
            speed_reduced_by_obj (list or None): A list containing [reduced speed, object type, object ID, distance] for the object that caused the most speed reduction, or None if no speed reduction.
            nearby_walkers (dict): A list of predicted bounding boxes of nearby walkers.
            nearby_walkers_ids (list): A list of IDs for nearby walkers.
        """
        target_speed_bicycle = target_speed
        target_speed_walker = target_speed
        target_speed_vehicle = target_speed
        ego_location = self.ego_vehicle.get_location()
        normal_color = self.color_bbox_forecasted_normal
        hazard_color = self.color_bbox_forecasted_hazard
        color = normal_color

        # Iterate over the ego vehicle's bounding boxes and predicted bounding boxes of other actors
        for i, ego_bbox in enumerate(ego_bboxes):
            for vehicle_id, bboxes in predicted_bboxes.items():
                # Skip leading and rear vehicles if not near a lane change
                if vehicle_id in leading_vehicle_ids and not near_lane_change:
                    continue
                elif vehicle_id in rear_vehicle_ids and not near_lane_change:
                    continue
                else:
                    # Check if the ego bounding box intersects with the predicted bounding box of the actor
                    intersects_with_ego = check_obb_intersection(ego_bbox, bboxes[i])
                    ego_speed = self.ego_vehicle.get_velocity().length()

                    if intersects_with_ego:
                        blocking_actor = self.world.get_actor(vehicle_id)

                        # Handle the case when the blocking actor is a bicycle
                        if (
                            "base_type" in blocking_actor.attributes
                            and blocking_actor.attributes["base_type"] == "bicycle"
                        ):
                            other_speed = blocking_actor.get_velocity().length()
                            distance_actor = ego_location.distance(
                                blocking_actor.get_location()
                            )

                            # Compute the target speed for bicycles using the IDM
                            target_speed_bicycle = min(
                                target_speed_bicycle,
                                self._get_speed_idm(
                                    ego_speed,
                                    target_speed,
                                    blocking_actor.bounding_box.extent.x * 2,
                                    other_speed,
                                    distance_actor,
                                    s0=self.idm_min_distance_bicycle,
                                    T=self.idm_time_bicycle,
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
                                    distance_actor,
                                ]

                        # Handle the case when the blocking actor is not a bicycle
                        else:
                            self.vehicle_hazard = True  # Set the vehicle hazard flag
                            self.vehicle_affecting_id = vehicle_id  # Store the ID of the vehicle causing the hazard
                            color = hazard_color  # Change the following colors from green to red (no hazard to hazard)
                            target_speed_vehicle = 0
                            distance_actor = blocking_actor.get_location().distance(
                                ego_location
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
                                    distance_actor,
                                ]

            # Iterate over nearby walkers and check for intersections with the ego bounding box
            for walker_bbox, walker_id in zip(nearby_walkers, nearby_walkers_ids):
                if check_obb_intersection(ego_bbox, walker_bbox[i]):
                    color = hazard_color
                    ego_speed = self.ego_vehicle.get_velocity().length()
                    blocking_actor = self.world.get_actor(walker_id)
                    distance_actor = ego_location.distance(
                        blocking_actor.get_location()
                    )

                    # Compute the target speed for walkers using the IDM
                    target_speed_walker = min(
                        target_speed_walker,
                        self._get_speed_idm(
                            ego_speed,
                            target_speed,
                            0.5 + self.ego_vehicle.bounding_box.extent.x,
                            0.0,
                            distance_actor,
                            s0=self.idm_min_distance_walker,
                            T=self.idm_time_walker,
                        ),
                    )

                    # Update the object causing the most speed reduction
                    if (
                        speed_reduced_by_obj is None
                        or speed_reduced_by_obj[0] > target_speed_walker
                    ):
                        speed_reduced_by_obj = [
                            target_speed_walker,
                            blocking_actor.type_id,
                            blocking_actor.id,
                            distance_actor,
                        ]

            if self.debug:
                self.world.debug.draw_box(
                    box=ego_bbox,
                    rotation=ego_bbox.rotation,
                    thickness=0.1,
                    color=color,
                    life_time=self.draw_life_time,
                )

        return (
            target_speed_bicycle,
            target_speed_walker,
            target_speed_vehicle,
            speed_reduced_by_obj,
        )

    def _get_speed_affected_by_traffic_light(
        self,
        ego_location,
        ego_speed,
        target_speed,
        distance_next_traffic_light,
        next_traffic_light,
    ):
        """Handles the behavior of the ego vehicle when approaching a traffic light.

        Args:
            ego_location (carla.Location): The location of the ego vehicle.
            ego_speed (float): The current speed of the ego vehicle.
            target_speed (float): The current target speed of the ego vehicle.
            distance_next_traffic_light (float): The distance from the ego vehicle to the next traffic light.
            next_traffic_light (carla.TrafficLight or None): The next traffic light in the route.

        Returns:
            float: The adjusted target speed for the ego vehicle.
        """
        for light, center, waypoints in self.traffic_light_list:
            center_loc = carla.Location(center)
            if center_loc.distance(ego_location) > self.light_radius:
                continue

            for wp in waypoints:
                # * 0.9 to make the box slightly smaller than the street to prevent overlapping boxes.
                length_bbox = carla.Vector3D(
                    (wp.lane_width / 2.0) * 0.9,
                    light.trigger_volume.extent.y,
                    light.trigger_volume.extent.z,
                )
                length_bbox = carla.Vector3D(1.5, 1.5, 0.5)

                bbox = carla.BoundingBox(wp.transform.location, length_bbox)

                global_rot = light.get_transform().rotation
                bbox.rotation = carla.Rotation(
                    pitch=global_rot.pitch, yaw=global_rot.yaw, roll=global_rot.roll
                )

                affect_ego = (
                    next_traffic_light is not None and light.id == next_traffic_light.id
                )

                if affect_ego and light.state in [
                    carla.libcarla.TrafficLightState.Red,
                    carla.libcarla.TrafficLightState.Yellow,
                ]:
                    self.traffic_light_loc.append(center_loc)

                if self.debug:
                    if light.state == carla.libcarla.TrafficLightState.Red:
                        color = carla.Color(255, 0, 0, 255)
                    elif light.state == carla.libcarla.TrafficLightState.Yellow:
                        color = carla.Color(255, 255, 0, 255)
                    elif light.state == carla.libcarla.TrafficLightState.Green:
                        color = carla.Color(0, 255, 0, 255)
                    elif light.state == carla.libcarla.TrafficLightState.Off:
                        color = carla.Color(0, 0, 0, 255)
                    else:  # unknown
                        color = carla.Color(0, 0, 255, 255)

                    self.world.debug.draw_box(
                        box=bbox,
                        rotation=bbox.rotation,
                        thickness=0.1,
                        color=color,
                        life_time=self.draw_life_time,
                    )

                    self.world.debug.draw_point(
                        wp.transform.location
                        + carla.Location(z=light.trigger_volume.location.z),
                        size=0.1,
                        color=color,
                        life_time=(1.0 / self.frame_rate_carla) + 1e-6,
                    )

        if (
            next_traffic_light is None
            or next_traffic_light.state == carla.TrafficLightState.Green
        ):
            # No traffic light or green light, continue with the current target speed
            return target_speed

        # Compute the target speed using the IDM
        target_speed = self._get_speed_idm(
            ego_speed,
            target_speed,
            0.0,
            0.0,
            distance_next_traffic_light,
            s0=self.idm_min_distance_traffic_light,
            T=self.idm_time_traffic_light,
        )

        return target_speed

    def _get_speed_affected_by_stop_sign(
        self, ego_location, ego_speed, target_speed, next_stop_sign, actor_list
    ):
        """Handles the behavior of the ego vehicle when approaching a stop sign.

        Args:
            ego_location (carla.Location): The location of the ego vehicle.
            ego_speed (float): The current speed of the ego vehicle.
            target_speed (float): The current target speed of the ego vehicle.
            next_stop_sign (carla.TrafficSign or None): The next stop sign in the route.
            actor_list (list): A list of all actors (vehicles, walkers, etc.) in the simulation.

        Returns:
            float: The adjusted target speed for the ego vehicle.
        """
        stop_signs = get_nearby_objects(
            self.ego_vehicle,
            actor_list.filter("*traffic.stop*"),
            self.stop_sign_radius,
        )

        for stop_sign in stop_signs:
            center_bbox_stop_sign = stop_sign.get_transform().transform(
                stop_sign.trigger_volume.location
            )

            stop_sign_extent = carla.Vector3D(1.5, 1.5, 0.5)
            bbox_stop_sign = carla.BoundingBox(center_bbox_stop_sign, stop_sign_extent)
            rotation_stop_sign = stop_sign.get_transform().rotation
            bbox_stop_sign.rotation = carla.Rotation(
                pitch=rotation_stop_sign.pitch,
                yaw=rotation_stop_sign.yaw,
                roll=rotation_stop_sign.roll,
            )

            affect_ego = (
                next_stop_sign is not None
                and next_stop_sign.id == stop_sign.id
                and not self.cleared_stop_sign
            )

            if affect_ego:
                self.stop_sign_bbox.append(bbox_stop_sign)

            if self.debug:
                color = carla.Color(0, 1, 0) if affect_ego else carla.Color(1, 0, 0)
                self.world.debug.draw_box(
                    box=bbox_stop_sign,
                    rotation=bbox_stop_sign.rotation,
                    thickness=0.1,
                    color=color,
                    life_time=(1.0 / self.frame_rate_carla) + 1e-6,
                )

        if next_stop_sign is None:
            # No stop sign, continue with the current target speed
            return target_speed

        # Calculate the accurate distance to the stop sign
        distance_stop_sign = (
            next_stop_sign.get_transform()
            .transform(next_stop_sign.trigger_volume.location)
            .distance(ego_location)
        )

        # Reset the stop sign flag if we are farther than 10m away
        if distance_stop_sign > self.distance_stop_sign_unclear:
            self.cleared_stop_sign = False
        else:
            # Set the stop sign flag if we are closer than 3m and speed is low enough
            if ego_speed < 0.1 and distance_stop_sign < self.distance_stop_sign_clear:
                self.cleared_stop_sign = True

        # Set the distance to stop sign as infinity if the stop sign has been cleared
        distance_stop_sign = np.inf if self.cleared_stop_sign else distance_stop_sign

        # Compute the target speed using the IDM
        target_speed = self._get_speed_idm(
            ego_speed,
            target_speed,
            0.0,
            0.0,
            distance_stop_sign,
            s0=self.idm_min_distance_stop_sign,
            T=self.idm_time_stop_sign,
        )

        # Return whether the ego vehicle is affected by the stop sign and the adjusted target speed
        return target_speed

    def _get_speed_brake_and_target(
        self,
        target_speed,
        actor_list,
        vehicle_list,
        route_points,
        distance_next_traffic_light,
        next_traffic_light,
        distance_next_stop_sign,
        next_stop_sign,
        speed_reduced_by_obj,
    ):
        """Compute the brake command and target speed for the ego vehicle based on various factors.

        Args:
            target_speed (float): The target speed for the ego vehicle.
            actor_list (list): A list of all actors (vehicles, walkers, etc.) in the simulation.
            vehicle_list (list): A list of vehicle actors in the simulation.
            route_points (numpy.ndarray): An array of waypoints representing the planned route.
            distance_to_next_traffic_light (float): The distance to the next traffic light.
            next_traffic_light (carla.TrafficLight): The next traffic light actor.
            distance_to_next_stop_sign (float): The distance to the next stop sign.
            next_stop_sign (carla.StopSign): The next stop sign actor.
            speed_reduced_by_obj (list or None): A list containing [reduced speed, object type, object ID, distance] for the object that caused the most speed reduction, or None if no speed reduction.
        """
        initial_target_speed = target_speed
        ego_location = self.ego_vehicle.get_location()
        ego_transform = self.ego_vehicle.get_transform()
        ego_speed = self.ego_vehicle.get_velocity().length()

        # Calculate the global bounding box of the ego vehicle
        center_ego_bbox_global = ego_transform.transform(
            self.ego_vehicle.bounding_box.location
        )
        ego_bbox_global = carla.BoundingBox(
            center_ego_bbox_global, self.ego_vehicle.bounding_box.extent
        )
        ego_bbox_global.rotation = ego_transform.rotation

        if self.debug:
            self.world.debug.draw_box(
                box=ego_bbox_global,
                rotation=ego_bbox_global.rotation,
                thickness=0.1,
                color=self.color_bbox_ego_vehicle,
                life_time=self.draw_life_time,
            )

        # Reset hazard flags
        self._reset_flags()

        # Compute if there will be a lane change close
        near_lane_change = is_near_lane_change(
            self.ego_vehicle,
            route_points,
            self.waypoint_planner.route_index,
            self.waypoint_planner.commands,
            safety_distance=self.safety_distance_addition_to_braking_distance,
            min_point_lookahead=self.min_lookahead_near_lane_change,
            min_point_previous=self.min_previous_near_lane_change,
            points_per_meter=self.points_per_meter,
        )

        # Compute the number of future frames to consider for collision detection
        num_future_frames = int(
            self.frame_rate_bicycle
            * (
                self.forecast_duration_lane_change
                if near_lane_change
                else self.forecast_duration_default
            )
        )

        # Get future bounding boxes of walkers
        nearby_walkers, nearby_walker_ids = self.forecast_walkers(
            ego_location, actor_list, num_future_frames
        )

        # Forecast the ego vehicle's bounding boxes for the future frames
        ego_bbox = self.forecast_ego_agent(
            ego_transform,
            ego_speed,
            initial_target_speed,
            num_future_frames,
            route_points,
        )

        # Forecast the ego vehicle's bounding boxes for the future frames
        predicted_bboxes = self.predict_bboxes(
            vehicle_list, near_lane_change, num_future_frames
        )

        # Compute the leading and rear vehicle IDs
        leading_vehicle_ids = self.waypoint_planner.compute_leading_vehicles(
            vehicle_list, self.ego_vehicle.id
        )
        rear_vehicle_ids = self.waypoint_planner.compute_rear_vehicles(
            vehicle_list, self.ego_vehicle.id
        )

        # Compute the target speed with respect to the leading vehicle
        (
            target_speed_leading,
            speed_reduced_by_obj,
        ) = self._get_speed_wrt_leading_vehicle(
            ego_location,
            initial_target_speed,
            predicted_bboxes,
            near_lane_change,
            leading_vehicle_ids,
            rear_vehicle_ids,
            speed_reduced_by_obj,
        )

        # Compute the target speeds with respect to all actors (vehicles, bicycles, walkers)
        (
            target_speed_bicycle,
            target_speed_walker,
            target_speed_vehicle,
            speed_reduced_by_obj,
        ) = self._get_speed_wrt_all_actors(
            initial_target_speed,
            ego_bbox,
            predicted_bboxes,
            near_lane_change,
            leading_vehicle_ids,
            rear_vehicle_ids,
            speed_reduced_by_obj,
            nearby_walkers,
            nearby_walker_ids,
        )

        # Compute the target speed with respect to the red light
        target_speed_traffic_light = self._get_speed_affected_by_traffic_light(
            ego_location,
            ego_speed,
            initial_target_speed,
            distance_next_traffic_light,
            next_traffic_light,
        )

        # Update the object causing the most speed reduction
        if (
            speed_reduced_by_obj is None
            or speed_reduced_by_obj[0] > target_speed_traffic_light
        ):
            speed_reduced_by_obj = [
                target_speed_traffic_light,
                None if next_traffic_light is None else next_traffic_light.type_id,
                None if next_traffic_light is None else next_traffic_light.id,
                distance_next_traffic_light,
            ]

        # Compute the target speed with respect to the stop sign
        target_speed_stop_sign = self._get_speed_affected_by_stop_sign(
            ego_location, ego_speed, initial_target_speed, next_stop_sign, actor_list
        )

        # Update the object causing the most speed reduction
        if (
            speed_reduced_by_obj is None
            or speed_reduced_by_obj[0] > target_speed_stop_sign
        ):
            speed_reduced_by_obj = [
                target_speed_stop_sign,
                None if next_stop_sign is None else next_stop_sign.type_id,
                None if next_stop_sign is None else next_stop_sign.id,
                distance_next_stop_sign,
            ]

        # Compute the minimum target speed considering all factors
        target_speed = min(
            target_speed_leading,
            target_speed_bicycle,
            target_speed_vehicle,
            target_speed_walker,
            target_speed_traffic_light,
            target_speed_stop_sign,
        )

        # Set the hazard flags based on the target speed and its cause
        if (
            target_speed == target_speed_walker
            and target_speed_walker != initial_target_speed
        ):
            self.walker_hazard = True
            self.walker_close = True
        elif (
            target_speed == target_speed_traffic_light
            and target_speed_traffic_light != initial_target_speed
        ):
            self.traffic_light_hazard = True
        elif (
            target_speed == target_speed_stop_sign
            and target_speed_stop_sign != initial_target_speed
        ):
            self.stop_sign_hazard = True
            self.stop_sign_close = True

        # Determine if the ego vehicle needs to brake based on the target speed
        brake = target_speed == 0
        return brake, target_speed, speed_reduced_by_obj

    def _get_steer(
        self, route_points, current_position, current_heading, current_speed
    ):
        """Calculate the steering angle based on the current position, heading, speed, and the route points.

        Args:
            route_points (numpy.ndarray): An array of (x, y) coordinates representing the route points.
            current_position (tuple): The current position (x, y) of the vehicle.
            current_heading (float): The current heading angle (in radians) of the vehicle.
            current_speed (float): The current speed of the vehicle (in m/s).

        Returns:
            float: The calculated steering angle.
        """
        speed_scale = self.lateral_pid_controller_speed_scale
        speed_offset = self.lateral_pid_controller_speed_offset

        # Calculate the lookahead index based on the current speed
        speed_in_kmph = current_speed * 3.6
        lookahead_distance = speed_scale * speed_in_kmph + speed_offset
        lookahead_distance = np.clip(
            lookahead_distance,
            self.lateral_pid_controller_default_lookahead,
            self.lateral_pid_controller_max_lookahead_distance,
        )
        lookahead_index = int(min(lookahead_distance, route_points.shape[0] - 1))

        # Get the target point from the route points
        self.target_wp = route_points[lookahead_index]

        # Calculate the angle between the current heading and the target point
        angle = get_angle_by_position(current_position, current_heading, self.target_wp)
        self.angle = angle / np.pi  # Normalize the angle to [-1, 1]

        # Calculate the steering angle using the turn controller
        steering_angle = self.lateral_controller.get_steering(
            route_points, current_speed, current_position, current_heading
        )
        steering_angle = round(steering_angle, 3)

        return steering_angle

    def predict_bboxes(self, actor_list, near_lane_change, num_future_frames):
        """Predict the future bounding boxes of actors for a given number of frames.

        Args:
            actor_list (list): A list of actors (e.g., vehicles) in the simulation.
            near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.
            num_future_frames (int): The number of future frames to predict.

        Returns:
            dict: A dictionary mapping actor IDs to lists of predicted bounding boxes for each future frame.
        """
        predicted_bboxes = {}
        ego_location = self.ego_vehicle.get_location()

        # Filter out nearby actors within the detection radius, excluding the ego vehicle
        nearby_actors = [
            actor
            for actor in actor_list
            if actor.id != self.ego_vehicle.id
            and actor.get_location().distance(ego_location) < self.detection_radius
        ]

        # If there are nearby actors, calculate their future bounding boxes
        if nearby_actors:
            # Get the previous control inputs (steering, throttle, brake) for the nearby actors
            previous_controls = [actor.get_control() for actor in nearby_actors]
            previous_actions = np.array(
                [
                    [control.steer, control.throttle, control.brake]
                    for control in previous_controls
                ]
            )

            # Get the current velocities, locations, and headings of the nearby actors
            velocities = np.array(
                [actor.get_velocity().length() for actor in nearby_actors]
            )
            locations = np.array(
                [
                    [
                        actor.get_location().x,
                        actor.get_location().y,
                        actor.get_location().z,
                    ]
                    for actor in nearby_actors
                ]
            )
            headings = np.deg2rad(
                np.array(
                    [actor.get_transform().rotation.yaw for actor in nearby_actors]
                )
            )

            # Initialize arrays to store future locations, headings, and velocities
            future_locations = np.empty(
                (num_future_frames, len(nearby_actors), 3), dtype=np.float64
            )
            future_headings = np.empty(
                (num_future_frames, len(nearby_actors)), dtype=np.float64
            )
            future_velocities = np.empty(
                (num_future_frames, len(nearby_actors)), dtype=np.float64
            )

            # Forecast the future locations, headings, and velocities for the nearby actors
            for i in range(num_future_frames):
                (
                    locations,
                    headings,
                    velocities,
                ) = self.vehicle_physics_model.forecast_other_vehicles(
                    locations, headings, velocities, previous_actions
                )
                future_locations[i] = locations.copy()
                future_velocities[i] = velocities.copy()
                future_headings[i] = headings.copy()

            # Convert future headings to degrees
            future_headings = np.rad2deg(future_headings)

            # Calculate the predicted bounding boxes for each nearby actor and future frame
            for actor_idx, actor in enumerate(nearby_actors):
                predicted_actor_boxes = []

                for i in range(num_future_frames):
                    # Calculate the future location of the actor
                    location = carla.Location(
                        x=future_locations[i, actor_idx, 0].item(),
                        y=future_locations[i, actor_idx, 1].item(),
                        z=future_locations[i, actor_idx, 2].item(),
                    )

                    # Calculate the future rotation of the actor
                    rotation = carla.Rotation(
                        pitch=0, yaw=future_headings[i, actor_idx], roll=0
                    )

                    # Get the extent (dimensions) of the actor's bounding box
                    extent = actor.bounding_box.extent
                    # Otherwise we would increase the extent of the bounding box of the vehicle
                    extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)

                    # Adjust the bounding box size based on velocity and lane change maneuver to adjust for uncertainty during forecasting
                    s = (
                        self.min_extent_factor_x_other_vehicle_lane_change
                        if near_lane_change
                        else self.min_extent_factor_x_other_vehicle
                    )
                    extent.x *= (
                        self.extent_factor_x_ego_low_speed
                        if future_velocities[i, actor_idx]
                        < self.speed_threshold_other_vehicle
                        else max(
                            s,
                            self.min_extent_factor_x_other_vehicle
                            * float(i)
                            / float(num_future_frames),
                        )
                    )
                    extent.y *= (
                        self.extent_factor_y_ego_low_speed
                        if future_velocities[i, actor_idx]
                        < self.speed_threshold_other_vehicle
                        else max(
                            self.min_extent_factor_y_other_vehicle,
                            self.extent_factor_y_other_vehicle
                            * float(i)
                            / float(num_future_frames),
                        )
                    )

                    # Create the bounding box for the future frame
                    bbox = carla.BoundingBox(location, extent)
                    bbox.rotation = rotation

                    # Append the bounding box to the list of predicted bounding boxes for this actor
                    predicted_actor_boxes.append(bbox)

                # Store the predicted bounding boxes for this actor in the dictionary
                predicted_bboxes[actor.id] = predicted_actor_boxes

                if self.debug:
                    for (
                        actor_idx,
                        actors_forecasted_bboxes,
                    ) in predicted_bboxes.items():
                        for bbox in actors_forecasted_bboxes:
                            self.world.debug.draw_box(
                                box=bbox,
                                rotation=bbox.rotation,
                                thickness=0.1,
                                color=self.color_bbox_forecasted_other_vehicle,
                                life_time=self.draw_life_time,
                            )

        return predicted_bboxes

    def forecast_ego_agent(
        self, ego_transform, ego_speed, target_speed, num_future_frames, route_points
    ):
        """Forecast the future states of the ego agent using the kinematic bicycle model and assume their is no hazard to check subsequently whether the ego vehicle would collide.

        Args:
            ego_transform (carla.Transform): The current transform of the ego vehicle.
            ego_speed (float): The current speed of the ego vehicle
            target_speed (float): The initial target speed for the ego vehicle.
            num_future_frames (int): The number of future frames to forecast.
            route_points (numpy.ndarray): An array of waypoints representing the planned route.

        Returns:
            list: A list of bounding boxes representing the future states of the ego vehicle.
        """
        self.lateral_controller.save()
        self.waypoint_planner.save()

        # Initialize the initial state without braking
        location = np.array(
            [
                ego_transform.location.x,
                ego_transform.location.y,
                ego_transform.location.z,
            ]
        )

        heading_angle = np.deg2rad(ego_transform.rotation.yaw)
        speed = ego_speed

        # Calculate the throttle command based on the target speed and current speed
        throttle = self.longitudinal_controller.get_throttle_extrapolation(
            target_speed, ego_speed
        )
        steering = self.lateral_controller.get_steering(
            route_points, speed, location, heading_angle.item()
        )
        action = np.array([steering, throttle, 0.0]).flatten()

        future_bboxes = []
        # Iterate over the future frames and forecast the ego agent's state
        for _ in range(num_future_frames):
            # Forecast the next state using the kinematic bicycle model
            (
                location,
                heading_angle,
                speed,
            ) = self.ego_physics_model.forecast_ego_vehicle(
                location, heading_angle, speed, action
            )

            # Update the route and extrapolate steering and throttle commands
            extrapolated_route, _, _, _, _, _, _, _ = self.waypoint_planner.run_step(
                location
            )
            steering = self.lateral_controller.get_steering(
                extrapolated_route, speed, location, heading_angle.item()
            )
            throttle = self.longitudinal_controller.get_throttle_extrapolation(
                target_speed, speed
            )
            action = np.array([steering, throttle, 0.0]).flatten()

            heading_angle_degrees = np.rad2deg(heading_angle).item()

            # Decrease the ego vehicles bounding box if it is slow and resolve permanent bounding box intersections at collisions.
            # In case of driving increase them for safety.
            extent = self.ego_vehicle.bounding_box.extent
            # Otherwise we would increase the extent of the bounding box of the vehicle
            extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)
            extent.x *= (
                self.extent_factor_x_ego_low_speed
                if ego_speed < self.speed_threshold_ego
                else self.extent_factor_x_ego_high_speed
            )
            extent.y *= (
                self.extent_factor_y_ego_low_speed
                if ego_speed < self.speed_threshold_ego
                else self.extent_factor_y_ego_high_speed
            )

            transform = carla.Transform(
                carla.Location(
                    x=location[0].item(), y=location[1].item(), z=location[2].item()
                )
            )

            ego_bbox = carla.BoundingBox(transform.location, extent)
            ego_bbox.rotation = carla.Rotation(
                pitch=0, yaw=heading_angle_degrees, roll=0
            )

            future_bboxes.append(ego_bbox)

        self.lateral_controller.load()
        self.waypoint_planner.load()

        return future_bboxes

    def forecast_walkers(self, ego_location, actors, num_future_frames):
        """
        Forecast the future locations of walkers in the vicinity of the ego vehicle assuming they
        keep their velocity and direction

        Args:
            ego_location (carla.Location): The location of the ego vehicle.
            actors (carla.ActorList): A list of actors in the simulation.
            num_future_frames (int): The number of future frames to forecast.

        Returns:
            tuple: A tuple containing two lists:
                - list: A list of lists, where each inner list contains the future bounding boxes for a walker.
                - list: A list of IDs for the walkers whose locations were forecasted.
        """
        nearby_walkers_bboxes, nearby_walker_ids = [], []

        # Filter walkers within the detection radius
        walkers = [
            walker
            for walker in actors.filter("*walker*")
            if walker.get_location().distance(ego_location) < self.detection_radius
        ]

        # If no walkers are found, return empty lists
        if not walkers:
            return nearby_walkers_bboxes, nearby_walker_ids

        # Extract walker locations, speeds, and directions
        walker_locations = np.array(
            [
                [
                    walker.get_location().x,
                    walker.get_location().y,
                    walker.get_location().z,
                ]
                for walker in walkers
            ]
        )
        walker_speeds = np.array([walker.get_velocity().length() for walker in walkers])
        walker_speeds = np.maximum(walker_speeds, self.min_speed_walker)
        walker_directions = np.array(
            [
                [
                    walker.get_control().direction.x,
                    walker.get_control().direction.y,
                    walker.get_control().direction.z,
                ]
                for walker in walkers
            ]
        )

        # Calculate future walker locations based on their current locations, speeds, and directions
        future_walker_locations = (
            walker_locations[:, None, :]
            + np.arange(1, num_future_frames + 1)[None, :, None]
            * walker_directions[:, None, :]
            * walker_speeds[:, None, None]
            / self.frame_rate_bicycle
        )

        # Iterate over walkers and calculate their future bounding boxes
        for i, walker in enumerate(walkers):
            bbox, transform = walker.bounding_box, walker.get_transform()
            rotation = carla.Rotation(
                pitch=bbox.rotation.pitch + transform.rotation.pitch,
                yaw=bbox.rotation.yaw + transform.rotation.yaw,
                roll=bbox.rotation.roll + transform.rotation.roll,
            )
            extent = bbox.extent
            extent.x = max(self.min_extent_walker, extent.x)  # Ensure a minimum width
            extent.y = max(self.min_extent_walker, extent.y)  # Ensure a minimum length

            walker_future_bboxes = []
            for j in range(num_future_frames):
                location = carla.Location(
                    future_walker_locations[i, j, 0],
                    future_walker_locations[i, j, 1],
                    future_walker_locations[i, j, 2],
                )

                bbox = carla.BoundingBox(location, extent)
                bbox.rotation = rotation
                walker_future_bboxes.append(bbox)

            nearby_walker_ids.append(walker.id)
            nearby_walkers_bboxes.append(walker_future_bboxes)

        # Visualize the future bounding boxes of walkers (if enabled)
        if self.debug:
            for bboxes in nearby_walkers_bboxes:
                for bbox in bboxes:
                    self.world.debug.draw_box(
                        box=bbox,
                        rotation=bbox.rotation,
                        thickness=0.1,
                        color=self.color_bbox_forecasted_walker,
                        life_time=self.draw_life_time,
                    )

        return nearby_walkers_bboxes, nearby_walker_ids
