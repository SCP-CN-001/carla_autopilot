##!/usr/bin/env python3
# @File: data_agent.py
# @Description:
# @CreatedTime: 2024/07/26
# @Author: Yueyuan Li

import logging
import os
import pickle as pkl
import threading
import time

import carla
import numpy as np
import wandb
from carla_autopilot.expert_agent import ExpertAgent
from carla_autopilot.utils.common import get_absolute_path, get_random_weather
from carla_autopilot.utils.geometry import get_transform_2D
from carla_autopilot.utils.visualizer import Visualizer
from omegaconf import OmegaConf
from srunner.scenariomanager.timer import GameTime


def get_entry_point():
    return "DataAgent"


class DataAgent(ExpertAgent):
    """This agent is used to collect sensor data and privileged information from the CARLA simulator.

    The general idea is to save information at each step in a pickle file.
    """

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

        t1 = time.time()
        control = self.run_step(input_data)
        t2 = time.time()
        control.manual_gear_shift = False

        if wandb.run is not None:
            wandb.log(
                {
                    "system_time": round(wallclock_diff, 3),
                    "game_time": round(timestamp, 3),
                    "sim_ratio": round(sim_ratio, 3),
                    "target_speed": self.target_speed,
                    "ego_speed": self.ego_vehicle.get_velocity().length(),
                    "step_time": round(t2 - t1, 3),
                }
            )

        return control

    def setup(self, path_to_conf_file: str):
        self.configs = OmegaConf.load(get_absolute_path(path_to_conf_file))
        self.configs = OmegaConf.to_container(self.configs, resolve=True)
        for key, value in self.configs.items():
            setattr(self, key, value)

        super().setup(get_absolute_path(self.path_agent_configs))

        # Set a random weather
        # TODO: This seems not working
        weather_params = get_random_weather(from_file=True)
        weather = carla.WeatherParameters(**weather_params)
        self.world.set_weather(weather)
        self.world.tick()

        self.sensor_configs = OmegaConf.load(
            get_absolute_path(self.path_sensor_configs)
        )
        self.sensor_configs = OmegaConf.to_container(self.sensor_configs, resolve=True)

        # Create a folder to save the data
        self.path_save = get_absolute_path(self.path_save)
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save)

        # Currently, CARLA's lidar can only run at 10Hz or 20Hz. If the lidar is running at 20Hz, it can only collect data from half of the full view. The other half will be collected at the next step.
        self.lidar_id_list = []
        self.lidar_cache = {}
        for sensor_id, sensor_config in self.sensor_configs.items():
            if sensor_id not in self.sensor_ids:
                continue
            if sensor_config["blueprint"] == "sensor.lidar.ray_cast":
                self.lidar_id_list.append(sensor_id)
                self.lidar_cache[sensor_id] = None

        # Setup the visualizer
        if self.visualize:
            logging.info("This simulation is in visualize mode.")
            self.visualize_configs = OmegaConf.load(
                get_absolute_path(self.path_visualizer_configs)
            )
            self.visualize_configs = OmegaConf.to_container(
                self.visualize_configs, resolve=True
            )

            self.visualizer = Visualizer(self.visualize_configs)
        else:
            logging.info("This simulation is not in visualize mode.")

        weather = self.world.get_weather()
        wandb.init(
            project=self.project,
            name=f"{self.name}_route_{self.route_subset}",
            config={
                "map": self.world_map.name,
                "route": self.route,
                "route_subset": self.route_subset,
                "weather_sun_azimuth": weather.sun_azimuth_angle,
                "weather_sun_altitude": weather.sun_altitude_angle,
                "weather_cloudiness": weather.cloudiness,
                "weather_precipitation": weather.precipitation,
                "weather_precipitation_deposits": weather.precipitation_deposits,
                "weather_wind_intensity": weather.wind_intensity,
                "weather_fog_density": weather.fog_density,
            },
            mode=self.wandb_mode,
        )

    def sensors(self):
        sensors = super().sensors()

        for sensor_id in self.sensor_ids:
            if sensor_id in self.sensor_configs:
                sensors.append(self.sensor_configs[sensor_id])
            else:
                logging.warning(f"No sensor names {sensor_id}. Skip.")

        return sensors

    def get_ego_state_dict(self):
        ego_state = {
            "velocity": {
                "value": self.ego_vehicle.get_velocity().length(),
                "x": self.ego_vehicle.get_velocity().x,
                "y": self.ego_vehicle.get_velocity().y,
                "z": self.ego_vehicle.get_velocity().z,
            },
            "acceleration": {
                "value": self.ego_vehicle.get_acceleration().length(),
                "x": self.ego_vehicle.get_acceleration().x,
                "y": self.ego_vehicle.get_acceleration().y,
                "z": self.ego_vehicle.get_acceleration().z,
            },
            "transform": {
                "x": self.ego_vehicle.get_location().x,
                "y": self.ego_vehicle.get_location().y,
                "z": self.ego_vehicle.get_location().z,
                "yaw": self.ego_vehicle.get_transform().rotation.yaw,
                "pitch": self.ego_vehicle.get_transform().rotation.pitch,
                "roll": self.ego_vehicle.get_transform().rotation.roll,
            },
        }
        return ego_state

    def get_ego_action_dict(self):
        ego_action = {
            "steer": self.control.steer,
            "throttle": self.control.throttle,
            "brake": self.control.brake,
            "hand_brake": self.control.hand_brake,
            "reverse": self.control.reverse,
            "manual_gear_shift": self.control.manual_gear_shift,
            "gear": self.control.gear,
        }
        return ego_action

    def get_route_points(self):
        ego_location = self.ego_vehicle.get_location()
        ego_transform = self.ego_vehicle.get_transform()

        route_points = []
        len_route_draw = 10
        for route_point in self.remaining_route_original[:len_route_draw]:
            route_points.append([route_point[0], route_point[1]])

        route_points = get_transform_2D(
            np.array(route_points),
            np.array([ego_location.x, ego_location.y]),
            np.deg2rad(ego_transform.rotation.yaw),
        )

        route_points = np.array(route_points) / np.array([10.0, 5.0])
        return route_points

    def get_sensor_data(self, input_data):
        sensor_data = {}
        for sensor_id in self.sensor_ids:
            if self.sensor_configs[sensor_id]["type"] in [
                "sensor.camera.rgb",
                "sensor.camera.semantic_segmentation",
            ]:
                sensor_data[sensor_id] = input_data[sensor_id][1]
            elif self.sensor_configs[sensor_id]["type"] == "sensor.lidar.ray_cast":
                if self.frame_rate_carla == 20:
                    if self.step > 0 and self.step % 2 == 0:
                        sensor_data[sensor_id] = np.concatenate(
                            [
                                np.array(input_data[sensor_id][1]),
                                np.array(self.lidar_cache[sensor_id]),
                            ]
                        )
                        self.lidar_cache[sensor_id] = None
                    elif self.step > 0 and self.step % 2 == 1:
                        self.lidar_cache[sensor_id] = input_data[sensor_id][1]
                elif self.frame_rate_carla == 10:
                    sensor_data[sensor_id] = input_data[sensor_id][1]

        return sensor_data

    def save_data(self, input_data):
        data = self.get_sensor_data(input_data)
        data["ego_state"] = self.get_ego_state_dict()
        data["ego_action"] = self.get_ego_action_dict()

        if self.save_route:
            data["route_points"] = self.get_route_points()

        if self.frame_rate_carla == 10:
            if self.step >= 10 and self.step % 2 == 0:
                path_file = f"{self.path_save}/frame_{self.step:06d}.pkl"
                pkl.dump(data, open(path_file, "wb"))
        elif self.frame_rate_carla == 20:
            pass

    def run_step(self, input_data):
        thread_1 = threading.Thread(target=super().run_step, args=(input_data,))
        thread_2 = threading.Thread(target=self.save_data, args=(input_data,))

        thread_1.start()
        thread_2.start()
        thread_1.join()
        thread_2.join()

        if self.visualize:
            self.visualizer.update(input_data)

        return self.control

    def destroy(self):
        super().destroy()

        if self.visualize:
            self.visualizer.quit()

        if wandb.run is not None:
            wandb.finish()
