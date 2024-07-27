##!/usr/bin/env python3
# @File: data_agent.py
# @Description:
# @CreatedTime: 2024/07/26
# @Author: Yueyuan Li

import json
import logging
import os

import wandb
from omegaconf import OmegaConf
from srunner.scenariomanager.timer import GameTime

from src.expert_agent import ExpertAgent
from src.utils.common import get_absolute_path
from src.utils.visualizer import Visualizer


def get_entry_point():
    return "DataAgent"


class DataAgent(ExpertAgent):
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

        if self.save_control_command_to_wandb and wandb.run is not None:
            wandb.log(
                {
                    "steer": control.steer,
                    "throttle": control.throttle,
                    "brake": control.brake,
                }
            )

        if self.save_runtime_to_wandb and wandb.run is not None:
            wandb.log(
                {
                    "system_time": round(wallclock_diff, 3),
                    "game_time": round(timestamp, 3),
                    "sim_ratio": round(sim_ratio, 3),
                }
            )

        return control

    def setup(self, path_to_conf_file: str):
        self.configs = OmegaConf.load(get_absolute_path(path_to_conf_file))
        self.configs = OmegaConf.to_container(self.configs, resolve=True)
        for key, value in self.configs.items():
            setattr(self, key, value)

        super().setup(get_absolute_path(self.path_agent_configs))

        self.sensor_configs = OmegaConf.load(
            get_absolute_path(self.path_sensor_configs)
        )
        self.sensor_configs = OmegaConf.to_container(self.sensor_configs, resolve=True)

        if self.save_control_command:
            self.control_commands = {
                "steer": [],
                "throttle": [],
                "brake": [],
                "hand_brake": [],
                "reverse": [],
                "manual_gear_shift": [],
                "gear": [],
            }

        # save data
        self.path_save = get_absolute_path(self.path_save)
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save)

        for sensor_id in self.sensor_ids:
            path_data = os.path.join(self.path_save, sensor_id)
            if not os.path.exists(path_data):
                os.makedirs(path_data)

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

        if self.save_runtime_to_wandb:
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
            )

    def sensors(self):
        sensors = super().sensors()

        for sensor_id in self.sensor_ids:
            if sensor_id in self.sensor_configs:
                sensors.append(self.sensor_configs[sensor_id])
            else:
                logging.warning(f"No sensor names {sensor_id}. Skip.")

        return sensors

    def cache_control_command(self, control):
        self.control_commands["steer"].append(control.steer)
        self.control_commands["throttle"].append(control.throttle)
        self.control_commands["brake"].append(control.brake)
        self.control_commands["hand_brake"].append(control.hand_brake)
        self.control_commands["reverse"].append(control.reverse)
        self.control_commands["manual_gear_shift"].append(control.manual_gear_shift)
        self.control_commands["gear"].append(control.gear)

    def save_image(self, image, folder_name):
        pass

    def save_lidar(self, lidar, folder_name):
        pass

    def run_step(self, input_data):
        control = super().run_step(input_data)

        # add data saver
        if self.save_control_command:
            self.cache_control_command(control)

        if self.visualize:
            self.visualizer.update(input_data)

        return control

    def destroy(self):
        super().destroy()

        if self.visualize:
            self.visualizer.quit()

        if wandb.run is not None:
            wandb.finish()

        if self.save_control_command:
            records = {"records": []}
            len_record = len(self.control_commands["steer"])
            for i in range(len_record):
                records["records"].append(
                    {
                        "control": {
                            "steer": self.control_commands["steer"][i],
                            "throttle": self.control_commands["throttle"][i],
                            "brake": self.control_commands["brake"][i],
                            "hand_brake": self.control_commands["hand_brake"][i],
                            "reverse": self.control_commands["reverse"][i],
                            "manual_gear_shift": self.control_commands[
                                "manual_gear_shift"
                            ][i],
                            "gear": self.control_commands["gear"][i],
                        }
                    }
                )

            with open(os.path.join(self.path_save, "log.json"), "w") as f:
                json.dump(records, f)
