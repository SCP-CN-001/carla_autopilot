##!/usr/bin/env python3
# @File: data_agent.py
# @Description:
# @CreatedTime: 2024/07/26
# @Author: Yueyuan Li

import copy
import json
import logging
import os
import pickle as pkl
import threading
import time

import carla
import cv2
import numpy as np
import open3d as o3d
import pygame
import wandb
from omegaconf import OmegaConf
from srunner.scenariomanager.timer import GameTime

from src.common_carla.bounding_box import get_polygon_shape
from src.expert_agent import ExpertAgent
from src.utils.common import get_absolute_path, get_random_weather
from src.utils.geometry import get_transform_2D, get_world_to_ego_to_image_coords
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

        t1 = time.time()
        control = self.run_step(input_data)
        t2 = time.time()
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
        weather_params = get_random_weather(from_file=True)
        weather = carla.WeatherParameters(**weather_params)
        self.world.set_weather(weather)
        self.world.tick()

        self.sensor_configs = OmegaConf.load(
            get_absolute_path(self.path_sensor_configs)
        )
        self.sensor_configs = OmegaConf.to_container(self.sensor_configs, resolve=True)

        self.control_commands = {
            "steer": [],
            "throttle": [],
            "brake": [],
            "hand_brake": [],
            "reverse": [],
            "manual_gear_shift": [],
            "gear": [],
        }
        self.states = {
            "acceleration": {
                "value": [],
                "x": [],
                "y": [],
                "z": [],
            },
            "transform": {
                "x": [],
                "y": [],
                "z": [],
                "yaw": [],
                "pitch": [],
                "roll": [],
            },
            "velocity": {
                "value": [],
                "x": [],
                "y": [],
                "z": [],
            },
            "route_points": {
                "remain": [],
            },
        }

        # save data
        self.path_save = get_absolute_path(self.path_save)
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save)

        for sensor_id in self.sensor_ids:
            path_data = os.path.join(self.path_save, sensor_id)
            if not os.path.exists(path_data):
                os.makedirs(path_data)

        if getattr(self, "save_semantic_bev") is not None:
            self.palette = np.array(
                OmegaConf.load(
                    get_absolute_path(self.save_semantic_bev["path_palette"])
                ).color_map
            )

            for path_save in self.save_semantic_bev["path_save"]:
                for sensor_id in self.save_semantic_bev["sensor_id"]:
                    if not os.path.exists(
                        get_absolute_path(f"{self.path_save}/{sensor_id}/{path_save}")
                    ):
                        os.makedirs(
                            get_absolute_path(
                                f"{self.path_save}/{sensor_id}/{path_save}"
                            )
                        )
                if not os.path.exists(
                    get_absolute_path(f"{self.path_save}/{sensor_id}/raw")
                ):
                    os.makedirs(get_absolute_path(f"{self.path_save}/{sensor_id}/raw"))

        if self.save_route:
            if not os.path.exists(f"{self.path_save}/route"):
                os.makedirs(f"{self.path_save}/route")

            if not os.path.exists(f"{self.path_save}/traffic_sign"):
                os.makedirs(f"{self.path_save}/traffic_sign")

            self.file_route_points = open(
                get_absolute_path(f"{self.path_save}/route_points.csv"), "w"
            )
            self.route_image = pygame.Surface(
                (self.route_image_width, self.route_image_height)
            )
            self.traffic_sign_image = pygame.Surface(
                (self.route_image_width, self.route_image_height)
            )  # TODO: define the traffic sign representation

        self.lidar_id_list = []
        self.lidar_cache = {}
        for sensor_id, sensor_config in self.sensor_configs.items():
            if sensor_id not in self.sensor_ids:
                continue
            if sensor_config["blueprint"] == "sensor.lidar.ray_cast":
                self.lidar_id_list.append(sensor_id)
                self.lidar_cache[sensor_id] = None

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

        self.route_counter = 0
        if self.save_runtime_to_wandb:
            if self.route_subset == "all":
                route_subset = self.route_counter
                self.route_counter += 1
            else:
                route_subset = self.route_subset
            weather = self.world.get_weather()
            wandb.init(
                project=self.project,
                name=f"{self.name}_route_{route_subset}",
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

    def cache_state(self):
        ego_location = self.ego_vehicle.get_location()
        ego_transform = self.ego_vehicle.get_transform()
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_acceleration = self.ego_vehicle.get_acceleration()

        self.states["acceleration"]["value"].append(ego_acceleration.length())
        self.states["acceleration"]["x"].append(ego_acceleration.x)
        self.states["acceleration"]["y"].append(ego_acceleration.y)
        self.states["acceleration"]["z"].append(ego_acceleration.z)

        self.states["transform"]["x"].append(ego_location.x)
        self.states["transform"]["y"].append(ego_location.y)
        self.states["transform"]["z"].append(ego_location.z)
        self.states["transform"]["yaw"].append(ego_transform.rotation.yaw)
        self.states["transform"]["pitch"].append(ego_transform.rotation.pitch)
        self.states["transform"]["roll"].append(ego_transform.rotation.roll)

        self.states["velocity"]["value"].append(ego_velocity.length())
        self.states["velocity"]["x"].append(ego_velocity.x)
        self.states["velocity"]["y"].append(ego_velocity.y)
        self.states["velocity"]["z"].append(ego_velocity.z)

        self.states["route_points"]["remain"].append(len(self.remaining_route_original))

    def save_image(self, image, folder_name):
        path_file = f"{self.path_save}/{folder_name}/frame_{self.step:06d}.png"
        cv2.imwrite(path_file, image)

    def save_lidar(self, lidar, folder_name):
        path_file = f"{self.path_save}/{folder_name}/frame_{self.step:06d}.ply"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar[:, :3])
        o3d.io.write_point_cloud(path_file, pcd)

    def save_semantic_segmentation(self, semantic_segmentation, folder_name):
        if getattr(self, "palette") is None:
            return

        images = self.palette[semantic_segmentation[:, :, -2]]
        raw_image = semantic_segmentation[:, :, -2].astype(np.uint8)

        assert images.shape[2] == len(
            self.save_semantic_bev["path_save"]
        ), "The number of semantic segmentation classes should be equal to the number of save semantic BEV classes."

        path_file = f"{self.path_save}/{folder_name}/raw/frame_{self.step:06d}.png"
        cv2.imwrite(path_file, raw_image)

        if self.save_semantic_bev["binary"]:
            images = images * 255
            for i, path_save in enumerate(self.save_semantic_bev["path_save"]):
                path_file = f"{self.path_save}/{folder_name}/{path_save}/frame_{self.step:06d}.png"
                cv2.imwrite(path_file, images[:, :, i])
        else:
            path_file = f"{self.path_save}/{folder_name}/frame_{self.step:06d}.png"
            cv2.imwrite(path_file, images)

    def save_sensor_data(self, input_data):
        def execute_save(input_data):
            for sensor_id in self.sensor_ids:
                if self.sensor_configs[sensor_id]["type"] == "sensor.camera.rgb":
                    self.save_image(input_data[sensor_id][1], sensor_id)
                elif (
                    self.sensor_configs[sensor_id]["type"]
                    == "sensor.camera.semantic_segmentation"
                ):
                    self.save_semantic_segmentation(input_data[sensor_id][1], sensor_id)
                elif self.sensor_configs[sensor_id]["type"] == "sensor.lidar.ray_cast":
                    self.save_lidar(input_data[sensor_id][1], sensor_id)

        # Currently, we can only handle two frame rates: 10 and 20
        if self.frame_rate_carla == 10:
            execute_save(input_data)

        elif self.frame_rate_carla == 20:
            if self.step > 0 and self.step % 2 == 0:
                sensor_data = copy.deepcopy(input_data)
                for lidar_id in self.lidar_id_list:
                    points = np.concatenate(
                        [
                            np.array(sensor_data[lidar_id][1]),
                            np.array(self.lidar_cache[lidar_id]),
                        ]
                    )
                    del sensor_data[lidar_id]
                    sensor_data[lidar_id] = (None, points)
                    execute_save(sensor_data)

                    self.lidar_cache[lidar_id] = None

            elif self.step > 0 and self.step % 2 == 1:
                for lidar_id in self.lidar_id_list:
                    self.lidar_cache[lidar_id] = input_data[lidar_id][1]

    def save_route_data(self):
        logging.debug("Saving route data.")
        ego_location = self.ego_vehicle.get_location()
        ego_transform = self.ego_vehicle.get_transform()

        # debug
        # ego_location_ = [ego_location.x, ego_location.y, ego_location.z]
        # ego_transform_ = [ego_transform.rotation.yaw, ego_transform.rotation.pitch, ego_transform.rotation.roll]
        # pkl.dump([self.remaining_route_original[:50], self.sensor_configs, ego_location_, ego_transform_]
        #           , open('/home/rowena/Documents/ramble/data/expert_data/route_5/data/%s.pkl'%self.step, 'wb'))

        self.route_image.fill((0, 0, 0))
        self.traffic_sign_image.fill((0, 0, 0))

        ego_transform_dict = {
            "x": ego_transform.location.x,
            "y": ego_transform.location.y,
            "z": ego_transform.location.z,
            "roll": np.deg2rad(ego_transform.rotation.roll),
            "pitch": np.deg2rad(ego_transform.rotation.pitch),
            "yaw": np.deg2rad(ego_transform.rotation.yaw),
        }

        camera_transform_dict = {
            "x": self.sensor_configs["semantic_bev"]["x"],
            "y": self.sensor_configs["semantic_bev"]["y"],
            "z": self.sensor_configs["semantic_bev"]["z"],
            "roll": np.deg2rad(self.sensor_configs["semantic_bev"]["roll"]),
            "pitch": np.deg2rad(self.sensor_configs["semantic_bev"]["pitch"]),
            "yaw": np.deg2rad(self.sensor_configs["semantic_bev"]["yaw"]),
        }

        route_points_world = []
        len_route_draw = 50
        for route_point in self.remaining_route_original[:len_route_draw]:
            route_points_world.append([route_point[0], route_point[1], route_point[2]])

        route_points = get_transform_2D(
            np.array(route_points_world)[:, :2],
            np.array([ego_location.x, ego_location.y]),
            np.deg2rad(ego_transform.rotation.yaw),
        )

        # save route points to file
        str_route_points = ",".join(f"[{x[0]}, {x[1]}]" for x in route_points)
        self.file_route_points.write(str_route_points + "\n")

        def _get_polygon_shape3d(bbox: carla.BoundingBox):
            diff_x = bbox.rotation.get_forward_vector() * bbox.extent.x
            diff_y = bbox.rotation.get_right_vector() * bbox.extent.y
            diff_z = bbox.rotation.get_up_vector() * bbox.extent.z

            pt1 = bbox.location + diff_x + diff_y + diff_z
            pt2 = bbox.location + diff_x - diff_y + diff_z
            pt3 = bbox.location - diff_x - diff_y + diff_z
            pt4 = bbox.location - diff_x + diff_y + diff_z

            polygon_shape = [
                [pt1.x, pt1.y, pt1.z],
                [pt2.x, pt2.y, pt2.z],
                [pt3.x, pt3.y, pt3.z],
                [pt4.x, pt4.y, pt4.z],
            ]  # we only consider the upper surface
            return polygon_shape

        # concatenate the location of traffic lights and stop signs to route points
        n_traffic_light = len(self.traffic_light_loc)
        n_stop_sign = len(self.stop_sign_bbox)
        other_locations = [[loc.x, loc.y, loc.z] for loc in self.traffic_light_loc]
        for bbox in self.stop_sign_bbox:
            other_locations.extend(_get_polygon_shape3d(bbox))

        assert len(other_locations) == n_traffic_light + n_stop_sign * 4

        points_to_transform = np.array(route_points_world + other_locations)

        # calculate the route points and location of signs in image coordinates
        points_image = get_world_to_ego_to_image_coords(
            points_to_transform,
            ego_transform_dict,
            camera_transform_dict,
            self.sensor_configs["semantic_bev"]["fov"],
            (
                self.sensor_configs["semantic_bev"]["width"],
                self.sensor_configs["semantic_bev"]["height"],
            ),
        )
        route_points_image = points_image[:len_route_draw]
        traffic_light_coords_on_image = points_image[
            len_route_draw : len_route_draw + n_traffic_light
        ]
        stop_sign_coords_on_image = points_image[
            len_route_draw + n_traffic_light :
        ].reshape(n_stop_sign, 4, 2)

        # draw route points
        pygame.draw.lines(
            self.route_image, (255, 255, 255), False, route_points_image, 3
        )

        # draw traffic light and stop sign
        for loc in traffic_light_coords_on_image:
            if loc[0] < route_points_image[0][0]:
                continue
            pygame.draw.circle(self.traffic_sign_image, (255, 255, 255), loc, 10)
        for bbox in stop_sign_coords_on_image:
            if np.min(bbox[:, 0]) < route_points_image[0][0]:
                continue
            pygame.draw.polygon(self.traffic_sign_image, (255, 255, 255), bbox)

        pygame.image.save(
            self.route_image, f"{self.path_save}/route/frame_{self.step:06d}.png"
        )
        pygame.image.save(
            self.traffic_sign_image,
            f"{self.path_save}/traffic_sign/frame_{self.step:06d}.png",
        )

    def run_step(self, input_data):
        thread_1 = threading.Thread(target=super().run_step, args=(input_data,))
        thread_2 = threading.Thread(target=self.save_sensor_data, args=(input_data,))

        thread_1.start()
        thread_2.start()
        thread_1.join()
        thread_2.join()

        if self.save_route:
            if self.frame_rate_carla == 10:
                self.save_route_data()
            elif self.frame_rate_carla == 20:
                if self.step % 2 == 0:
                    self.save_route_data()

        if self.visualize:
            self.visualizer.update(input_data)

        if self.frame_rate_carla == 10:
            self.cache_state()
            self.cache_control_command(self.control)
        elif self.frame_rate_carla == 20:
            if self.step % 2 == 0:
                self.cache_state()
                self.cache_control_command(self.control)

        return self.control

    def destroy(self):
        super().destroy()

        if self.visualize:
            self.visualizer.quit()

        if wandb.run is not None:
            if self.save_control_command_to_wandb:
                steer_table = wandb.Table(
                    data=[[x] for x in self.control_commands["steer"]],
                    columns=["steer"],
                )
                wandb.log(
                    {
                        "steer_distribution": wandb.plot.histogram(
                            steer_table, "steer"
                        ),
                    }
                )
            wandb.finish()

        if self.save_route:
            self.file_route_points.close()

        # if self.log_format == "json":
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
                        "manual_gear_shift": self.control_commands["manual_gear_shift"][
                            i
                        ],
                        "gear": self.control_commands["gear"][i],
                    },
                    "state": {
                        "acceleration": {
                            "value": self.states["acceleration"]["value"][i],
                            "x": self.states["acceleration"]["x"][i],
                            "y": self.states["acceleration"]["y"][i],
                            "z": self.states["acceleration"]["z"][i],
                        },
                        "transform": {
                            "x": self.states["transform"]["x"][i],
                            "y": self.states["transform"]["y"][i],
                            "z": self.states["transform"]["z"][i],
                            "yaw": self.states["transform"]["yaw"][i],
                            "pitch": self.states["transform"]["pitch"][i],
                            "roll": self.states["transform"]["roll"][i],
                        },
                        "velocity": {
                            "value": self.states["velocity"]["value"][i],
                            "x": self.states["velocity"]["x"][i],
                            "y": self.states["velocity"]["y"][i],
                            "z": self.states["velocity"]["z"][i],
                        },
                    },
                    "route_points": {
                        "remain": self.states["route_points"]["remain"][i],
                    },
                }
            )

        with open(os.path.join(self.path_save, "log.json"), "w") as f:
            json.dump(records, f, indent=4)
        # elif self.log_format == "csv":
        with open(os.path.join(self.path_save, "log.csv"), "w") as f:
            f.write(
                "steer,throttle,brake,hand_brake,reverse,manual_gear_shift,gear,acceleration_value,acceleration_x,acceleration_y,acceleration_z,transform_x,transform_y,transform_z,transform_yaw,transform_pitch,transform_roll,velocity_value,velocity_x,velocity_y,velocity_z,remain_route_point\n"
            )
            for i in range(len(self.control_commands["steer"])):
                f.write(
                    f"{self.control_commands['steer'][i]},{self.control_commands['throttle'][i]},{self.control_commands['brake'][i]},{self.control_commands['hand_brake'][i]},{self.control_commands['reverse'][i]},{self.control_commands['manual_gear_shift'][i]},{self.control_commands['gear'][i]},{self.states['acceleration']['value'][i]},{self.states['acceleration']['x'][i]},{self.states['acceleration']['y'][i]},{self.states['acceleration']['z'][i]},{self.states['transform']['x'][i]},{self.states['transform']['y'][i]},{self.states['transform']['z'][i]},{self.states['transform']['yaw'][i]},{self.states['transform']['pitch'][i]},{self.states['transform']['roll'][i]},{self.states['velocity']['value'][i]},{self.states['velocity']['x'][i]},{self.states['velocity']['y'][i]},{self.states['velocity']['z'][i]}, {self.states['route_points']['remain'][i]}\n"
                )
