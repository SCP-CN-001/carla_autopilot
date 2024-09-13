##!/usr/bin/env python3
# @File: video_writier.py
# @Description:
# @CreatedTime: 2024/07/29
# @Author: Yueyuan Li

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import re

import cv2
import numpy as np
import open3d as o3d
from carla_autopilot.utils.common import get_absolute_path
from omegaconf import OmegaConf
from visualizer import Visualizer


def write_to_video(path_config):
    path_config = get_absolute_path(path_config)
    configs = OmegaConf.load(path_config)
    path_data = get_absolute_path(configs.path_data)
    path_video = get_absolute_path(configs.path_video)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        path_video, fourcc, configs.frame_rate, (configs.width, configs.height)
    )

    # Get the frame range from the folder
    first_data_folder = list(configs.layout.values())[0].folder_name
    files = os.listdir(f"{path_data}/{first_data_folder}")
    frames = [int(re.findall(r"\d+", file)[0]) for file in files]
    frame_range = (min(frames), max(frames))

    if configs.frame_range is not None:
        frame_range = (
            max(frame_range[0], configs.frame_range[0]),
            min(frame_range[1], configs.frame_range[1]),
        )

    frames = [x for x in frames if frame_range[0] <= x <= frame_range[1]]
    frames.sort()

    for frame in frames:
        surface = 255 * np.ones((configs.height, configs.width, 3), dtype=np.uint8)
        for layout in configs.layout.values():
            path_file = (
                f"{path_data}/{layout.folder_name}/frame_{frame:06d}.{layout.suffix}"
            )

            if layout.suffix in ["jpg", "png"]:
                image = cv2.imread(path_file)
                image = cv2.resize(image, (layout.width, layout.height))
                surface[
                    layout.loc_y : layout.loc_y + layout.height,
                    layout.loc_x : layout.loc_x + layout.width,
                ] = image
            elif layout.suffix in ["ply"]:
                points = o3d.io.read_point_cloud(path_file)
                points = np.asarray(points.points)
                image = Visualizer.visualize_lidar(points, layout.size)
                image = np.vstack(([image], [image], [image])).transpose(1, 2, 0)
                surface[
                    layout.loc_y : layout.loc_y + layout.size,
                    layout.loc_x : layout.loc_x + layout.size,
                ] = image

        video.write(surface)
    video.release()


if __name__ == "__main__":
    write_to_video("carla_autopilot/configs/video_layout.yaml")
