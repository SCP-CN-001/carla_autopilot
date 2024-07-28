##!/usr/bin/env python3
# @File: visualizer.py
# @Description:
# @CreatedTime: 2024/06/02
# @Author: Yueyuan Li

import logging
import time

import numpy as np
import pygame
from omegaconf import OmegaConf

from src.utils.common import get_absolute_path


class Visualizer:
    def __init__(self, configs) -> None:
        self._width = configs["width"]
        self._height = configs["height"]
        self._surface = None

        self._surface_configs = configs["surface_configs"]
        self._cnt = 0

        if "path_semantic_palette" in configs:
            self.semantic_palette = np.array(
                OmegaConf.load(
                    get_absolute_path(configs["path_semantic_palette"])
                ).color_map
            )

        if "path_binary_palette" in configs:
            self.binary_palette = np.array(
                OmegaConf.load(
                    get_absolute_path(configs["path_binary_palette"])
                ).color_map
            )

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        logging.info("Visualizer initialized.")

    def visualize_semantic_segmentation(self, image, path_palette=None):
        """Visualize a semantic segmentation image using a color palette.

        Args:
            image (np.ndarray): The semantic segmentation image. The image should be a 3D numpy array with shape (H, W, 3). Only the first channel is used for visualization with integer values representing the class labels.
            path_palette (str): The path to the color palette to use for visualization.

            Return:
                pygame.Surface: The surface object containing the visualization.
        """
        if path_palette is not None:
            palette = np.array(
                OmegaConf.load(get_absolute_path(path_palette)).color_map
            )
        else:
            palette = self.semantic_palette
        rgb_image = palette[image[:, :, 0]]

        return rgb_image

    def visualize_semantic_segmentation_binary(self, image, path_palette=None):
        """Visualize a semantic segmentation image to four binary layers.

        Args:
            image (np.ndarray): The semantic segmentation image. The image should be a 3D numpy array with shape (H, W, 3). Only the first channel is used for visualization with integer values representing the class labels.
            path_palette (str): The path to the color palette to use for visualization.

        Returns:
            List[pygame.Surface]:
        """
        if path_palette is not None:
            palette = np.array(
                OmegaConf.load(get_absolute_path(path_palette)).color_map
            )
        else:
            palette = self.binary_palette

        n_binary = len(palette[0])
        assert n_binary < 8, "The number of binary layers should be less than 8."

        binary_images = palette[image[:, :, 0]] * 255

        if n_binary <= 4:
            binary_image = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
            for i in range(n_binary):
                binary_image[
                    image.shape[0] * (i // 2) : image.shape[0] * (i // 2 + 1),
                    image.shape[1] * (i % 2) : image.shape[1] * (i % 2 + 1),
                ] = binary_images[:, :, i]
        else:
            binary_image = np.zeros((image.shape[0], image.shape[1]))
            for i in range(n_binary):
                binary_image += binary_images[:, :, i] * (2**i)

        return binary_image

    @staticmethod
    def visualize_lidar(lidar_points, size=500) -> np.ndarray:
        """Visualize LiDAR data as a 2D histogram.

        Refer to https://github.com/angelomorgado/CARLA-Sensor-Visualization/tree/main.
        """
        # Extract X, Y, Z coordinates and intensity values
        points_xyz = lidar_points[:, :3]
        intensity = lidar_points[:, 3]

        # Intensity scaling factor
        intensity_scale = 255.0  # Adjust this value to control the brightness

        # Create a 2D histogram with a predetermined size
        lidar_image = np.zeros((size, size))

        # Scale and shift X and Y coordinates to fit within the histogram size
        x_scaled = ((points_xyz[:, 0] + 50) / 100) * (size - 1)
        y_scaled = ((points_xyz[:, 1] + 50) / 100) * (size - 1)

        # Round the scaled coordinates to integers
        x_ids = np.round(x_scaled).astype(int)
        y_ids = np.round(y_scaled).astype(int)

        # Clip the indices to stay within the image bounds
        x_ids = np.clip(x_ids, 0, size - 1)
        y_ids = np.clip(y_ids, 0, size - 1)

        # Assign scaled intensity values to the corresponding pixel in the histogram
        lidar_image[y_ids, x_ids] = intensity * intensity_scale

        # Clip the intensity values to stay within the valid color range
        lidar_image = np.clip(lidar_image, 0, 255)

        return lidar_image

    def _get_scaled_surface(
        self, surface: pygame.Surface, scale: float
    ) -> pygame.Surface:
        h_surface = surface.get_height()
        w_surface = surface.get_width()
        h_expected = self._height * scale
        w_expected = self._width * scale
        scale_transform = min(h_expected / h_surface, w_expected / w_surface)

        return pygame.transform.scale_by(surface, scale_transform)

    def update(self, raw_data):
        black_array = np.zeros((self._width, self._height, 3))
        self._surface = pygame.surfarray.make_surface(black_array)

        for sensor_id, visualizer_config in self._surface_configs.items():
            if visualizer_config["render_method"] == "camera":
                image = raw_data[sensor_id][1][:, :, -2::-1]
                surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
                surface = self._get_scaled_surface(surface, visualizer_config["scale"])

            elif visualizer_config["render_method"] == "semantic_segmentation":
                image = raw_data[sensor_id][1][:, :, -2::-1]
                image = self.visualize_semantic_segmentation(image)
                surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
                surface = self._get_scaled_surface(surface, visualizer_config["scale"])

            elif visualizer_config["render_method"] == "semantic_segmentation_binary":
                image = raw_data[sensor_id][1][:, :, -2::-1]
                image = self.visualize_semantic_segmentation_binary(image)
                surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
                surface = self._get_scaled_surface(surface, visualizer_config["scale"])

            elif visualizer_config["render_method"] == "lidar":
                lidar = raw_data[sensor_id][1]
                image = self.visualize_lidar(
                    lidar,
                    size=min(
                        int(self._width * visualizer_config["scale"]),
                        int(self._height * visualizer_config["scale"]),
                    ),
                )
                surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

            self._surface.blit(
                surface,
                (visualizer_config["loc_x"], visualizer_config["loc_y"]),
            )

        self._cnt += 1
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

        logging.debug("Visualizer updated.")

    def clean(self):
        black_array = np.zeros((self._width, self._height))
        self._surface = pygame.surfarray.make_surface(black_array)
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def quit(self):
        pygame.quit()
