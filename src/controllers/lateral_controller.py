##!/usr/bin/env python3
# @File: lateral_controller.py
# @Description: Lateral controller for the vehicle.
# @CreatedTime: 2024/07/08
# @Author: PDM-Lite


import numpy as np

from src.controllers.controller_base import ControllerBase


class LateralPIDController(ControllerBase):
    """
    Lateral controller based on a Proportional-Integral-Derivative (PID) controller.

    Adapted from https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/team_code/lateral_controller.py.
    """

    def __init__(self, configs):
        super().__init__(configs)

        self.kp = self.configs.kp
        self.kd = self.configs.kd
        self.ki = self.configs.ki

        self.speed_scale = self.configs.speed_scale
        self.speed_offset = self.configs.speed_offset
        self.default_lookahead = self.configs.default_lookahead
        self.speed_threshold = self.configs.speed_threshold

        self.route_points = self.configs.route_points
        self.window_size = self.configs.window_size
        self.min_lookahead_distance = self.configs.min_lookahead_distance
        self.max_lookahead_distance = self.configs.max_lookahead_distance

    def get_steering(
        self,
        route_points,
        current_speed,
        vehicle_position,
        vehicle_heading,
        inference_mode=False,
    ):
        """
        Computes the steering angle based on the route, current speed, vehicle position, and heading.

        Args:
            route_points (numpy.ndarray): Array of (x, y) coordinates representing the route.
            current_speed (float): Current speed of the vehicle in m/s.
            vehicle_position (numpy.ndarray): Array of (x, y) coordinates representing the vehicle's position.
            vehicle_heading (float): Current heading angle of the vehicle in radians.
            inference_mode (bool): Controls whether to TF or PDM-Lite executes this method.

        Returns:
            float: Computed steering angle in the range [-1.0, 1.0].
        """
        current_speed_kph = current_speed * 3.6  # Convert speed from m/s to km/h

        # Compute the lookahead distance based on the current speed
        # Transfuser predicts checkpoints 1m apart, whereas in the expert the route points have distance 10cm.
        if inference_mode:
            lookahead_distance = self.speed_scale * current_speed + self.speed_offset
            lookahead_distance = (
                np.clip(
                    lookahead_distance,
                    self.min_lookahead_distance,
                    self.max_lookahead_distance,
                )
                / self.route_points
            )  # range [2.4, 10.5]
            lookahead_distance = lookahead_distance - 2  # range [0.4, 8.5]
        else:
            lookahead_distance = (
                self.speed_scale * current_speed_kph + self.speed_offset
            )
            lookahead_distance = np.clip(
                lookahead_distance,
                self.min_lookahead_distance,
                self.max_lookahead_distance,
            )

        lookahead_distance = int(min(lookahead_distance, route_points.shape[0] - 1))

        # Calculate the desired heading vector from the lookahead point
        desired_heading_vec = route_points[lookahead_distance] - vehicle_position
        desired_heading_angle = np.arctan2(
            desired_heading_vec[1], desired_heading_vec[0]
        )

        # Calculate the heading error
        heading_error = (desired_heading_angle - vehicle_heading) % (2 * np.pi)
        heading_error = (
            heading_error if heading_error < np.pi else heading_error - 2 * np.pi
        )

        # Scale the heading error (leftover from a previous implementation)
        heading_error = heading_error * 180.0 / np.pi / 90.0

        # Update the error history. Only use the last window_size errors like in a deque.
        self.error_history.append(heading_error)
        self.error_history = self.error_history[-self.window_size :]

        # Calculate the derivative and integral terms
        derivative = (
            0.0
            if len(self.error_history) == 1
            else self.error_history[-1] - self.error_history[-2]
        )
        integral = np.mean(self.error_history)

        # Compute the steering angle using the PID control law
        steering = np.clip(
            self.kp * heading_error + self.kd * derivative + self.ki * integral,
            -1.0,
            1.0,
        ).item()

        return steering
