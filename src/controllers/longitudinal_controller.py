##!/usr/bin/env python3
# @File: longitudinal_controller.py
# @Description:
# @CreatedTime: 2024/07/08
# @Author: PDM-Lite

import numpy as np

from src.controllers.controller_base import ControllerBase


class LongitudinalPIDController(ControllerBase):
    """
    Longitudinal controller based on a Proportional-Integral-Derivative (PID) controller.

    Adapted from https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/team_code/longitudinal_controller.py.

    This class was used for the ablations. Currently, the expert agent uses the linear regression controller for longitudinal control by default.
    """

    def __init__(self, config):
        super().__init__(config)

        # These parameters are tuned with Bayesian Optimization on a test track
        self.proportional_gain = self.config.longitudinal_pid_proportional_gain
        self.derivative_gain = self.config.longitudinal_pid_derivative_gain
        self.integral_gain = self.config.longitudinal_pid_integral_gain
        self.max_window_length = self.config.longitudinal_pid_max_window_length
        self.speed_error_scaling = self.config.longitudinal_pid_speed_error_scaling
        self.braking_ratio = self.config.longitudinal_pid_braking_ratio
        self.minimum_target_speed = self.config.longitudinal_pid_minimum_target_speed

    def get_throttle_and_brake(self, hazard_brake, target_speed, current_speed):
        """
        Get the throttle and brake values based on the target speed, current speed,
        and hazard brake condition using a PID controller.

        Args:
            hazard_brake (bool): Flag indicating whether to apply hazard braking.
            target_speed (float): The desired target speed in m/s.
            current_speed (float): The current speed of the vehicle in m/s.

        Returns:
            tuple: A tuple containing the throttle and brake values.
        """
        # If there's a hazard or the target speed is very small, apply braking
        if hazard_brake or target_speed < 1e-5:
            throttle, brake = 0.0, True
            return throttle, brake

        target_speed = max(
            self.minimum_target_speed, target_speed
        )  # Avoid very small target speeds

        current_speed, target_speed = (
            3.6 * current_speed,
            3.6 * target_speed,
        )  # Convert to km/h

        # Test if the speed is "much" larger than the target speed
        if current_speed / target_speed > self.braking_ratio:
            self.error_history = [0] * self.max_window_length

            throttle, brake = 0.0, True
            return throttle, brake

        speed_error = target_speed - current_speed
        speed_error = (
            speed_error + speed_error * current_speed * self.speed_error_scaling
        )

        self.error_history.append(speed_error)
        self.error_history = self.error_history[-self.max_window_length :]

        derivative = (
            0
            if len(self.error_history) == 1
            else self.error_history[-1] - self.error_history[-2]
        )
        integral = np.mean(self.error_history)

        throttle = (
            self.proportional_gain * speed_error
            + self.derivative_gain * derivative
            + self.integral_gain * integral
        )
        throttle, brake = np.clip(throttle, 0.0, 1.0), False

        return throttle, brake

    def get_throttle_extrapolation(self, target_speed, current_speed):
        """
        Get the throttle value for the given target speed and current speed, assuming no hazard brake condition.

        Args:
            target_speed (float): The desired target speed in m/s.
            current_speed (float): The current speed of the vehicle in m/s.

        Returns:
            float: The throttle value.
        """
        return self.get_throttle(False, target_speed, current_speed)


class LongitudinalLinearRegressionController(ControllerBase):
    """
    This class holds the linear regression module used for longitudinal control. It's used by default.

    Adapted from https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/team_code/longitudinal_controller.py.
    """

    def __init__(self, config):
        super().__init__(config)

        self.minimum_target_speed = (
            self.config.longitudinal_linear_regression_minimum_target_speed
        )
        self.params = self.config.longitudinal_linear_regression_params
        self.maximum_acceleration = (
            self.config.longitudinal_linear_regression_maximum_acceleration
        )
        self.maximum_deceleration = (
            self.config.longitudinal_linear_regression_maximum_deceleration
        )

    def get_throttle_and_brake(self, hazard_brake, target_speed, current_speed):
        """
        Get the throttle and brake values based on the target speed, current speed, and hazard brake condition using a
        linear regression model.

        Args:
            hazard_brake (bool): Flag indicating whether to apply hazard braking.
            target_speed (float): The desired target speed in m/s.
            current_speed (float): The current speed of the vehicle in m/s.

        Returns:
            tuple: A tuple containing the throttle and brake values.
        """
        if target_speed < 1e-5 or hazard_brake:
            return 0.0, True
        elif target_speed < self.minimum_target_speed:  # Avoid very small target speeds
            target_speed = self.minimum_target_speed

        current_speed = current_speed * 3.6
        target_speed = target_speed * 3.6
        params = self.params
        speed_error = target_speed - current_speed

        # Maximum acceleration 1.9 m/tick
        if speed_error > self.maximum_acceleration:
            return 1.0, False

        if current_speed / target_speed > params[-1] or hazard_brake:
            throttle, control_brake = 0.0, True
            return throttle, control_brake

        speed_error_cl = np.clip(speed_error, 0.0, np.inf) / 100.0
        current_speed /= 100.0
        features = np.array(
            [
                current_speed,
                current_speed**2,
                100 * speed_error_cl,
                speed_error_cl**2,
                current_speed * speed_error_cl,
                current_speed**2 * speed_error_cl,
            ]
        )

        throttle, control_brake = np.clip(features @ params[:-1], 0.0, 1.0), False

        return throttle, control_brake

    def get_throttle_extrapolation(self, target_speed, current_speed):
        """
        Get the throttle value for the given target speed and current speed, assuming no hazard brake condition.
        This method is used for forecasting.

        Args:
            target_speed (float): The desired target speed in m/s.
            current_speed (float): The current speed of the vehicle in m/s.

        Returns:
            float: The throttle value.
        """
        current_speed = current_speed * 3.6  # Convert to km/h
        target_speed = target_speed * 3.6  # Convert to km/h
        params = self.params
        speed_error = target_speed - current_speed

        # Maximum acceleration 1.9 m/tick
        if speed_error > self.maximum_acceleration:
            return 1.0
        # Maximum deceleration -4.82 m/tick
        elif speed_error < self.maximum_deceleration:
            return 0.0

        throttle = 0.0
        # 0.1 to ensure small distances are overcome fast
        if target_speed < 0.1 or current_speed / target_speed > params[-1]:
            return throttle

        speed_error_cl = (
            np.clip(speed_error, 0.0, np.inf) / 100.0
        )  # The scaling is a leftover from the optimization
        current_speed /= 100.0  # The scaling is a leftover from the optimization
        features = np.array(
            [
                current_speed,
                current_speed**2,
                100 * speed_error_cl,
                speed_error_cl**2,
                current_speed * speed_error_cl,
                current_speed**2 * speed_error_cl,
            ]
        ).flatten()

        throttle = np.clip(features @ params[:-1], 0.0, 1.0)

        return throttle
