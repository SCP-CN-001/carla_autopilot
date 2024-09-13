##!/usr/bin/env python3
# @File: controller_base.py
# @Description: Base class for controllers.
# @CreatedTime: 2024/07/08
# @Author: Yueyuan Li

from abc import ABC


class ControllerBase(ABC):
    def __init__(self, configs):
        self.configs = configs

        self.error_history = []
        self.saved_error_history = []

    def save(self):
        """
        Saves the current state of the controller. Useful during forecasting.
        """
        self.saved_error_history = self.error_history.copy()

    def load(self):
        """
        Loads the previously saved state of the controller. Useful during forecasting.
        """
        self.error_history = self.saved_error_history.copy()
