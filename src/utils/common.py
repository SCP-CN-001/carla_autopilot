import os
import random

import pandas as pd

WORKSPACE = os.path.join(os.path.dirname(__file__), "../..")


def get_absolute_path(path_):
    if not os.path.isabs(path_):
        return os.path.join(WORKSPACE, path_)
    else:
        return path_


def get_random_weather(from_file: bool = False) -> dict:
    if from_file:
        path_file = get_absolute_path("src/configs/weathers.csv")
        df_weathers = pd.read_csv(path_file)
        weather = df_weathers.sample(axis=0, weights=df_weathers.frequency).to_dict(
            orient="records"
        )[0]
        weather.pop("frequency")
        return weather

    else:
        weather = {
            "cloudiness": random.uniform(0, 100),
            "precipitation": random.uniform(0, 100),
            "precipitation_deposits": random.uniform(0, 100),
            "wind_intensity": random.uniform(0, 100),
            "sun_azimuth_angle": random.uniform(0, 360),
            "sun_altitude_angle": random.uniform(-90, 90),
            "fog_density": random.uniform(0, 100),
            "wetness": random.uniform(0, 100),
        }

        return weather
