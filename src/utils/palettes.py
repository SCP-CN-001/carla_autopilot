import numpy as np

# This palette is mainly used for visualization.
CITYSCAPE_PALETTE = np.array(
    [
        [0, 0, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [110, 190, 160],
        [170, 120, 50],
        [55, 90, 80],
        [45, 60, 150],
        [157, 234, 50],
        [81, 0, 81],
        [150, 100, 100],
        [230, 150, 140],
        [180, 165, 180],
    ]
)

# This palette is customized for more beautiful visualization.
# TODO: Select the colors later
CUSTOM_PALETTE = np.array(
    [
        [0, 0, 0],  # None
        [93, 101, 95],  # Road
        [115, 124, 123],  # SideWalks
        [73, 117, 104],  # Buildings
        [134, 157, 157],  # Walls
        [106, 106, 106],  # Fences
        [105, 105, 105],  # Poles
        [255, 255, 0],  # TrafficLight
        [255, 0, 0],  # TrafficSigns
        [0, 255, 0],  # Vegetation
        [0, 0, 255],  # Terrain
        [0, 255, 255],  # Sky
        [255, 0, 255],  # Pedestrians
        [255, 255, 255],  # Rider
        [255, 0, 0],  # Car
        [0, 255, 0],  # Truck
        [0, 0, 255],  # Bus
        [255, 255, 0],  # Train
        [255, 0, 255],  # Motorcycle
        [0, 255, 255],  # Bicycle
        [255, 255, 255],  # Static
        [0, 0, 0],  # Dynamic
        [0, 0, 0],  # Other
        [0, 0, 0],  # Water
        [255, 255, 0],  # RoadLines
        [0, 255, 0],  # Ground
        [0, 0, 255],  # Bridge
        [255, 0, 255],  # RailTrack
        [0, 255, 255],  # GuardRail
    ]
)

# This palette maps the tags to the following layers: roadline and traffic signs, road, static obstacle, dynamic obstacle
BINARY_PALETTE = np.array(
    [
        [0, 0, 0, 0],  # None
        [0, 1, 0, 0],  # Road
        [0, 0, 1, 0],  # SideWalks
        [0, 0, 1, 0],  # Buildings
        [0, 0, 1, 0],  # Walls
        [0, 0, 1, 0],  # Fences
        [0, 0, 1, 0],  # Poles
        [1, 0, 0, 0],  # TrafficLight
        [1, 0, 0, 0],  # TrafficSigns
        [0, 0, 1, 0],  # Vegetation
        [0, 0, 0, 0],  # Terrain
        [0, 0, 0, 0],  # Sky
        [0, 0, 0, 1],  # Pedestrians
        [0, 0, 0, 1],  # Rider
        [0, 0, 0, 1],  # Car
        [0, 0, 0, 1],  # Truck
        [0, 0, 0, 1],  # Bus
        [0, 0, 0, 1],  # Train
        [0, 0, 0, 1],  # Motorcycle
        [0, 0, 0, 1],  # Bicycle
        [0, 0, 1, 0],  # Static
        [0, 0, 0, 1],  # Dynamic
        [0, 0, 1, 1],  # Other
        [0, 0, 0, 0],  # Water
        [1, 0, 0, 0],  # RoadLines
        [0, 0, 0, 0],  # Ground
        [0, 1, 0, 0],  # Bridge
        [0, 1, 0, 0],  # RailTrack
        [0, 1, 0, 0],  # GuardRail
    ]
)

CLASS_DICT = {
    0: "None",
    1: "Road",
    2: "SideWalks",
    3: "Buildings",
    4: "Walls",
    5: "Fences",
    6: "Poles",
    7: "TrafficLight",
    8: "TrafficSigns",
    9: "Vegetation",
    10: "Terrain",
    11: "Sky",
    12: "Pedestrians",
    13: "Rider",
    14: "Car",
    15: "Truck",
    16: "Bus",
    17: "Train",
    18: "Motorcycle",
    19: "Bicycle",
    20: "Static",
    21: "Dynamic",
    22: "Other",
    23: "Water",
    24: "RoadLines",
    25: "Ground",
    26: "Bridge",
    27: "RailTrack",
    28: "GuardRail",
}
