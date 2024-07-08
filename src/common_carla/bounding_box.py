from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def detect_vehicle_obstacle():
    """Check ."""


def get_bounding_boxes(lidar=None):
    ego_vehicle = CarlaDataProvider.get_hero_actor()
    world = CarlaDataProvider.get_world()

    ego_transform = ego_vehicle.get_transform()
    ego_rotation = ego_transform.rotation

    ego_control = ego_vehicle.get_control()
    ego_brake = ego_control.brake

    actors = world.get_actors()
    actor_list = actors.filter("*vehicle*")
