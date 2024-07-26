from collections import deque

import carla
import numpy as np

from src.common_carla.geometry import dot_product, location_global_to_local

traffic_light_states = {
    0: carla.TrafficLightState.Green,
    1: carla.TrafficLightState.Yellow,
    2: carla.TrafficLightState.Red,
}


def get_traffic_light_waypoints(traffic_light, carla_map):
    """Get area of a given traffic light"""
    base_transform = traffic_light.get_transform()

    trigger_volume_loc = traffic_light.trigger_volume.location
    area_loc = carla.Location(base_transform.transform(trigger_volume_loc))

    # Discretize the trigger box into points
    area_ext = traffic_light.trigger_volume.extent
    x_values = np.arange(
        -0.9 * area_ext.x, 0.9 * area_ext.x, 1.0
    )  # 0.9 to avoid crossing to adjacent lanes

    area = []
    for x in x_values:
        point_location = base_transform.transform(
            trigger_volume_loc + carla.Location(x=x)
        )
        area.append(point_location)

    # Get the waypoints of these points, removing duplicates
    ini_wps = []
    for pt in area:
        wpx = carla_map.get_waypoint(pt)
        # As x_values are arranged in order, only the last one has to be checked
        if (
            not ini_wps
            or ini_wps[-1].road_id != wpx.road_id
            or ini_wps[-1].lane_id != wpx.lane_id
        ):
            ini_wps.append(wpx)

    # Advance them until the intersection
    wps = []
    stopline_vertices = []
    for wpx in ini_wps:
        while not wpx.is_intersection:
            next_wp = wpx.next(0.5)[0]
            if next_wp and not next_wp.is_intersection:
                wpx = next_wp
            else:
                break
        wps.append(wpx)

        vec_forward = wpx.transform.get_forward_vector()
        vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)

        loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
        loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
        stopline_vertices.append([loc_left, loc_right])

    # all paths at junction for this traffic light
    junction_paths = []
    path_wps = []
    wps_queue = deque(wps)
    while len(wps_queue) > 0:
        current_wp = wps_queue.pop()
        path_wps.append(current_wp)
        next_wps = current_wp.next(1.0)
        for next_wp in next_wps:
            if next_wp.is_junction:
                wps_queue.append(next_wp)
            else:
                junction_paths.append(path_wps)
                path_wps = []

    return area_loc, wps, stopline_vertices, junction_paths


def get_before_traffic_light_waypoints(traffic_light, carla_map):
    """Get only the waypoints before a given traffic light."""
    base_transform = traffic_light.get_transform()
    base_loc = traffic_light.get_location()
    trigger_volume_loc = traffic_light.trigger_volume.location
    area_loc = carla.Location(base_transform.transform(trigger_volume_loc))

    # Discretize the trigger box into points
    area_ext = traffic_light.trigger_volume.extent
    x_values = np.arange(
        -0.9 * area_ext.x, 0.9 * area_ext.x, 1.0
    )  # 0.9 to avoid crossing to adjacent lanes

    area = []
    for x in x_values:
        point_location = base_transform.transform(
            trigger_volume_loc + carla.Location(x=x)
        )
        area.append(point_location)

    # Get the waypoints of these points, removing duplicates
    ini_wps = []
    for pt in area:
        wpx = carla_map.get_waypoint(pt)
        # As x_values are arranged in order, only the last one has to be checked
        if (
            not ini_wps
            or ini_wps[-1].road_id != wpx.road_id
            or ini_wps[-1].lane_id != wpx.lane_id
        ):
            ini_wps.append(wpx)

    # Advance them until the intersection
    wps = []
    eu_wps = []
    for wpx in ini_wps:
        distance_to_light = base_loc.distance(wpx.transform.location)
        eu_wps.append(wpx)
        next_distance_to_light = distance_to_light + 1.0
        while not wpx.is_intersection:
            next_wp = wpx.next(0.5)[0]
            next_distance_to_light = base_loc.distance(next_wp.transform.location)
            if (
                next_wp
                and not next_wp.is_intersection
                and next_distance_to_light <= distance_to_light
            ):
                eu_wps.append(next_wp)
                distance_to_light = next_distance_to_light
                wpx = next_wp
            else:
                break

        if not next_distance_to_light <= distance_to_light and len(eu_wps) >= 4:
            wps.append(eu_wps[-4])
        else:
            wps.append(wpx)

    return area_loc, wps


class TrafficLightHandler:
    """Class used to generate stop lines for the traffic lights."""

    world_map = None
    num_traffic_light = 0
    list_traffic_light = []
    list_trigger_volume_loc = []
    list_stopline_wps = []
    list_stopline_vtx = []
    list_junction_paths = []

    @staticmethod
    def reset(world):
        TrafficLightHandler.world_map = world.get_map()
        TrafficLightHandler.num_traffic_light = 0
        TrafficLightHandler.list_traffic_light = []
        TrafficLightHandler.list_stopline_vtx = []
        TrafficLightHandler.list_junction_paths = []

        all_actors = world.get_actors()
        for actor in all_actors:
            if "traffic_light" in actor.type_id:
                (
                    trigger_volume_loc,
                    stopline_wps,
                    stopline_vtx,
                    junction_paths,
                ) = get_traffic_light_waypoints(actor, TrafficLightHandler.world_map)

                TrafficLightHandler.list_traffic_light.append(actor)
                TrafficLightHandler.list_trigger_volume_loc.append(trigger_volume_loc)
                TrafficLightHandler.list_stopline_wps.append(stopline_wps)
                TrafficLightHandler.list_stopline_vtx.append(stopline_vtx)
                TrafficLightHandler.list_junction_paths.append(junction_paths)
                TrafficLightHandler.num_traffic_light += 1

    @staticmethod
    def get_light_state(vehicle: carla.Vehicle, offset=0.0, dist_threshold=15.0):
        vehicle_transform = vehicle.get_transform()
        vehicle_direction = vehicle_transform.get_forward_vector()

        hit_loc = vehicle_transform.transform(carla.Location(x=offset))
        hit_wp = TrafficLightHandler.carla_map.get_waypoint(hit_loc)

        light_loc = None
        light_state = None
        light_id = None

        for i in range(TrafficLightHandler.num_traffic_light):
            light = TrafficLightHandler.list_traffic_light[i]
            trigger_volume_loc = TrafficLightHandler.list_tv_loc[i]
            if trigger_volume_loc.distance(hit_loc) > dist_threshold:
                continue

            for wp in TrafficLightHandler.list_stopline_wps[i]:
                wp_direction = wp.transform.get_forward_vector()
                dot_vehicle_wp = dot_product(wp_direction, vehicle_direction)

                wp_1 = wp.previous(4.0)[0]

                same_road = (hit_wp.road_id == wp.road_id) and (
                    hit_wp.lane_id == wp.lane_id
                )
                same_road_1 = (hit_wp.road_id == wp_1.road_id) and (
                    hit_wp.lane_id == wp_1.lane_id
                )

                if (same_road or same_road_1) and dot_vehicle_wp > 0:
                    # This light is red and is affecting our lane
                    light_loc = location_global_to_local(
                        wp.transform.location, vehicle_transform
                    )
                    light_loc_np = np.array(
                        [light_loc.x, light_loc.y, light_loc.z], dtype=np.float64
                    )
                    light_state = light.state
                    light_id = light.id

        return light_state, light_loc_np, light_id

    @staticmethod
    def get_junction_paths(vehicle, color=0, dist_threshold=50.0):
        vehicle_location = vehicle.get_location()
        traffic_light_state = traffic_light_states[color]

        junction_paths = []
        for i in range(TrafficLightHandler.num_traffic_light):
            traffic_light = TrafficLightHandler.list_traffic_light[i]
            trigger_volume_loc = TrafficLightHandler.list_trigger_volume_loc[i]

            if trigger_volume_loc.distance(vehicle_location) > dist_threshold:
                continue
            if traffic_light.state != traffic_light_state:
                continue

            junction_paths += TrafficLightHandler.list_junction_paths[i]

        return junction_paths

    @staticmethod
    def get_stopline_vertices(
        vehicle, color=0, dist_threshold=50.0, close_traffic_lights=None
    ):
        vehicle_location = vehicle.get_location()
        traffic_light_state = traffic_light_states[color]

        stopline_vtx = []
        for i in range(TrafficLightHandler.num_traffic_light):
            traffic_light = TrafficLightHandler.list_traffic_light[i]
            trigger_volume_loc = TrafficLightHandler.list_trigger_volume_loc[i]

            if trigger_volume_loc.distance(vehicle_location) > dist_threshold:
                continue
            if traffic_light.state != traffic_light_state:
                continue

            if close_traffic_lights is not None:
                for close_traffic_light in close_traffic_lights:
                    if (
                        traffic_light.id == int(close_traffic_light[2])
                        and close_traffic_light[3]
                    ):
                        stopline_vtx += TrafficLightHandler.list_stopline_vtx[i]
                        break
            else:
                stopline_vtx += TrafficLightHandler.list_stopline_vtx[i]

        return stopline_vtx
