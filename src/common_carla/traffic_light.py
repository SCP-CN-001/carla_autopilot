from collections import deque

import carla
import numpy as np


def get_traffic_light_waypoints(traffic_light, carla_map):
    """
    get area of a given traffic light
    """
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
    wps_queue = deque(wps.copy())
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
