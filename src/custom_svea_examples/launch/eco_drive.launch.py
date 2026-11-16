#! /usr/bin/env python3
"""Tutorial 2: eco-drive through traffic lights on the TUM floor-2 map.

    # one light on the first straight (Task 1)
    ros2 launch custom_svea_examples eco_drive.launch.py

    # bonus: full lap with a light on every leg (curved segments, several lights)
    ros2 launch custom_svea_examples eco_drive.launch.py lap:=true

    # stress test: random light timing jitter (the planner uses guaranteed windows)
    ros2 launch custom_svea_examples eco_drive.launch.py alpha:=0.5

    # energy vs. time trade-off: try w_acc:=0.5 (sporty) ... w_acc:=40 (very eco)
    ros2 launch custom_svea_examples eco_drive.launch.py w_acc:=20.0

    # load test: 24 random lights on the lap, some only 0.4-1.0 m apart, random phases
    ros2 launch custom_svea_examples eco_drive.launch.py lap:=true random_lights:=24 seed:=3

    # adversarial: close lights with uncoordinated phases (may be impassable: car must wait)
    ros2 launch custom_svea_examples eco_drive.launch.py lap:=true random_lights:=16 coordinated:=false

    # real car (lights run on the laptop, shown in Foxglove)
    ros2 launch custom_svea_examples eco_drive.launch.py is_sim:=false

The lights start their cycle when the planner has its first plan
(``wait_for_start``), so every run begins the same way. Pass
``wait_for_start:=false`` to let them free-run instead.
"""
from better_launch import BetterLaunch, launch_this

MAP_NAME = "tum_d_floor2"

# Corridor loop from Tutorial 1 (floor2.launch.py)
LAP_POINTS = [-0.38, 0.96, 15.0, -2.69, 18.13, 10.64, 3.22, 14.32]
# Task 1: the first straight leg only; the light sits about half way along it
LEG_POINTS = LAP_POINTS[:4]
LEG_LIGHTS = {"light_a": (7.0, -0.79)}
# Bonus: one light half way along every leg of the lap
LAP_LIGHTS = {
    "light_a": (7.0, -0.79),
    "light_b": (16.57, 3.98),
    "light_c": (10.68, 12.48),
    "light_d": (1.42, 7.64),
}


def as_bool(v) -> bool:
    """Command-line launch args arrive as strings ("false" would be truthy!)."""
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


def random_lights_on_route(points, closed, n, seed, alpha, cluster_prob, coordinated):
    """Random light layout along the route (see custom_svea_core.eco_drive.scenarios)."""
    import numpy as np
    from custom_svea_core.eco_drive.route import Route
    from custom_svea_core.eco_drive.scenarios import random_layout
    from custom_svea_core.eco_drive.traffic_signal import LightTiming

    route = Route(np.asarray(points, float).reshape(-1, 2), closed=closed)
    layout = random_layout(route.length, n, seed, LightTiming(alpha=alpha), cluster_prob=cluster_prob,
                           closed=closed, coordinate_clusters=coordinated)
    lights, offsets = {}, []
    for p in layout:
        x, y = route.xy(p.s)
        lights[p.name] = (round(float(x), 3), round(float(y), 3))
        offsets.append(round(p.offset, 3))
    gaps = np.diff([p.s for p in layout])
    print(f"[eco_drive] {len(layout)} random lights (seed {seed}), "
          f"{int(np.sum(gaps < 0.9))} pairs closer than 0.9 m, min gap {gaps.min():.2f} m")
    return lights, offsets


@launch_this
def main(
    is_sim: bool = True,
    use_foxglove: bool = True,
    lap: bool = False,
    initial_pose_x: float = -2.25,
    initial_pose_y: float = 0.89,
    initial_pose_a: float = -0.1,
    alpha: float = 0.0,
    wait_for_start: bool = True,
    w_time: float = 1.0,
    w_acc: float = 4.0,
    w_jerk: float = 0.5,
    a_max: float = 1.0,
    v_max: float = 1.2,
    random_lights: int = 0,
    seed: int = 0,
    cluster_prob: float = 0.35,
    coordinated: bool = True,
):
    bl = BetterLaunch()
    is_sim, use_foxglove, lap, wait_for_start, coordinated = map(
        as_bool, (is_sim, use_foxglove, lap, wait_for_start, coordinated))
    random_lights, seed, cluster_prob = int(random_lights), int(seed), float(cluster_prob)
    initial_pose_x, initial_pose_y, initial_pose_a, alpha, w_time, w_acc, w_jerk, a_max, v_max = map(
        float, (initial_pose_x, initial_pose_y, initial_pose_a, alpha, w_time, w_acc, w_jerk, a_max, v_max))
    lights = LAP_LIGHTS if lap else LEG_LIGHTS
    offsets = None
    if random_lights > 0:
        lights, offsets = random_lights_on_route(
            LAP_POINTS if lap else [initial_pose_x, initial_pose_y] + LEG_POINTS,
            lap, random_lights, seed, alpha, cluster_prob, coordinated)
    name = "svea_a" if is_sim else "self"

    bl.include("custom_svea_core", "svea_tum.launch.py",
               name=name,
               is_sim=is_sim,
               is_indoor=True,
               map_name=MAP_NAME,
               initial_pose_x=initial_pose_x,
               initial_pose_y=initial_pose_y,
               initial_pose_a=initial_pose_a)

    # Traffic lights (outside the vehicle namespace: topics are /<light>/state, /<light>/time_left)
    if offsets is None:
        for light, (x, y) in lights.items():
            bl.node("custom_svea_examples", "traffic_light.py",
                    name=light,
                    params={"name": light, "pos_x": x, "pos_y": y, "alpha": alpha,
                            "wait_for_start": wait_for_start})
    else:
        # one process for all random lights (same topics and behaviour as traffic_light.py)
        bl.node("custom_svea_examples", "traffic_light_group.py",
                name="traffic_lights",
                params={"names": list(lights.keys()),
                        "positions": [c for xy in lights.values() for c in xy],
                        "offsets": offsets, "alpha": alpha, "seed": seed,
                        "wait_for_start": wait_for_start})

    params = {
        "is_sim": is_sim,
        "points": LAP_POINTS if lap else LEG_POINTS,
        "closed": lap,
        "lights": list(lights.keys()),
        "light_positions": [c for xy in lights.values() for c in xy],
        "alpha": alpha,
        "start_lights": wait_for_start,
        "w_time": w_time,
        "w_acc": w_acc,
        "w_jerk": w_jerk,
        "a_max": a_max,
        "v_max": v_max,
    }
    if is_sim:
        params["localization/base_frame"] = f"{name}/base_link"

    with bl.group(name):
        bl.node("custom_svea_examples", "eco_drive.py", name="eco_drive", params=params)

    if use_foxglove:
        bl.include("foxglove_bridge", "foxglove_bridge_launch.xml")
