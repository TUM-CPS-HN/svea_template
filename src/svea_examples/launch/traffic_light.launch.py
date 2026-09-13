#! /usr/bin/env python3
from better_launch import BetterLaunch, launch_this


@launch_this
def main(
    name: str = "light_a",
    pos_x: float = 3.34,
    pos_y: float = 0.90,
    alpha: float = 0.0,
    wait_for_start: bool = False,
):
    """Launch a single traffic light, fixed at (pos_x, pos_y) on the map.

    Meant to be run alongside floor2.launch.py so students can build their
    own logic for reacting to the light's state, published on
    '/<name>/state' ('Rd', 'YG', 'Gr', 'YR') and '/<name>/time_left'.
    """
    bl = BetterLaunch()

    bl.node("svea_examples", "traffic_light.py",
            name=name,
            params={
                "name": name,
                "pos_x": pos_x,
                "pos_y": pos_y,
                "alpha": alpha,
                "wait_for_start": wait_for_start,
            })
