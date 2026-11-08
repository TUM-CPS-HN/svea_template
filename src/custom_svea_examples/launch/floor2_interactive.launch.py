#! /usr/bin/env python3
from better_launch import BetterLaunch, launch_this

MAP_NAME = "tum_d_floor2"

@launch_this
def main(
    is_sim: bool = True,
    use_foxglove: bool = True,
    initial_pose_x: float = -2.25,
    initial_pose_y: float = 0.89,
    initial_pose_a: float = -0.1,
):
    bl = BetterLaunch()

    if not is_sim:

        # Start SVEA in real-world mode
        bl.include("custom_svea_core", "svea_tum.launch.py",
                   is_sim=is_sim, # = False
                   map_name=MAP_NAME,
                   initial_pose_x=initial_pose_x,
                   initial_pose_y=initial_pose_y,
                   initial_pose_a=initial_pose_a)

        # The SVEA launch system is built to be compatible with multiple SVEAs running simultaneously.
        # Default name is "self", so to add the pure_pursuit_interactive node we need to namespace accordingly.
        with bl.group("self"):
        
            bl.node("custom_svea_examples", "pure_pursuit_interactive.py",
                    name="pure_pursuit_interactive",
                    params={'is_sim': is_sim})

    if is_sim:
        # Start two SVEAs (svea_a and svea_b) in simulation, each with its own pure_pursuit_interactive node

        INITIAL_POSES = {
            "svea_a": (initial_pose_x, initial_pose_y, initial_pose_a),
            # "svea_b": (0.0, 0.0, 0.0),
        }

        for name, (init_x, init_y, init_a) in INITIAL_POSES.items():
            
            bl.include("custom_svea_core", "svea_tum.launch.py",
                       name=name,
                       is_sim=is_sim,
                       is_indoor=True,
                       map_name=MAP_NAME,
                       initial_pose_x=init_x,
                       initial_pose_y=init_y,
                       initial_pose_a=init_a)

            # Add namespace to pure_pursuit_interactive node so that it can be run for each SVEA independently
            with bl.group(name):

                bl.node("custom_svea_examples", "pure_pursuit_interactive.py",
                        name="pure_pursuit_interactive",
                        params={
                            "is_sim": is_sim,
                            "localization/base_frame": f"{name}/base_link",
                        })
    if use_foxglove:
        bl.include("foxglove_bridge", "foxglove_bridge_launch.xml")