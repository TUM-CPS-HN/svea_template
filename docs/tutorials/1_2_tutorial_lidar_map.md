# Lidar Map Tutorial

## How to Scan a Map

### I. Preliminary Setup

1. Connect your laptop and the SVEA to the same network (campus WiFi doesn't work).

2. Open two terminals and SSH into the corresponding SVEA car from both, using `ssh user@sveax.local`. For example, to SSH into svea1:
```bash
ssh user@svea1.local
```

3. Connect the remote control to the SVEA.

4. In one terminal, `cd` into the SVEA workspace and launch the low-level interface:
```bash
util/run
ros2 launch svea_core lli.xml
```

If something starts counting down, stop the program, press RST on the STM32 board, and reconnect the remote control before launching again. If nothing counts down, press Ctrl+C and continue.

### II. Scan and Save the Map

1. Run:
```bash
ros2 launch svea_localization slam.launch.py use_two_encoders:=False use_foxglove:=True
```

2. Open Foxglove and change `localhost` to `sveax.local` (e.g. `svea1.local` for svea1) to connect. Click **Add panel** in the top right, then select **Transform Tree**. It should look like this:

![Transform tree](../media/transform_tree.png)

Check that the vehicle's position isn't drifting in the transform tree; it's fine if only a small part of the map has appeared so far. Then use the remote control to drive the vehicle around and scan the map.

> **Open question:** how to avoid the black noise lines in the map?

3. In another SSH'd terminal, run `util/exec`, then save the map (remember to change `map_name`):
```bash
ros2 run nav2_map_server map_saver_cli -f /svea_ws/src/svea_core/maps/map_name --ros-args -p save_map_timeout:=10.0
```
You should then find the saved map in `src/svea_core/maps`.(a .pgm file and a .yaml file)

## How to Build an Obstacle Map

The scanned map uses black and grey pixels to show which areas are free and which are occupied by obstacles, but the simulator has no way of interpreting that directly. So you need to manually build an obstacle map that defines each obstacle as a polygon.

1. Connect the SVEA to a monitor via HDMI. `cd` into the SVEA workspace, but do not enter the Docker container. Open the GUI with:
```bash
python3 src/svea_core/util/obstacles_builder_gui.py
```

2. Click **Load map**, click **Add new polygon**, click on the obstacle's corner points, then click **Export** to create the obstacle map.

3. Open `svea_examples/launch/floor2.launch.py` and change `MAP_NAME` at the top of the file. The SVEA will now run using the new map.
