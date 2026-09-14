# Adaptive Cruise Control

In this tutorial, you will build a two-vehicle scenario where one SVEA (the **leader**) drives the loop as usual, while a second SVEA (the **follower**) tracks the same loop but paces its speed off the leader instead of driving at a constant target velocity — an Adaptive Cruise Control (ACC) system. 

ACC is a *reactive* controller: at every control step you look at the gap to the leader and decide on a speed right now. Correctness (never rear-ending the leader, never running a red light) matters more than elegance, but exactly how aggressively the follower closes gaps or brakes is your design choice — be ready to justify it.

Most of the files used in this tutorial — the extended leader node and the launch file that ties everything together — are ones you will create yourself in your own package, the same way you did in Tutorials 1 and 2. The one exception is the ACC follower: `svea_examples/scripts/acc_follower.py` is provided as a starter file with the path-following boilerplate filled in and the ACC-specific logic left as TODOs for you to complete.

## Prerequisites

Before starting, ensure you have:

1. Completed [Tutorial 1: Path-Tracking and Pure-Pursuit](1_1_path_tracking.md) — you should be comfortable with the rosonic framework, `PurePursuitController`, and writing your own launch file.
2. Completed [Tutorial 2: Eco-Drive](2_eco_drive.md) — you should already know how `svea_examples/traffic_light.py` works and how to read `/<name>/state` and `/<name>/time_left`.
3. ROS 2 Jazzy installed and workspace sourced, with `svea_core` built.

## Look at the Multi-Vehicle Scaffold That's Already There

Open `src/svea_examples/launch/floor2.launch.py`. When `is_sim=True`, it already loops over an `INITIAL_POSES` dictionary and spawns one SVEA (with its own `svea.launch.py` include and its own `pure_pursuit` node) per entry, each pushed into its own namespace with `bl.group(name)`:

```python
INITIAL_POSES = {
    "svea_a": (initial_pose_x, initial_pose_y, initial_pose_a),
    # "svea_b": (0.0, 0.0, 0.0),
}

for name, (init_x, init_y, init_a) in INITIAL_POSES.items():
    bl.include("svea_core", "svea.launch.py",
               name=name, is_sim=is_sim, is_indoor=True,
               map_name=MAP_NAME,
               initial_pose_x=init_x, initial_pose_y=init_y, initial_pose_a=init_a)

    with bl.group(name):
        bl.node("svea_examples", "pure_pursuit.py",
                name="pure_pursuit",
                params={"points": points, "is_sim": is_sim,
                        "localization/base_frame": f"{name}/base_link"})
```

`svea_b` is commented out precisely because nobody has needed a second vehicle yet — that's the gap this tutorial fills. Two things fall out of this pattern that matter for ACC:

- Every vehicle's odometry ends up on `/<name>/odometry/local` (e.g. `/svea_a/odometry/local`), because `simulation.launch.py` wraps the simulated vehicle in `bl.group(name)` and `LocalizationInterface`'s odometry subscription is the relative topic `odometry/local`.
- A node only ever sees its own vehicle's odometry through `LocalizationInterface` — to see another vehicle's state, you subscribe to that vehicle's topic directly (an absolute topic name starting with `/` bypasses your own node's namespace).

## Task 1: ACC Follower (No Traffic Light)

### The control law

Your follower needs to turn the relative distance and relative velocity to the leader into a target speed for its own `PurePursuitController`, recomputed every control step. Your implementation should:

- converge to a safe, steady cruise speed when the leader is far ahead,
- close an overly large gap smoothly rather than aggressively,
- slow down proportionally to how fast the gap is closing, not just how large it currently is,
- respect `max_velocity` and `max_accel`/`max_decel` bounds so commands don't jump or overshoot.

Treat the gains and any distance/time thresholds you introduce as tunable `rx.Parameter`s rather than hardcoded constants.

Also think about what should happen if the leader's odometry stops arriving (leader out of view, launch order issue, etc.) — a `leader_timeout` parameter that falls back to a conservative behavior (e.g. stop, or cruise at a low fixed speed) is worth adding before you test on hardware.

### Writing the follower node

`svea_examples/scripts/acc_follower.py` is a *starter* file, not a working example — copy it into your own package's `scripts/` folder and fill in the TODOs. The path-following boilerplate (steering via `PurePursuitController`, updating the goal/trajectory) is already filled in, the same as `pure_pursuit.py` from Tutorial 1. What's left for you to implement is the ACC-specific part:

1. **Receive the leader's state independently of your own.** Recall from the namespacing discussion above that `LocalizationInterface` only ever resolves to *your own* vehicle. The starter file has a commented-out `rx.Subscriber` pointed at `leader_odom_topic` — uncomment it, write the callback, and decide what to store from each incoming message (at minimum, the leader's latest position and velocity, and when you last heard from it).
2. **Implement `compute_acc_velocity`**, currently just a `raise NotImplementedError`, satisfying the requirements above.
3. **Wire it into the loop** — the one TODO line in `loop()` already shows where.

The starter file already declares the parameters common to any control law (`cruise_velocity`, `max_velocity`, `max_accel`/`max_decel`, `leader_timeout`, `leader_odom_topic`) — add whatever else your specific control law needs (gains, distance/time thresholds, ...) as `rx.Parameter`s alongside them, so everything stays tunable from the launch file without editing the script.

### Writing the launch file

In your package's `launch/`, write e.g. `acc.launch.py` that spawns two SVEAs the way `floor2.launch.py`'s commented-out `svea_b` block hints at — `svea_a` running the unmodified `svea_examples/pure_pursuit.py` as leader, `svea_b` running your `acc_follower.py`. Give `svea_b` an initial pose a short distance behind `svea_a` along the same `points` loop (both vehicles should use the *same* `points` list — spacing is then maintained by your control law, not by giving them different paths).

### Try it out

```bash
ros2 launch <your_package> acc.launch.py
```

Watch both vehicles loop in Foxglove. The follower should settle into a roughly constant gap behind the leader rather than colliding with it or falling arbitrarily far behind. Try changing the leader's `target_velocity` — the follower should adapt.

## Task 2: A Traffic-Light-Aware Leader

Right now `svea_a` runs the stock `pure_pursuit.py`, which has no idea a light exists — it will happily drive straight through on red. As in Tutorial 1, before changing anything you should make your own editable copy — e.g. `scripts/traffic_pure_pursuit.py` — rather than editing `svea_examples` directly.

Extend it so the vehicle obeys the traffic light instead of driving through it blindly. Extend your launch file to also bring up a light (reuse `svea_examples/traffic_light.py`, as in Tutorial 2).

Verify: the leader should now stop before the light on red/yellow-red, and proceed through on green/yellow-green, without ever crossing on red.

## Task 3: Traffic-Light-Aware Following

At this point the follower only reacts to the *leader* slowing down — which is fragile. If the follower is between the light and the leader when it turns red, it needs to stop too, on its own, not just because it's chasing a leader that has already stopped.

Extend `acc_follower.py` from Task 1 with the same light-awareness you just built for the leader. Traffic-light compliance should take priority over the ACC velocity — the follower is never allowed to run a red light just because that's what the spacing controller happens to want.

Test this deliberately: adjust the timing or spacing so the leader clears the light before it turns red but the follower doesn't. Confirm the follower stops at the light on its own, rather than continuing to follow the leader through it.

## Testing the Solution

### Simulation Testing

1. Run Task 1's `acc.launch.py` and confirm the follower maintains a sensible, non-colliding gap while both vehicles loop.
2. Run Task 2's version and confirm the leader stops precisely for red/yellow-red and proceeds on green/yellow-green.
3. Run Task 3's version and confirm the follower independently respects the light, not just the leader's braking.
4. Try different gain/threshold values and different light positions/timings to make sure your solution isn't overfit to the defaults.

### Real Vehicle Testing

When ready to test on real vehicles, launch with `is_sim:=False`. This tutorial doesn't provide a real leading SVEA to test against — testing the full two-vehicle scenario on hardware is a natural extension option to pursue in your final project.

### Safety Checklist

Before running on real hardware:

- [ ] RC remote is on and ready for override
- [ ] Start with low target/cruise speeds (< 1 m/s)
- [ ] Verify each vehicle's localization independently before combining them
- [ ] Test the leader-timeout fallback with the leader's launch killed
- [ ] Have someone ready to hit the emergency stop for each vehicle

## Expected Outcomes

After completing this tutorial, you should have:

1. **A working ACC follower** that maintains a dynamic, speed-dependent gap to a leader vehicle.
2. **A multi-vehicle launch file** built on the namespacing pattern already scaffolded in `floor2.launch.py`.
3. **A traffic-light-aware leader**, extending your Tutorial 1 pure-pursuit experience with an external velocity constraint.
4. **A follower that independently satisfies two constraints at once** (spacing and traffic law) by combining velocity caps.

