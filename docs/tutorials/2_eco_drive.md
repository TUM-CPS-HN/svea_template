# Eco-Drive

In this tutorial, you will extend the path-tracking setup from [Tutorial 1](1_1_path_tracking.md) by adding a traffic light, and design a trajectory planner that gets the vehicle through the light while it is green — spending as little energy as possible while still reaching the goal quickly. How you define "energy efficiency" is up to you; as a rule of thumb, aggressive acceleration and braking consume more energy than smooth, gradual speed changes. In this tutorial, correctness (never crossing on red) matters more than elegance, but how you weigh energy efficiency against getting there quickly is your call to make — and to justify.

## Prerequisites

Before starting, ensure you have:
1. Completed [Tutorial 1: Path-Tracking and Pure-Pursuit](1_1_path_tracking.md)
2. ROS 2 Jazzy installed and workspace sourced, with `svea_core` built
3. `casadi` available in your Python environment (`python3 -c "import casadi"` should not error)
4. A basic understanding of nonlinear optimization (decision variables, constraints, objective)

## Launch the Environment

The map/vehicle and the traffic light live in two separate launch files, so you can bring up either one on its own while you develop.

```bash
# Terminal 1: map, Foxglove, and the vehicle running the default path follower
ros2 launch svea_examples floor2.launch.py

# Terminal 2: a single traffic light, fixed on the map
ros2 launch svea_examples traffic_light.launch.py name:=light_a pos_x:=3.34 pos_y:=0.90
```

Open Foxglove and you should see the vehicle looping the Floor 2 path, and a colored sphere marker for the light that cycles red -> yellow-green -> green -> yellow-red. With the default arguments, the light sits on the long straight leg of the default path. You can move the light by changing `pos_x`/`pos_y` in the command above, or add more lights by launching `traffic_light.launch.py` again with a different `name` in another terminal.

The light publishes its own state, so you don't need to guess it:

```bash
ros2 topic echo /light_a/state       # one of 'Rd', 'YG', 'Gr', 'YR'
ros2 topic echo /light_a/time_left   # seconds remaining in the current phase
```

Take a look at `src/svea_examples/scripts/traffic_light.py` to see exactly how the cycle is timed (`RED_TIME`, `YELLOW_TIME`, `GREEN_TIME`, and the `TRANSITIONS` table).

As in Tutorial 1, before changing anything, copy the relevant files into your own package rather than editing `svea_examples` directly. The Floor 2 example above is only there to show you how the pieces fit together — since you'll eventually run this in the TUM building, switch to the map you scanned in Tutorial 1.

## Task 1: Trajectory Planning and Tracking

### Problem Formulation

Right now, `pure_pursuit.py` drives at a constant `target_velocity` and has no idea the light exists — it will happily run a red light, or brake as hard as physically possible right at the stop line. Your job is to replace that with a plan that:

1. **Never crosses the light's position while it is red or yellow-red.** It must only be at the light's `s`-coordinate while the state is (or will be) green.
2. **Minimizes energy use**, which in practice means smooth, gentle speed changes rather than aggressive ones.

To simplify the problem Let's first consider in this task that the vehicle only drives on one leg of the corridor. It's reasonable to reduce the problem to one dimension: arc-length `s` along the trajectory, with velocity `v` and acceleration `a` as your other states, rather than full `(x, y, yaw)`. You can always map `s` back to `(x, y)` afterwards since the segment is a straight line.

Then you need to formulate the optimization problem. Before writing any CasADi code, work out the math on paper (or in a comment block). You'll need:

**Decision variables**, over a fixed number of steps `N` spanning a planning horizon:
- `s_k` — position along the segment at step `k`
- `v_k` — velocity at step `k`
- `a_k` — acceleration at step `k`

**Dynamics constraints** (forward Euler is enough):
```
s[k+1] == s[k] + dt * v[k]
v[k+1] == v[k] + dt * a[k]
```

**Bounds:**
- `0 <= v_k <= v_max`
- `a_min <= a_k <= a_max` — this is where your energy/comfort limit lives; tighter bounds mean gentler driving, but may make it harder (or impossible) to reach the light in time for green.

**Boundary conditions:**
- `s_0`, `v_0` fixed to the vehicle's current position and speed
- `s_N` reaching some goal point beyond the light

**The traffic-light constraint** is the interesting part. You need to find the time step at which the planned trajectory crosses the light's position, and constrain that crossing time to fall inside a green window. Some things to think about:
- How do you compute *when* the upcoming green windows are, given the current `/light_a/state` and `/light_a/time_left` (or the fixed `RED_TIME`/`YELLOW_TIME`/`GREEN_TIME` constants)?
- Should you always aim for the very next green window, or is it sometimes cheaper (in energy) to wait for the one after?
- How do you express "trajectory is at position `light_s` at time `t_cross`, and `t_cross` falls in `[green_start, green_end]`" as constraints CasADi can solve?

**Objective.** A natural energy proxy is the sum of squared accelerations over the horizon, e.g. `sum(a_k^2) * dt` — this penalizes both hard acceleration and hard braking, which is exactly what you want for eco-driving. You may also want to add a small jerk term (penalizing `a[k+1] - a[k]`) for extra smoothness, and/or a time term if you want the vehicle to prefer arriving sooner rather than crawling. How you weight these against each other is a design decision — be ready to explain your choice.

### Solve the problem using CasADi
Once you formulate the optimization problem. CasADi's `Opti` stack is a convenient way to build and solve this kind of problem:

```python
import casadi as ca

opti = ca.Opti()
s = opti.variable(N + 1)
v = opti.variable(N + 1)
a = opti.variable(N)

# TODO: dt, bounds, dynamics, boundary conditions,
#       the green-window crossing constraint, and the objective go here

opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})
solution = opti.solve()
```

If you're testing multiple candidate green windows (this cycle vs. later ones), solve the NLP once per candidate and keep whichever feasible solution has the lowest objective value — don't assume the very next green window is always the best (or even reachable) one.

### Track Your Planned Trajectory

An optimized `(t, s, v, a)` profile is only useful if something drives the vehicle along it. You have three reasonable options:

1. Extend `pure_pursuit.py` so that, instead of a constant `target_velocity`, it looks up the planned speed at the current elapsed time and feeds that to the controller.
2. Track it with the generic MPC controller in `svea_core/svea_core/controllers/mpc.py` instead of pure pursuit — see `svea_examples/scripts/mpc_path_tracking.py` for an example of how it's wired up. 
3. Write your own separate small tracking node that samples your planned trajectory over time and sends steering/velocity commands directly.

If your planner already works for the nominal case, try to change the position and timing of traffic light, and check if your implementation still works.

Compare a few different weightings between the time term and the energy term. Plot the resulting speed profiles and their total `sum(a^2) * dt` cost — how much energy do you save by being willing to arrive later?

## Testing the Solution

1. **Visualize before you drive.** Plot your planned `s(t)`, `v(t)`, and `a(t)` with matplotlib before sending anything to the vehicle. Check by eye that the crossing time lands inside a green window and that the acceleration profile stays within your bounds.
2. **Verify in simulation.** Run both launch files together and confirm visually (Foxglove marker color) and via `ros2 topic echo /light_a/state` that the vehicle is only ever at the light's position while it's green.
3. **Verify on real hardware.** Place your laptop where the traffic light is and use its screen to display the light's state, then confirm your planner behaves correctly in the real world.

### Safety Checklist

Before running on real hardware:

- [ ] RC remote is on and ready for override
- [ ] Start with low target speeds (< 1 m/s)
- [ ] Verify localization is working correctly
- [ ] Have someone ready to hit the emergency stop

## Expected Outcomes

After completing this tutorial, you should have:

1. **Optimal control formulation** for crossing a signalized point under timing constraints
2. **Working familiarity with CasADi's `Opti` interface** for building and solving nonlinear programs
3. **A trajectory tracker** that follows a time-parameterized speed profile rather than a constant target velocity
4. **An understanding of the time/energy trade-off** in eco-driving, and a justified choice of objective weights

## Bonus Task

Once your single-light planner works reliably, consider extending it to a corridor with more than one light — e.g. by launching a second `traffic_light.launch.py` instance with a different `name`/`pos_x`/`pos_y` further along the path. The same green-window idea applies, but now your plan has to satisfy multiple crossing constraints at once.

Drive the vehicle around the whole lap with a traffic light on each segment, so the planner has to handle curved segments rather than a single straight line.
