#! /usr/bin/env python3
"""Offline analysis of the eco-drive planner (no ROS required).

Generates the figures used in docs/tutorials/2_eco_drive_solution.md:

    python3 eco_drive_analysis.py all            # everything below
    python3 eco_drive_analysis.py spacetime      # s-t diagram, eco vs naive, one light
    python3 eco_drive_analysis.py corridor       # bonus: three lights on a straight
    python3 eco_drive_analysis.py lap            # bonus: full lap on the TUM map, four lights
    python3 eco_drive_analysis.py pareto         # time/energy trade-off sweep
    python3 eco_drive_analysis.py montecarlo     # randomized safety + energy statistics
    python3 eco_drive_analysis.py loadtest       # 4-32 random lights per lap, close clusters
    python3 eco_drive_analysis.py log run.csv    # plot a log recorded by eco_drive.py

Figures go to ``--out`` (default: docs/media/eco_drive next to the repo root,
or the current directory when run from an installed package).
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

try:  # inside the ROS workspace
    from custom_svea_core.eco_drive import (EcoDrivePlanner, LightTiming, PlannerConfig,
                                            VehicleLimits, Weights, Gate, green_windows,
                                            random_layout, phase_at_offset)
    from custom_svea_core.eco_drive.sim import LightSpec, run_episode
except Exception:  # plain laptop: import the package straight from the source tree
    _src = Path(__file__).resolve().parents[2] / "custom_svea_core" / "custom_svea_core"
    sys.path.insert(0, str(_src))
    from eco_drive import (EcoDrivePlanner, LightTiming, PlannerConfig,  # type: ignore
                           VehicleLimits, Weights, Gate, green_windows,
                           random_layout, phase_at_offset)
    from eco_drive.sim import LightSpec, run_episode  # type: ignore

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

LIGHT_COLORS = {"Rd": "#d62728", "YG": "#f2b400", "Gr": "#2ca02c", "YR": "#f2b400"}
ECO, NAIVE, PLAN = "#1f6feb", "#8b949e", "#1f6feb"

plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 9.5, "axes.titlesize": 10.5,
    "legend.frameon": False,
})


def default_out() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "docs" / "media" / "eco_drive" if (root / "docs").is_dir() else Path.cwd()


# --------------------------------------------------------------------------- helpers

def draw_light_bands(ax, episode, lights, t_end, height=0.35):
    """Colored bars at each light's position showing its actual state over time."""
    for L in lights:
        hist = episode.lights[L.name].history
        times = [t for t, _ in hist] + [max(t_end, hist[-1][0])]
        for (t0, st), t1 in zip(hist, times[1:]):
            t0c, t1c = max(t0, 0.0), min(t1, t_end)
            if t1c <= t0c:
                continue
            ax.fill_between([t0c, t1c], L.s - height / 2, L.s + height / 2,
                            color=LIGHT_COLORS[st], alpha=0.85, lw=0, zorder=1)
        ax.annotate(L.name, (0, L.s + height / 2), xytext=(3, 2), textcoords="offset points",
                    va="bottom", fontsize=7.5, color="#444")


def colored_line(ax, x, y, c, cmap="viridis", vmin=0, vmax=1.2, lw=2.2):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(vmin, vmax), lw=lw, zorder=3)
    lc.set_array(np.asarray(c)[:-1])
    ax.add_collection(lc)
    return lc


def episode_summary(ep):
    return (f"arrival {ep.arrival_time:5.2f} s | ∫a²dt {ep.acc_cost:5.2f} | "
            f"battery {ep.battery_energy:5.1f} J | max|a| {ep.max_abs_accel:4.2f} | "
            f"violations {len(ep.violations)}")


def clearance_margin(ep, light, clear_dist=0.6):
    """Seconds between the rear of the car clearing the light and the end of green.

    Negative means the car was still in the intersection after green ended
    (or entered on red/yellow).
    """
    if light.name not in ep.crossings:
        return np.nan
    t_in = ep.crossings[light.name][0]
    k = np.searchsorted(ep.s, light.s + clear_dist)
    t_out = ep.t[min(k, len(ep.t) - 1)]
    hist = ep.lights[light.name].history
    ends = [t for t, _ in hist[1:]] + [np.inf]
    for (t0, st), t1 in zip(hist, ends):
        if st == "Gr" and t0 <= t_in <= t1:
            return t1 - t_out
    return -np.inf


def make_planner(**kw):
    return EcoDrivePlanner(kw.pop("limits", VehicleLimits()), kw.pop("weights", Weights()),
                           kw.pop("config", PlannerConfig()))


# --------------------------------------------------------------------------- figures

def fig_spacetime(out: Path, lights=None, s_goal=15.0, name="spacetime", title=None):
    lights = lights or [LightSpec("light_a", 8.0, LightTiming(), "Gr", 3.0)]
    planner = make_planner()
    eco = run_episode(lights, s_goal, planner, mode="eco")
    naive = run_episode(lights, s_goal, planner, mode="naive")
    print(f"[{name}] eco   :", episode_summary(eco))
    print(f"[{name}] naive :", episode_summary(naive))
    t_end = max(eco.t[-1], naive.t[-1]) + 0.5

    fig = plt.figure(figsize=(10, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.7, 1], height_ratios=[1, 1, 1], hspace=0.45, wspace=0.25)
    ax = fig.add_subplot(gs[:, 0])
    draw_light_bands(ax, eco, lights, t_end)
    ax.plot(naive.t, naive.s, color=NAIVE, lw=1.8, label="naive (1.2 m/s + brake at red)", zorder=2)
    for tp, pl in eco.plans[::4]:
        ax.plot(tp + pl.t, pl.s, color=PLAN, lw=0.6, alpha=0.18, zorder=2)
    tp0, pl0 = eco.plans[0]
    ax.plot(tp0 + pl0.t, pl0.s, "--", color="k", lw=1.0, alpha=0.7, label="first plan (t = 0)", zorder=4)
    lc = colored_line(ax, eco.t, eco.s, eco.v)
    ax.plot([], [], color="#3b8", lw=2.2, label="eco-drive (executed, colour = speed)")
    for L in lights:
        if L.name in eco.crossings:
            tc, _ = eco.crossings[L.name]
            ax.plot(tc, L.s, "o", ms=6, mfc="white", mec="k", zorder=5)
    ax.set_xlim(0, t_end)
    ax.set_ylim(-0.3, s_goal + 0.6)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("arc length s [m]")
    ax.set_title(title or "Space–time diagram: the plan threads the green window")
    cax = ax.inset_axes([0.62, 0.06, 0.33, 0.025])
    cb = fig.colorbar(lc, cax=cax, orientation="horizontal")
    cb.ax.set_title("eco speed [m/s]", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", fontsize=8)

    def smooth(ep, key):
        y = getattr(ep, key)
        if key != "a":
            return y
        k = max(1, int(round(0.25 / np.median(np.diff(ep.t)))))  # 0.25 s moving average
        return np.convolve(y, np.ones(k) / k, mode="same")

    for i, (key, ylabel) in enumerate([("v", "v [m/s]"), ("a", "a [m/s²] (0.25 s avg)")]):
        a2 = fig.add_subplot(gs[i, 1])
        a2.plot(naive.t, smooth(naive, key), color=NAIVE, lw=1.2, label="naive")
        a2.plot(eco.t, smooth(eco, key), color=ECO, lw=1.4, label="eco")
        if key == "v":
            a2.axhline(1.2, color="k", ls=":", lw=0.8)
            a2.text(0.2, 1.22, "speed limit", fontsize=7)
        a2.set_ylabel(ylabel)
        a2.set_xlim(0, t_end)
        if i == 0:
            a2.legend(fontsize=8, loc="lower right")

    a3 = fig.add_subplot(gs[2, 1])
    labels = ["arrival [s]", "∫a²dt [m²/s³]", "battery [J/10]"]
    e_vals = [eco.arrival_time, eco.acc_cost, eco.battery_energy / 10]
    n_vals = [naive.arrival_time, naive.acc_cost, naive.battery_energy / 10]
    x = np.arange(3)
    a3.bar(x - 0.2, n_vals, 0.4, color=NAIVE, label="naive")
    a3.bar(x + 0.2, e_vals, 0.4, color=ECO, label="eco")
    for xi, (e, n) in enumerate(zip(e_vals, n_vals)):
        a3.text(xi + 0.2, e, f"{100 * (e - n) / n:+.0f}%", ha="center", va="bottom", fontsize=7.5)
    a3.set_xticks(x, labels, fontsize=8)
    a3.grid(axis="x", visible=False)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    return eco, naive


def fig_corridor(out: Path):
    tm = LightTiming()
    lights = [LightSpec("light_a", 7.0, tm, "Rd", 2.0), LightSpec("light_b", 15.0, tm, "Gr", 1.0),
              LightSpec("light_c", 22.0, tm, "YR", 1.0)]
    return fig_spacetime(out, lights, s_goal=28.0, name="corridor",
                         title="Bonus: three lights, one optimisation (green-wave)")


LAP_POINTS = [-0.38, 0.96, 15.0, -2.69, 18.13, 10.64, 3.22, 14.32]  # eco_drive.launch.py
LAP_LIGHTS = {"light_a": (7.0, -0.79), "light_b": (16.57, 3.98),
              "light_c": (10.68, 12.48), "light_d": (1.42, 7.64)}


def _load_map(name="tum_d_floor2"):
    """Occupancy map scanned in Tutorial 1 (pgm + yaml), or None if not found."""
    maps = Path(__file__).resolve().parents[2] / "custom_svea_core" / "maps"
    try:
        from PIL import Image
        meta = dict(line.split(":", 1) for line in (maps / f"{name}.yaml").read_text().splitlines() if ":" in line)
        res = float(meta["resolution"])
        ox, oy = [float(v) for v in meta["origin"].strip(" []").split(",")[:2]]
        img = np.asarray(Image.open(maps / f"{name}.pgm"))
        h, w = img.shape[:2]
        return img, [ox, ox + w * res, oy, oy + h * res]
    except Exception as e:  # noqa: BLE001
        print("map not available:", e)
        return None


def fig_lap(out: Path):
    """Bonus: a full lap on the TUM floor-2 map, four lights, curvature speed limits."""
    try:
        from custom_svea_core.eco_drive import Route
    except Exception:
        from eco_drive import Route  # type: ignore
    route = Route(np.reshape(LAP_POINTS, (-1, 2)), closed=True, fillet_radius=1.0)
    lim = VehicleLimits()
    planner = EcoDrivePlanner(lim, Weights(), PlannerConfig(),
                              speed_limit_fn=lambda s: route.speed_limit(s, lim.v_max, lim.a_lat_max))
    tm = LightTiming()
    # as launched with wait_for_start: all lights start red together when the first plan exists
    lights = [LightSpec(n, route.project(x, y)[0], tm, "Rd", tm.red) for n, (x, y) in LAP_LIGHTS.items()]
    eco = run_episode(lights, route.length, planner, mode="eco", t_max=150)
    naive = run_episode(lights, route.length, planner, mode="naive", t_max=150)
    print("[lap] eco   :", episode_summary(eco))
    print("[lap] naive :", episode_summary(naive))

    fig, (ax_m, ax_st) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
    m = _load_map()
    if m is not None:
        ax_m.imshow(m[0], cmap="gray", extent=m[1], origin="upper", alpha=0.55, interpolation="nearest")
    x, y = route.xy(eco.s)
    lc = colored_line(ax_m, np.asarray(x), np.asarray(y), eco.v, lw=3.0)
    vlim = route.speed_limit(route.s, lim.v_max, lim.a_lat_max)
    corner = vlim < lim.v_max - 1e-3
    ax_m.scatter(route.x[corner], route.y[corner], s=2, color="#e36209", zorder=4, label="curvature speed limit")
    for n, (lx, ly) in LAP_LIGHTS.items():
        ax_m.plot(lx, ly, "o", ms=9, mfc=LIGHT_COLORS["Gr"], mec="k", zorder=5)
        ax_m.annotate(n, (lx, ly), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax_m.plot(*route.xy(0.0), "k>", ms=8, zorder=5, label="start / finish")
    pad = 2.5
    ax_m.set_xlim(route.x.min() - pad, route.x.max() + pad)
    ax_m.set_ylim(route.y.min() - pad, route.y.max() + pad)
    ax_m.set_aspect("equal")
    ax_m.grid(False)
    ax_m.set_xlabel("x [m]"), ax_m.set_ylabel("y [m]")
    ax_m.set_title("One lap on the scanned TUM floor-2 map (colour = speed)")
    ax_m.legend(fontsize=7.5, loc="lower right")
    fig.colorbar(lc, ax=ax_m, fraction=0.04, pad=0.01).set_label("speed [m/s]")

    t_end = max(eco.t[-1], naive.t[-1]) + 0.5
    draw_light_bands(ax_st, eco, lights, t_end, height=0.9)
    ax_st.plot(naive.t, naive.s, color=NAIVE, lw=1.6, label=f"naive: {naive.arrival_time:.1f} s, "
               f"{naive.battery_energy:.0f} J")
    colored_line(ax_st, eco.t, eco.s, eco.v)
    ax_st.plot([], [], color="#3b8", lw=2.2, label=f"eco: {eco.arrival_time:.1f} s, {eco.battery_energy:.0f} J")
    for L in lights:
        if L.name in eco.crossings:
            ax_st.plot(eco.crossings[L.name][0], L.s, "o", ms=5, mfc="white", mec="k", zorder=5)
    ax_st.set_xlim(0, t_end), ax_st.set_ylim(-0.5, route.length + 1)
    ax_st.set_xlabel("time [s]"), ax_st.set_ylabel("arc length s [m]")
    ax_st.set_title("Space–time diagram of the lap")
    ax_st.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "lap.png", bbox_inches="tight")
    plt.close(fig)
    return eco, naive


def fig_pareto(out: Path):
    """Sweep the energy weight; show the time/energy trade-off and speed profiles."""
    tm = LightTiming()
    gates = [Gate("light_a", 6.0, green_windows("Rd", 1.0, tm))]
    w_accs = np.logspace(-2, 2.3, 26)
    rows = []
    for w in w_accs:
        pl = make_planner(weights=Weights(time=1.0, acc=w, jerk=0.1 * w)).plan(0.0, 0.0, gates, s_goal=15.0)
        rows.append((w, pl))
    # also: hard cap on |a| instead of a penalty (objective 3 of the task)
    caps = [1.0, 0.5, 0.3, 0.2, 0.15]
    cap_rows = []
    for c in caps:
        pl = make_planner(limits=VehicleLimits(a_max=c, a_min=-c), weights=Weights(time=1.0, acc=1e-3, jerk=1e-3)) \
            .plan(0.0, 0.0, gates, s_goal=15.0)
        cap_rows.append((c, pl))

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    cmap = plt.get_cmap("plasma")
    norm = matplotlib.colors.LogNorm(w_accs[0], w_accs[-1])
    for w, pl in rows:
        axs[0].plot(pl.t, pl.v, color=cmap(norm(w)), lw=1.1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axs[0], label="energy weight w_acc (w_time = 1)")
    axs[0].set_xlabel("time [s]"), axs[0].set_ylabel("planned v [m/s]")
    axs[0].set_title("Speed profiles vs. energy weight")
    w0 = green_windows("Rd", 1.0, tm)
    for w in w0[:2]:
        axs[0].axvspan(w.start, w.end, color=LIGHT_COLORS["Gr"], alpha=0.12, lw=0)

    T = np.array([pl.duration for _, pl in rows])
    A = np.array([pl.acc_cost for _, pl in rows])
    E = np.array([pl.battery_energy for _, pl in rows])
    axs[1].scatter(T, A, c=w_accs, cmap=cmap, norm=norm, s=22, zorder=3, label="penalty  w_acc·∫a²")
    axs[1].plot(T, A, color="#999", lw=0.8, zorder=2)
    Tc = [pl.duration for _, pl in cap_rows]
    Ac = [pl.acc_cost for _, pl in cap_rows]
    axs[1].plot(Tc, Ac, "s--", color="k", ms=4, lw=0.8, label="hard cap |a| ≤ a_cap")
    for (c, pl) in cap_rows:
        axs[1].annotate(f"{c}", (pl.duration, pl.acc_cost), xytext=(4, 3), textcoords="offset points", fontsize=7)
    d = Weights()
    pl_d = make_planner().plan(0.0, 0.0, gates, s_goal=15.0)
    axs[1].plot(pl_d.duration, pl_d.acc_cost, "*", ms=13, color=ECO, mec="k", zorder=4, label="our default")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("arrival time [s]"), axs[1].set_ylabel("∫a²dt  (log)")
    axs[1].set_title("Pareto front: time vs. energy proxy")
    axs[1].legend(fontsize=7.5)

    axs[2].plot(T, E, "o-", color="#555", ms=3, lw=0.8)
    axs[2].plot(pl_d.duration, pl_d.battery_energy, "*", ms=13, color=ECO, mec="k")
    axs[2].set_xlabel("arrival time [s]"), axs[2].set_ylabel("battery energy model [J]")
    axs[2].set_title("Same sweep, physical energy estimate")
    fig.tight_layout()
    fig.savefig(out / "pareto.png", bbox_inches="tight")
    plt.close(fig)

    print("[pareto]  w_acc   arrival   ∫a²dt   battery   window")
    for w, pl in rows[::3]:
        win = list(pl.crossings.values())[0].index if pl.crossings else "-"
        print(f"         {w:7.3f}  {pl.duration:6.2f}  {pl.acc_cost:7.3f}  {pl.battery_energy:7.2f}   #{win}")
    print(f"[pareto]  default  {pl_d.duration:6.2f}  {pl_d.acc_cost:7.3f}  {pl_d.battery_energy:7.2f}"
          f"  (w_time={d.time}, w_acc={d.acc}, w_jerk={d.jerk})")


def fig_montecarlo(out: Path, n: int = 60, seed: int = 7):
    """Random light phases/positions; eco vs naive. Also a jitter + noise stress test."""
    rng = random.Random(seed)
    planner = make_planner()
    res = {k: [] for k in ("eco", "naive", "eco_stress", "naive_stress")}
    t0 = time.time()
    for i in range(n):
        s_light = rng.uniform(4.0, 11.0)
        st = rng.choice(["Rd", "YG", "Gr", "YR"])
        tl = rng.uniform(0.05, LightTiming().nominal(st))
        for stress in (False, True):
            tm = LightTiming(alpha=1.0 if stress else 0.0)
            lights = [LightSpec("light_a", s_light, tm, st, tl)]
            for mode in ("eco", "naive"):
                ep = run_episode(lights, 15.0, planner, mode=mode, seed=i,
                                 speed_noise=0.05 if stress else 0.0)
                key = mode + ("_stress" if stress else "")
                res[key].append((ep.arrival_time, ep.acc_cost, ep.battery_energy, len(ep.violations),
                                 ep.guard_interventions, ep.max_abs_accel,
                                 np.max(ep.solve_times) if ep.solve_times else 0.0,
                                 clearance_margin(ep, lights[0])))
    print(f"[montecarlo] {n} scenarios x 4 runs in {time.time() - t0:.0f} s")
    R = {k: np.array(v) for k, v in res.items()}
    for k, r in R.items():
        print(f"  {k:13s} arrival {np.nanmean(r[:, 0]):5.2f} s | ∫a² {r[:, 1].mean():6.3f} | "
              f"battery {r[:, 2].mean():5.1f} J | max|a| {r[:, 5].mean():4.2f} | "
              f"violations {int(r[:, 3].sum())}/{len(r)} runs | guard ticks {int(r[:, 4].sum())} | "
              f"worst solve {1e3 * r[:, 6].max():.0f} ms")

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    e, nv = R["eco"], R["naive"]
    axs[0].scatter(nv[:, 2], e[:, 2], s=14, color=ECO)
    lim = [min(e[:, 2].min(), nv[:, 2].min()) - 1, max(e[:, 2].max(), nv[:, 2].max()) + 1]
    axs[0].plot(lim, lim, "k:", lw=0.8)
    axs[0].set_xlabel("naive battery energy [J]"), axs[0].set_ylabel("eco battery energy [J]")
    axs[0].set_title(f"Energy per trip (below the line = saved)\nmean saving "
                     f"{100 * (1 - e[:, 2].mean() / nv[:, 2].mean()):.0f}%")
    axs[1].scatter(nv[:, 0], e[:, 0], s=14, color=ECO)
    lim = [min(e[:, 0].min(), nv[:, 0].min()) - 0.5, max(e[:, 0].max(), nv[:, 0].max()) + 0.5]
    axs[1].plot(lim, lim, "k:", lw=0.8)
    axs[1].set_xlabel("naive arrival [s]"), axs[1].set_ylabel("eco arrival [s]")
    axs[1].set_title(f"Trip time (mean +{e[:, 0].mean() - nv[:, 0].mean():.2f} s)")
    bins = np.linspace(-2.5, 5.0, 31)
    for key, color, hatch in [("naive", NAIVE, None), ("naive_stress", NAIVE, "//"),
                              ("eco", ECO, None), ("eco_stress", ECO, "//")]:
        m = R[key][:, 7]
        m = np.clip(m[np.isfinite(m) | (m < 0)], -2.4, 4.9)
        axs[2].hist(m, bins=bins, histtype="step", lw=1.4, color=color, hatch=hatch,
                    label=f"{key.replace('_stress', ' +jitter+noise')}  (viol. {int(R[key][:, 3].sum())})")
    axs[2].axvline(0, color=LIGHT_COLORS["Rd"], lw=1)
    axs[2].text(-0.1, axs[2].get_ylim()[1] * 0.92, "green ends\nwith car inside ←", ha="right",
                fontsize=7, color=LIGHT_COLORS["Rd"])
    axs[2].set_xlabel("time from rear clearing the light to end of green [s]")
    axs[2].set_ylabel("runs")
    axs[2].set_title(f"Safety margin over {n} random scenarios")
    axs[2].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "montecarlo.png", bbox_inches="tight")
    plt.close(fig)
    return R


LOAD_LEVELS = (4, 8, 16, 24, 32)
LAP_LENGTH = 56.98  # length of the filleted lap in eco_drive.launch.py [m]


def _load_layout(n, seed, alpha, coordinated=True):
    tm = LightTiming(alpha=alpha)
    lay = random_layout(LAP_LENGTH, n, seed, tm, closed=True, coordinate_clusters=coordinated)
    return [LightSpec(p.name, p.s, tm, *phase_at_offset(p.offset, tm)) for p in lay]


def _load_run(args):
    """One load-test episode (top level so it can run in a worker process)."""
    n, seed, mode, stress, coordinated = args
    lights = _load_layout(n, seed, 0.5 if stress else 0.0, coordinated)
    planner = make_planner() if mode == "eco" else None
    kw = dict(actuator_gain=1 / 1.27, actuator_dropout=True, speed_noise=0.03) if stress else {}
    t0 = time.time()
    ep = run_episode(lights, LAP_LENGTH, planner, mode=mode, seed=seed, t_max=300.0, **kw)
    gaps = np.diff([L.s for L in lights])
    return dict(n=n, seed=seed, mode=mode, stress=stress, coordinated=coordinated,
                arrival=ep.arrival_time, energy=ep.battery_energy, acc=ep.acc_cost,
                violations=len(ep.violations), guard=ep.guard_interventions,
                solve=np.array(ep.solve_times), failed=sum(p.status == "failed" for _, p in ep.plans),
                min_gap=float(gaps.min()) if len(gaps) else np.nan,
                close_pairs=int(np.sum(gaps < 0.9)), wall=time.time() - t0,
                episode=ep if (n == max(LOAD_LEVELS) and seed == 0 and not stress and coordinated) else None,
                lights=lights)


def fig_loadtest(out: Path, seeds: int = 6):
    """Load test: many randomly placed, partly very close lights on one lap."""
    jobs = [(n, sd, mode, stress, True) for n in LOAD_LEVELS for sd in range(seeds)
            for mode, stress in (("eco", False), ("eco", True), ("naive", False))]
    # adversarial: clusters with random (uncoordinated) phases, which may be impassable
    jobs += [(16, sd, "eco", False, False) for sd in range(seeds)]
    t0 = time.time()
    # each worker caches up to 24 NLPs (~0.3 GB); keep the pool small enough for an 8 GB container
    with ProcessPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2) - 1))) as ex:
        res = list(ex.map(_load_run, jobs))
    print(f"[loadtest] {len(jobs)} episodes in {time.time() - t0:.0f} s")

    def sel(mode, stress, coordinated=True):
        return [r for r in res if r["mode"] == mode and r["stress"] == stress and r["coordinated"] == coordinated]

    print(f"  {'lights':>6} {'close pairs':>11} {'min gap':>7} | {'eco t':>6} {'naive t':>7} "
          f"{'eco J':>6} {'naive J':>7} | {'solve med':>9} {'p95':>5} {'max':>5} | "
          f"{'viol eco/stress/naive':>21} {'guard':>5} {'stress t':>8}")
    rows = []
    for n in LOAD_LEVELS:
        e = [r for r in sel("eco", False) if r["n"] == n]
        st = [r for r in sel("eco", True) if r["n"] == n]
        nv = [r for r in sel("naive", False) if r["n"] == n]
        solve = 1e3 * np.concatenate([r["solve"] for r in e + st])
        row = dict(n=n, close=np.mean([r["close_pairs"] for r in e]), gap=np.min([r["min_gap"] for r in e]),
                   t_eco=np.mean([r["arrival"] for r in e]), t_naive=np.mean([r["arrival"] for r in nv]),
                   t_stress=np.mean([r["arrival"] for r in st]),
                   J_eco=np.mean([r["energy"] for r in e]), J_naive=np.mean([r["energy"] for r in nv]),
                   med=np.median(solve), p95=np.percentile(solve, 95), mx=solve.max(),
                   v=(sum(r["violations"] for r in e), sum(r["violations"] for r in st),
                      sum(r["violations"] for r in nv)),
                   guard=sum(r["guard"] for r in e + st), solve=solve,
                   done=sum(np.isfinite(r["arrival"]) for r in e + st), runs=len(e + st))
        rows.append(row)
        print(f"  {n:6d} {row['close']:11.1f} {row['gap']:6.2f}m | {row['t_eco']:5.1f}s {row['t_naive']:6.1f}s "
              f"{row['J_eco']:5.0f}J {row['J_naive']:6.0f}J | {row['med']:7.0f}ms {row['p95']:4.0f} {row['mx']:5.0f} | "
              f"{'%d / %d / %d' % row['v']:>21} {row['guard']:5d} {row['t_stress']:7.1f}s")
    adv = sel("eco", False, False)
    adv_done = sum(np.isfinite(r["arrival"]) for r in adv)
    print(f"  adversarial (16 lights, uncoordinated clusters): {adv_done}/{len(adv)} finished, "
          f"violations {sum(r['violations'] for r in adv)}, guard ticks {sum(r['guard'] for r in adv)}")

    fig = plt.figure(figsize=(13, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.38, wspace=0.28)

    # (a) s-t diagram of the densest scenario
    ax = fig.add_subplot(gs[0, :])
    hero = next(r for r in res if r["episode"] is not None)
    ep, lights = hero["episode"], hero["lights"]
    nv = next(r for r in sel("naive", False) if r["n"] == hero["n"] and r["seed"] == 0)
    t_end = ep.arrival_time + 2
    draw_light_bands(ax, ep, lights, t_end, height=0.28)
    for txt in list(ax.texts):
        txt.remove()
    lc = colored_line(ax, ep.t, ep.s, ep.v, lw=2.4)
    ax.set_xlim(0, t_end), ax.set_ylim(-0.5, LAP_LENGTH + 0.5)
    ax.set_xlabel("time [s]"), ax.set_ylabel("distance along the lap [m]")
    ax.set_title(f"(a) {len(lights)} random lights on one lap, {hero['close_pairs']} pairs closer than "
                 f"0.9 m (min {hero['min_gap']:.2f} m): arrival {ep.arrival_time:.1f} s, "
                 f"{hero['violations']} red-light violations")
    fig.colorbar(lc, ax=ax, pad=0.01, label="speed [m/s]")

    # (b) solve time vs number of lights
    ax = fig.add_subplot(gs[1, 0])
    ax.boxplot([r["solve"] for r in rows], positions=range(len(rows)), widths=0.55, whis=(5, 95),
               showfliers=False, medianprops=dict(color=ECO))
    ax.scatter(range(len(rows)), [r["mx"] for r in rows], marker="v", color="#444", s=16, zorder=3, label="max")
    ax.axhline(500, color=LIGHT_COLORS["Rd"], lw=1, ls="--", label="replan period")
    ax.set_xticks(range(len(rows)), [str(r["n"]) for r in rows])
    ax.set_xlabel("lights on the lap"), ax.set_ylabel("planner time per replan [ms]")
    ax.set_title("(b) Solve time (box 5-95 %)"), ax.legend(fontsize=7, loc="upper left")

    # (c) trip time
    ax = fig.add_subplot(gs[1, 1])
    ns = [r["n"] for r in rows]
    ax.plot(ns, [r["t_naive"] for r in rows], "o-", color=NAIVE, label="naive (full speed, brake at red)")
    ax.plot(ns, [r["t_eco"] for r in rows], "o-", color=ECO, label="eco")
    ax.plot(ns, [r["t_stress"] for r in rows], "o--", color=ECO, mfc="white",
            label="eco, jitter ±0.5 s + laggy actuator")
    ax.axhline(LAP_LENGTH / 1.2, color="#444", lw=0.8, ls=":", label="no lights, 1.2 m/s")
    ax.set_xlabel("lights on the lap"), ax.set_ylabel("lap time [s]")
    ax.set_title("(c) Lap time"), ax.legend(fontsize=7)

    # (d) energy
    ax = fig.add_subplot(gs[1, 2])
    w = 0.38
    x = np.arange(len(rows))
    ax.bar(x - w / 2, [r["J_naive"] for r in rows], w, color=NAIVE, label="naive")
    ax.bar(x + w / 2, [r["J_eco"] for r in rows], w, color=ECO, label="eco")
    for i, r in enumerate(rows):
        ax.text(i + w / 2, r["J_eco"], f"−{100 * (1 - r['J_eco'] / r['J_naive']):.0f}%",
                ha="center", va="bottom", fontsize=7, color=ECO)
    ax.set_xticks(x, [str(n) for n in ns])
    ax.set_xlabel("lights on the lap"), ax.set_ylabel("battery energy per lap [J]")
    tot_v = sum(sum(r["v"][:2]) for r in rows)
    ax.set_title(f"(d) Energy  ·  eco violations: {tot_v} in {sum(r['runs'] for r in rows)} runs")
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(out / "loadtest.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def fig_log(out: Path, path: Path):
    """Plot a CSV written by the ROS node (simulation or real car)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("empty log")
        return
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    t = col("t")
    fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    lights = [k[: -len("_state")] for k in rows[0] if k.endswith("_state")]
    for name in lights:
        s_l = float(rows[0][f"{name}_s"])
        st = [r[f"{name}_state"] for r in rows]
        for i in range(len(t) - 1):
            axs[0].fill_between([t[i], t[i + 1]], s_l - 0.15, s_l + 0.15,
                                color=LIGHT_COLORS.get(st[i], "#ccc"), lw=0)
    axs[0].plot(t, col("s_plan"), "--", color="k", lw=0.8, label="plan")
    axs[0].plot(t, col("s"), color=ECO, lw=1.5, label="measured")
    axs[0].set_ylabel("s [m]"), axs[0].legend()
    axs[1].plot(t, col("v_plan"), "--", color="k", lw=0.8, label="plan")
    axs[1].plot(t, col("v_cmd"), color="#e36209", lw=0.8, label="command")
    axs[1].plot(t, col("v"), color=ECO, lw=1.3, label="measured")
    axs[1].set_ylabel("v [m/s]"), axs[1].legend()
    axs[2].plot(t, col("a_plan"), color=ECO, lw=1.2)
    axs[2].set_ylabel("a plan [m/s²]"), axs[2].set_xlabel("time [s]")
    g = col("guard")
    for i in np.where(g > 0)[0]:
        axs[1].axvline(t[i], color="red", alpha=0.08)
    fig.suptitle(f"{path.name}: guard active in {int((g > 0).sum())} of {len(g)} control ticks")
    fig.tight_layout()
    fig.savefig(out / f"{path.stem}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{path.stem}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["all", "spacetime", "corridor", "lap", "pareto", "montecarlo", "loadtest", "log"])
    ap.add_argument("logfile", nargs="?", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("-n", type=int, default=60, help="Monte Carlo scenarios")
    ap.add_argument("--seeds", type=int, default=6, help="random layouts per load-test level")
    args = ap.parse_args()
    out = args.out or default_out()
    out.mkdir(parents=True, exist_ok=True)
    if args.what == "log":
        fig_log(out, args.logfile)
        return
    if args.what in ("all", "spacetime"):
        fig_spacetime(out)
    if args.what in ("all", "corridor"):
        fig_corridor(out)
    if args.what in ("all", "lap"):
        fig_lap(out)
    if args.what in ("all", "pareto"):
        fig_pareto(out)
    if args.what in ("all", "montecarlo"):
        fig_montecarlo(out, args.n)
    if args.what in ("all", "loadtest"):
        fig_loadtest(out, args.seeds)
    print("figures in", out)


if __name__ == "__main__":
    main()
