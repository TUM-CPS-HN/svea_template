r"""Eco-drive speed planner: multi-phase, free-time optimal control with CasADi.

Problem
=======

Along the route (1-D, arc length ``s``) the car must pass every traffic light
while it is green, reach the goal quickly, and use little energy. Instead of
the "find the time step where ``s`` crosses the light" formulation, we split
the trajectory into **phases whose end points are fixed positions** and whose
**durations are decision variables** (the free-final-time trick from Problem 2
of the lecture, applied per phase):

.. code-block:: text

      phase 0          phase 1 (commit)   phase 2 (clear)        phase 3
    s0 ───────────── s_stop ─────────── s_light ─────────── s_exit ────────── s_goal
                       │                  │                    │
                  (may wait here)   τ1 ∈ green window     τ2 ≤ window end

Phase ``p`` has ``N_p`` intervals of length ``dt_p = T_p / N_p``. The
cumulative time ``τ_p = Σ_{q≤p} T_q`` at which the car reaches a boundary is
then an ordinary variable, so

    "the car is at the light at a time inside a green window"

becomes the box constraint ``lo ≤ τ_light ≤ hi`` – no integer variables, no
non-smooth crossing-index search. Because ``v ≥ 0`` at every node and the
acceleration is piecewise constant, ``s(t)`` is monotone, hence the car is
*before* the light for all ``t < τ_light`` and *after* it for all
``t > τ_light``. The crossing time is exact, not only exact at grid points.

Decision variables (per phase): ``s_k, v_k`` (N+1), ``a_k`` (N), ``T``.

Dynamics (exact zero-order-hold discretisation of the double integrator)::

    s[k+1] = s[k] + dt v[k] + dt² a[k] / 2
    v[k+1] = v[k] + dt a[k]

Constraints::

    0 ≤ v_k ≤ 1.2                  speed limit (hard)
    v_k ≤ v_curve(s_k) + ε_k       curvature limit (soft, heavily penalised)
    a_min ≤ a_k ≤ a_max            comfort / battery limit
    s_end(phase p) = boundary p    phases glue the position grid together
    lo_p - δ_p ≤ τ_p ≤ hi_p + δ_p  green windows with a 0.3 s margin, of which up to
    0 ≤ δ_p ≤ 0.2 s                0.2 s may be "eaten" at a cost (recursive feasibility)
    v_k ≥ v_cross - σ              in commit/clear phases: no creeping at the line
    s_0 = s(now), v_0 = v(now)

Objective::

    J = w_time · τ_final  +  w_acc · Σ a_k² dt  +  w_jerk · Σ (Δa_k)² / dt
        +  w_slack · (σ² + Σ ε²)  +  w_margin · Σ δ²

Which green window to use for each light is a discrete choice. We enumerate
the candidates that are reachable (checked light by light), sort them by an
exact lower bound on their cost (J >= w_time * earliest possible arrival, since
every other term is >= 0), and solve one NLP per candidate, cheapest bound
first. A candidate whose bound already exceeds the best plan found is skipped
without solving: typically 1-2 NLPs per plan instead of ~7, same optimum. The
NLP is built once per *structure* and re-used with CasADi parameters. If no window is reachable
(e.g. the light is not running) a *hold* plan stops smoothly at the stop line.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

try:
    import casadi as ca
except ImportError as e:  # pragma: no cover - only hit when casadi is missing
    raise ImportError("eco_drive.planner requires casadi (pip install casadi==3.6.7)") from e

from .energy import accel_cost, battery_energy
from .traffic_signal import GreenWindow

__all__ = ["VehicleLimits", "Weights", "PlannerConfig", "Gate", "Plan", "EcoDrivePlanner"]

BIG = 1e4


@dataclass
class VehicleLimits:
    v_max: float = 1.2  # speed limit from the task [m/s]
    a_max: float = 1.0  # hard cap on acceleration [m/s^2]; the energy penalty keeps it gentle
    a_min: float = -0.8  # max (comfortable) deceleration [m/s^2]
    a_lat_max: float = 0.8  # lateral acceleration limit for curves [m/s^2]


@dataclass
class Weights:
    time: float = 1.0  # [1/s]      weight on arrival time
    acc: float = 4.0  # [s^3/m^2]  weight on integral of a^2 (energy proxy)
    jerk: float = 0.5  # weight on integral of jerk^2 (smoothness)
    slack: float = 1e3  # soft "keep moving through the intersection"
    margin: float = 30.0  # penalty for eating into the preferred time margin [1/s]


@dataclass
class PlannerConfig:
    n_free: int = 16  # intervals in a free phase
    n_short: int = 4  # intervals in commit / clear phases
    stop_margin: float = 0.30  # stop line this far before the light [m]
    clear_dist: float = 0.60  # rear of the car must be this far past the light [m]
    time_margin: float = 0.30  # preferred margin to green start / end [s] (soft)
    time_margin_min: float = 0.10  # hard margin; the soft part absorbs tracking errors [s]
    v_cross: float = 0.25  # minimum speed between stop line and clearing point [m/s]
    windows_per_light: int = 4  # candidate green windows considered per light
    max_candidates: int = 12
    max_gates: int = 6  # lights per plan; further ones are left to later replans
    max_combos: int = 64  # window combinations generated per plan (depth-first, earliest first)
    short_free: float = 1.5  # free phases shorter than this use n_short intervals [m]
    horizon: float = 25.0  # planning horizon [m] (the goal is used when closer)
    vlim_ds: float = 0.25  # resolution of the speed-limit lookup table [m]
    max_iter: int = 400
    cache_size: int = 24  # NLPs kept built (LRU); every distinct phase structure is one NLP


@dataclass
class Gate:
    """A traffic light as seen by the planner."""

    name: str
    s: float  # arc length of the light (unrolled, same frame as s0)
    windows: list[GreenWindow]  # guaranteed green windows, relative to now


@dataclass
class Plan:
    t: np.ndarray
    s: np.ndarray
    v: np.ndarray
    a: np.ndarray  # acceleration on [t_k, t_{k+1}); len = len(t) - 1
    cost: float
    status: str  # 'goal' | 'horizon' | 'hold' | 'failed'
    crossings: dict = field(default_factory=dict)  # light name -> (t_cross, window)
    solve_time: float = 0.0
    n_candidates: int = 0
    s0: float = 0.0  # absolute arc length of s[0]

    @property
    def ok(self) -> bool:
        return self.status != "failed"

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    @property
    def acc_cost(self) -> float:
        return accel_cost(self.t, self.a)

    @property
    def battery_energy(self) -> float:
        return battery_energy(self.t, self.v, self.a)

    def sample(self, t):
        """Exact state of the piecewise-constant-acceleration plan at time(s) t."""
        t = np.clip(np.asarray(t, dtype=float), self.t[0], self.t[-1])
        k = np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.a) - 1)
        tau = t - self.t[k]
        a = self.a[k]
        v = self.v[k] + a * tau
        s = self.s[k] + self.v[k] * tau + 0.5 * a * tau**2
        return s, v, a

    def time_at(self, s_query: float) -> float:
        """First time at which the plan reaches ``s_query`` (inf if never)."""
        if s_query <= self.s[0]:
            return 0.0
        if s_query > self.s[-1] + 1e-9:
            return math.inf
        k = int(np.searchsorted(self.s, s_query, side="left")) - 1
        k = min(max(k, 0), len(self.a) - 1)
        v, a, ds = self.v[k], self.a[k], s_query - self.s[k]
        if abs(a) < 1e-9:
            tau = ds / max(v, 1e-9)
        else:
            disc = max(v * v + 2 * a * ds, 0.0)
            tau = (-v + math.sqrt(disc)) / a
        return float(self.t[k] + tau)


# --------------------------------------------------------------------------- #


@dataclass
class _Boundary:
    s: float  # relative to s0
    lo: float = -BIG
    hi: float = BIG
    soft: float = 0.0  # how far lo/hi may be violated (at a cost)
    kind: str = "free"  # kind of the phase that ENDS here: free | commit
    n: int | None = None  # number of intervals of the phase that ends here
    v_end: float | None = None  # fixed terminal speed (only for the last boundary)


class EcoDrivePlanner:
    """Receding-horizon eco-driving speed planner (see module docstring)."""

    def __init__(self, limits=None, weights=None, config=None, speed_limit_fn=None):
        """
        Args:
            limits: :class:`VehicleLimits`.
            weights: :class:`Weights`.
            config: :class:`PlannerConfig`.
            speed_limit_fn: optional ``f(s_abs: ndarray) -> v_lim ndarray``, e.g.
                ``lambda s: route.speed_limit(s, v_max, a_lat)``. Defaults to
                the constant ``limits.v_max``.
        """
        self.limits = limits or VehicleLimits()
        self.weights = weights or Weights()
        self.cfg = config or PlannerConfig()
        self.speed_limit_fn = speed_limit_fn
        self._cache: OrderedDict = OrderedDict()
        self.last_plan: Plan | None = None

    # ------------------------------------------------------------ public API

    def plan(self, s0: float, v0: float, gates: list[Gate], s_goal: float | None = None,
             a_prev: float = 0.0) -> Plan:
        """Compute the best plan from the current state.

        Args:
            s0: current (unrolled) arc length.
            v0: current speed.
            gates: lights ahead (any order, passed ones are ignored).
            s_goal: arc length of the goal (stop there). ``None`` for laps:
                then the plan ends ``horizon`` metres ahead with free speed.
            a_prev: last commanded acceleration (for jerk continuity).
        """
        t_start = time.perf_counter()
        cfg, lim = self.cfg, self.limits
        v0 = float(np.clip(v0, 0.0, lim.v_max))

        # which part of the route does this plan cover?
        gates, reach = self._select_gates(s0, gates)
        if s_goal is not None and s_goal <= reach + 1e-6:
            s_end, terminal = s_goal, "goal"
        else:
            s_end, terminal = reach, "horizon"
        gates = [g for g in gates if g.s < s_end]

        if s_end - s0 < 0.01:
            plan = self._standstill(s0, "goal" if terminal == "goal" else "hold")
            plan.solve_time = time.perf_counter() - t_start
            return plan

        candidates = self._candidates(s0, v0, gates, s_end, terminal)
        if not any(c[2] != "hold" for c in candidates):
            # no way through every light: plan through the longest prefix of lights that
            # can be passed, and end before the next stop line (later replans decide there)
            candidates = self._prefix_candidates(s0, v0, gates) + [c for c in candidates if c[2] == "hold"]
        best: Plan | None = None
        self.n_solved = 0
        for bnds, crossings, status, lb in candidates[: cfg.max_candidates]:
            # holding is only a fallback: never prefer it over a feasible crossing
            if status == "hold" and best is not None:
                break
            # exact pruning: every term of J is >= 0, so J >= w_time * (earliest arrival);
            # a candidate whose bound cannot beat the best plan so far need not be solved
            if best is not None and lb >= best.cost:
                continue
            self.n_solved += 1
            plan = self._solve(s0, v0, a_prev, bnds, status)
            if plan is None:
                continue
            plan.crossings = crossings
            if best is None or plan.cost < best.cost - 1e-9:
                best = plan
        if best is None:
            first_stop = gates[0].s - cfg.stop_margin if gates else math.inf
            if v0 < 0.05 and s0 >= first_stop - 0.1:
                best = self._standstill(s0, "hold")  # waiting at a red light
            else:
                best = self._fallback(s0, v0, gates)
        best.solve_time = time.perf_counter() - t_start
        best.n_candidates = len(candidates)
        self.last_plan = best
        return best

    # -------------------------------------------------------- candidate sets

    # ------------------------------------------------------------ gate sets

    def _chained(self, a: Gate, b: Gate) -> bool:
        """True if the car cannot stop between lights a and b (b's stop line lies
        inside a's clearing zone): they must be planned, held for, and guarded together."""
        return b.s - self.cfg.stop_margin < a.s + self.cfg.clear_dist + 1e-6

    def _select_gates(self, s0, gates):
        """Lights this plan handles and where the plan ends.

        At most ``max_gates`` lights (more would make the NLP large and slow); the
        cut never splits a chain of close lights, and the plan ends before the
        stop line of the first light it does not handle, so it never drives through
        a light it has not checked.
        """
        cfg = self.cfg
        ahead = sorted((g for g in gates if s0 + 1e-3 < g.s + cfg.clear_dist), key=lambda g: g.s)
        k = min(sum(1 for g in ahead if g.s < s0 + cfg.horizon), cfg.max_gates)
        j = k
        if k < len(ahead):
            while j > 0 and self._chained(ahead[j - 1], ahead[j]):
                j -= 1
            if j == 0:  # the first chain alone is longer than the limit: take all of it
                j = max(k, 1)
                while j < len(ahead) and self._chained(ahead[j - 1], ahead[j]):
                    j += 1
        inc, rest = ahead[:j], ahead[j:]
        reach = s0 + cfg.horizon
        if inc:
            # always plan a bit past the clearing point of the last light considered
            reach = max(reach, inc[-1].s + cfg.clear_dist + 2.0)
        if rest:
            reach = min(reach, rest[0].s - cfg.stop_margin - 0.05)
            if inc:
                reach = max(reach, inc[-1].s + cfg.clear_dist + 0.02)
        return inc, reach

    def _min_time(self, v0: float, dist: float) -> float:
        """Lower bound on the time to drive `dist` (max accel, then cruise)."""
        lim = self.limits
        if dist <= 0:
            return 0.0
        d_acc = max(lim.v_max**2 - v0**2, 0.0) / (2 * lim.a_max)
        if dist <= d_acc:
            return (-v0 + math.sqrt(v0**2 + 2 * lim.a_max * dist)) / lim.a_max
        return (lim.v_max - v0) / lim.a_max + (dist - d_acc) / lim.v_max

    def _candidates(self, s0, v0, gates, s_end, terminal):
        cfg = self.cfg
        m = cfg.time_margin_min  # prefilter with the hard margin only
        # reachable windows per gate
        options = []
        for g in gates:
            t_min = self._min_time(v0, g.s - s0)
            t_clear_min = self._min_time(v0, g.s + cfg.clear_dist - s0)
            ws = [w for w in g.windows
                  if w.end - m >= t_clear_min and (w.end - w.start) > 2 * m + 1e-3]
            ws = [w for w in ws if w.end - m >= max(w.start + (m if w.start > 1e-6 else 0.0), t_min)]
            options.append(ws[: cfg.windows_per_light])

        out = []
        # full crossing candidates (one window per gate), generated depth first with the
        # chained reachability check (so infeasible prefixes are cut early, never 3^n),
        # then sorted by a lower bound on their cost, so the planner can prune exactly
        s_last = gates[-1].s if gates else s0
        for chosen, t_last in self._window_combos(s0, v0, gates, options, cfg.max_combos):
            lb = self.weights.time * (t_last + max(s_end - s_last, 0.0) / self.limits.v_max)
            bnds, crossings = self._boundaries(s0, gates, chosen, s_end, terminal)
            if bnds is not None:
                out.append((bnds, crossings, terminal, lb))
        out.sort(key=lambda c: c[3])

        # hold candidate: cross the first k gates, stop at the stop line of gate k.
        # Never stop between chained lights (that would be inside an intersection).
        for k in range(len(gates)):
            s_stop = gates[k].s - cfg.stop_margin
            if s_stop <= s0 + 0.02 or (k and self._chained(gates[k - 1], gates[k])):
                continue
            prefix = next(iter(self._window_combos(s0, v0, gates[:k], options[:k], 1)), None)
            if prefix is None:
                break
            bnds, crossings = self._boundaries(s0, gates[:k], prefix[0], s_stop, "hold")
            if bnds is not None:
                out.append((bnds, crossings, "hold", 0.0))
            break  # holding at the first possible stop line is enough
        return out

    def _prefix_candidates(self, s0, v0, gates):
        """Crossing candidates for the longest passable prefix of ``gates`` (cut between chains)."""
        cfg = self.cfg
        for k in range(len(gates) - 1, 0, -1):
            if self._chained(gates[k - 1], gates[k]):
                continue
            s_end = gates[k].s - cfg.stop_margin - 0.05
            if s_end <= gates[k - 1].s + cfg.clear_dist:
                continue
            cands = [c for c in self._candidates(s0, v0, gates[:k], s_end, "horizon") if c[2] != "hold"]
            if cands:
                return cands
        return []

    def _window_combos(self, s0, v0, gates, options, limit):
        """Yield up to ``limit`` reachable window choices ``(windows, t_last)``.

        Chains the lights: the car cannot reach light i before it could reach light
        i-1 plus the drive between them at full speed, nor before window i opens. If
        it cannot stop between two lights, it must also reach light i before
        window i-1 closes plus the gap at the minimum crossing speed. Only hard
        margins are used, so this never rejects a feasible candidate.
        ``t_last`` is the earliest time at the last light (for the cost bound).
        """
        m, lim, cfg = self.cfg.time_margin_min, self.limits, self.cfg
        count = 0

        def rec(i, chosen, t_prev, s_prev):
            nonlocal count
            if count >= limit:
                return
            if i == len(gates):
                count += 1
                yield list(chosen), t_prev
                return
            g = gates[i]
            for w in options[i]:
                t = max(self._min_time(v0, g.s - s0), t_prev + (g.s - s_prev) / lim.v_max)
                if w.start > 1e-6:
                    t = max(t, w.start + m)
                if t + cfg.clear_dist / lim.v_max > w.end - m + 1e-6:
                    continue
                if i and self._chained(gates[i - 1], g):
                    gap = max(g.s - gates[i - 1].s - cfg.clear_dist, 0.0)
                    if t > chosen[-1].end - m + gap / cfg.v_cross + 1e-6:
                        continue
                chosen.append(w)
                yield from rec(i + 1, chosen, t, g.s)
                chosen.pop()
                if count >= limit:
                    return

        yield from rec(0, [], 0.0, s0)

    def _boundaries(self, s0, gates, windows, s_end, terminal):
        """Phase end points for one window choice, sorted and merged.

        Each light gives three events (stop line, light, clearing point). With close
        lights these interleave, so the events are sorted by position and events at
        the same position are merged (their time bounds intersected). A phase is a
        *commit* phase (no stopping) if it lies inside any light's stop-to-clear zone.
        """
        cfg, m = self.cfg, self.cfg.time_margin
        soft = max(cfg.time_margin - cfg.time_margin_min, 0.0)
        events = []  # (s_rel, lo, hi)
        crossings = {}
        zones = []
        for g, w in zip(gates, windows):
            s_stop, s_exit = g.s - cfg.stop_margin, g.s + cfg.clear_dist
            zones.append((s_stop - s0, s_exit - s0))
            if s_stop > s0 + 0.02:
                events.append((s_stop - s0, -BIG, BIG))
            if g.s > s0 + 0.02:
                # no start margin for a green phase that is already on
                lo = w.start + m if w.start > 1e-6 else -BIG
                events.append((g.s - s0, lo, w.end - m))
            if s_exit > s0 + 0.02:
                events.append((s_exit - s0, -BIG, w.end - m))
            crossings[g.name] = w
        if s_end - s0 > 0.02:
            events.append((s_end - s0, -BIG, BIG))
        events = [e for e in events if e[0] <= s_end - s0 + 1e-9]
        events.sort(key=lambda e: e[0])

        merged: list[list[float]] = []
        for s, lo, hi in events:
            if merged and s - merged[-1][0] < 0.02:
                merged[-1][1], merged[-1][2] = max(merged[-1][1], lo), min(merged[-1][2], hi)
            else:
                merged.append([s, lo, hi])
        if not merged:
            return None, crossings

        bnds: list[_Boundary] = []
        s_prev = 0.0
        for s, lo, hi in merged:
            if lo > hi - 1e-3:
                return None, crossings  # two lights demand disjoint times at one point
            mid = 0.5 * (s_prev + s)
            commit = any(a < mid < b for a, b in zones)
            n = cfg.n_short if commit or s - s_prev < cfg.short_free else cfg.n_free
            finite = lo > -BIG / 2 or hi < BIG / 2
            bnds.append(_Boundary(s, lo=lo, hi=hi, soft=soft if finite else 0.0,
                                  kind="commit" if commit else "free", n=n))
            s_prev = s
        if terminal in ("goal", "hold"):
            bnds[-1].v_end = 0.0
        return bnds, crossings

    # ----------------------------------------------------------------- NLP

    def _structure(self, bnds):
        cfg = self.cfg
        ns = tuple(b.n if b.n is not None else (cfg.n_short if b.kind == "commit" else cfg.n_free)
                   for b in bnds)
        kinds = tuple(b.kind for b in bnds)
        return ns, kinds, bnds[-1].v_end is not None

    def _build(self, key):
        ns, kinds, fixed_vend = key
        cfg, lim = self.cfg, self.limits
        P = len(ns)
        opti = ca.Opti()

        # parameters
        p_v0 = opti.parameter()
        p_aprev = opti.parameter()
        p_s = opti.parameter(P)  # boundary positions (relative)
        p_lo = opti.parameter(P)
        p_hi = opti.parameter(P)
        p_soft = opti.parameter(P)
        p_vend = opti.parameter()
        p_wt = opti.parameter()  # time weight (0 for hold plans)
        n_lut = int(math.ceil(BIG_HORIZON / cfg.vlim_ds)) + 1
        p_vlim = opti.parameter(n_lut)
        lut = ca.interpolant("vlim", "linear", [np.linspace(0.0, BIG_HORIZON, n_lut).tolist()])

        S, V, A, T, slacks, margins, overs = [], [], [], [], [], [], []
        J_acc, J_jerk = 0, 0
        prev_a = p_aprev
        prev_dt = None
        tau = 0
        s_start, v_start = 0.0, p_v0
        constraints = []
        for p, (N, kind) in enumerate(zip(ns, kinds)):
            s = opti.variable(N + 1)
            v = opti.variable(N + 1)
            a = opti.variable(N)
            Tp = opti.variable()
            dt = Tp / N
            S.append(s), V.append(v), A.append(a), T.append(Tp)

            constraints += [s[0] == s_start, v[0] == v_start]
            constraints += [s[1:] == s[:-1] + dt * v[:-1] + 0.5 * dt**2 * a,
                            v[1:] == v[:-1] + dt * a]
            # curvature speed limit is soft: the measured speed may already exceed it
            # (e.g. entering a corner slightly fast), which must not make the NLP infeasible
            over = opti.variable(N + 1)
            overs.append(over)
            constraints += [opti.bounded(lim.a_min, a, lim.a_max), v >= 0, v <= lim.v_max,
                            over >= 0, v <= self._lut_expr(lut, s, p_vlim) + over]
            constraints += [s[N] == p_s[p], Tp >= 0.02, Tp <= 600]
            tau = tau + Tp
            dm = opti.variable()  # eaten-up part of the preferred time margin
            margins.append(dm)
            constraints += [tau >= p_lo[p] - dm, tau <= p_hi[p] + dm, dm >= 0, dm <= p_soft[p]]
            if kind == "commit":
                sl = opti.variable()
                slacks.append(sl)
                constraints += [sl >= 0, v >= cfg.v_cross - sl]

            J_acc += dt * ca.sumsqr(a)
            # jerk: (a_k - a_{k-1})^2 / dt, across phase boundaries as well
            da0 = a[0] - prev_a
            dt0 = dt if prev_dt is None else 0.5 * (dt + prev_dt)
            J_jerk += da0**2 / dt0 + ca.sumsqr(a[1:] - a[:-1]) / dt
            prev_a, prev_dt = a[N - 1], dt
            s_start, v_start = s[N], v[N]

        if fixed_vend:
            constraints.append(V[-1][-1] == p_vend)
        opti.subject_to(constraints)

        w = self.weights
        J = p_wt * tau + w.acc * J_acc + w.jerk * J_jerk
        if slacks:
            J += w.slack * ca.sumsqr(ca.vertcat(*slacks))
        J += w.margin * ca.sumsqr(ca.vertcat(*margins))
        J += w.slack * ca.sumsqr(ca.vertcat(*overs))
        opti.minimize(J)
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes",
                              "ipopt.max_iter": cfg.max_iter, "ipopt.tol": 1e-6,
                              "ipopt.acceptable_tol": 1e-4})
        return dict(opti=opti, S=S, V=V, A=A, T=T, slacks=slacks, J=J,
                    p=dict(v0=p_v0, aprev=p_aprev, s=p_s, lo=p_lo, hi=p_hi, soft=p_soft, vend=p_vend,
                           wt=p_wt, vlim=p_vlim), n_lut=n_lut)

    @staticmethod
    def _lut_expr(lut, s, p_vlim):
        return ca.vertcat(*[lut(s[i], p_vlim) for i in range(s.shape[0])])

    def _vlim_table(self, s0, n_lut):
        grid = np.linspace(0.0, BIG_HORIZON, n_lut)
        if self.speed_limit_fn is None:
            vals = np.full(n_lut, self.limits.v_max)
        else:
            vals = np.minimum(self.limits.v_max, np.asarray(self.speed_limit_fn(s0 + grid), float))
        # never let the limit drop below the crossing speed (keeps commit phases feasible)
        return np.maximum(vals, self.cfg.v_cross + 0.05)

    def _solve(self, s0, v0, a_prev, bnds, status) -> Plan | None:
        key = self._structure(bnds)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = self._build(key)
            while len(self._cache) > self.cfg.cache_size:
                self._cache.popitem(last=False)  # many lights: bound the memory
        P = self._cache[key]
        opti, p = P["opti"], P["p"]

        ss = np.array([b.s for b in bnds])
        if ss[-1] > BIG_HORIZON - 1:
            return None
        opti.set_value(p["v0"], v0)
        opti.set_value(p["aprev"], float(np.clip(a_prev, self.limits.a_min, self.limits.a_max)))
        opti.set_value(p["s"], ss)
        opti.set_value(p["lo"], np.array([b.lo for b in bnds]))
        opti.set_value(p["hi"], np.array([b.hi for b in bnds]))
        opti.set_value(p["soft"], np.array([b.soft for b in bnds]))
        opti.set_value(p["vend"], bnds[-1].v_end or 0.0)
        # a hold plan has no meaningful arrival time; a small weight keeps it from crawling
        opti.set_value(p["wt"], (0.1 if status == "hold" else 1.0) * self.weights.time)
        opti.set_value(p["vlim"], self._vlim_table(s0, P["n_lut"]))

        self._initial_guess(P, v0, bnds)
        try:
            sol = opti.solve()
        except RuntimeError:
            return None

        t_all, s_all, v_all, a_all = [0.0], [0.0], [v0], []
        t_off = 0.0
        for S, V, A, T in zip(P["S"], P["V"], P["A"], P["T"]):
            Tv = float(sol.value(T))
            sv, vv, av = (np.atleast_1d(sol.value(x)) for x in (S, V, A))
            N = len(av)
            t_all += list(t_off + Tv * np.arange(1, N + 1) / N)
            s_all += list(sv[1:])
            v_all += list(vv[1:])
            a_all += list(av)
            t_off += Tv
        plan = Plan(t=np.array(t_all), s=s0 + np.array(s_all), v=np.maximum(np.array(v_all), 0.0),
                    a=np.array(a_all), cost=float(sol.value(P["J"])), status=status, s0=s0)
        return plan

    def _initial_guess(self, P, v0, bnds):
        """Simple feasible-ish guess: cruise, respecting the window lower bounds."""
        opti, lim = P["opti"], self.limits
        v_nom = max(0.6 * lim.v_max, 0.3)
        tau_prev, s_prev = 0.0, 0.0
        for b, S, V, A, T in zip(bnds, P["S"], P["V"], P["A"], P["T"]):
            d = b.s - s_prev
            tau = tau_prev + d / v_nom
            tau = min(max(tau, b.lo + 1e-2), b.hi - 1e-2) if b.hi < BIG else max(tau, b.lo + 1e-2)
            Tp = max(tau - tau_prev, 0.05)
            N = A.shape[0]
            opti.set_initial(T, Tp)
            opti.set_initial(S, np.linspace(s_prev, b.s, N + 1))
            opti.set_initial(V, np.full(N + 1, max(d / Tp, 0.0)))
            opti.set_initial(A, np.zeros(N))
            tau_prev, s_prev = tau_prev + Tp, b.s

    # -------------------------------------------------------------- fallbacks

    def _standstill(self, s0, status):
        return Plan(t=np.array([0.0, 1.0]), s=np.array([s0, s0]), v=np.zeros(2), a=np.zeros(1),
                    cost=0.0, status=status, s0=s0)

    def _fallback(self, s0, v0, gates):
        """Analytic comfortable stop – used only if every NLP failed."""
        lim = self.limits
        b = -lim.a_min
        if gates:
            d = max(gates[0].s - self.cfg.stop_margin - s0, 1e-3)
            b = max(b, v0**2 / (2 * d))
        T = v0 / b if v0 > 1e-6 else 1.0
        t = np.linspace(0.0, T, 11)
        a = np.full(10, -b if v0 > 1e-6 else 0.0)
        v = np.maximum(v0 - b * t, 0.0)
        s = s0 + v0 * t - 0.5 * b * t**2
        return Plan(t=t, s=s, v=v, a=a, cost=math.inf, status="failed", s0=s0)


# maximum length of the speed-limit lookup table (relative to s0) [m]
BIG_HORIZON = 80.0
