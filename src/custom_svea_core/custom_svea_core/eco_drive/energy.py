"""Energy metrics for speed profiles.

Two measures are reported everywhere:

* ``accel_cost``: the integral of a² dt. This is the energy proxy the tutorial suggests,
  and it is also the term the optimizer minimizes.
* ``battery_energy``: a simple physical estimate of the electrical energy drawn from
  the battery by a 1:10 car. It counts tractive power only, with no regeneration
  (the SVEA ESC does not recuperate):

      P = m a v  +  m g c_rr v  +  ½ ρ c_d A v³        (wheel power)
      E = ∫ max(P, 0) / η  dt

  The parameters are rough numbers for a SVEA (about 6 kg). They are good enough
  to compare driving styles, not to predict absolute battery drain.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CarParams", "accel_cost", "battery_energy"]


class CarParams:
    mass = 6.0  # [kg]
    c_rr = 0.03  # rolling resistance coefficient [-]
    cd_a = 0.03  # drag area [m^2]
    rho = 1.2  # air density [kg/m^3]
    eta = 0.6  # battery -> wheel efficiency [-]
    g = 9.81


def accel_cost(t, a) -> float:
    """Integral of a^2 dt for a piecewise constant acceleration a on the grid t."""
    t, a = np.asarray(t, float), np.asarray(a, float)
    dt = np.diff(t)[: len(a)]
    return float(np.sum(a[: len(dt)] ** 2 * dt))


def battery_energy(t, v, a, car: CarParams = CarParams) -> float:
    """Battery energy [J] for samples (t, v) and piecewise constant a."""
    t, v, a = np.asarray(t, float), np.asarray(v, float), np.asarray(a, float)
    n = min(len(t) - 1, len(a))
    dt = np.diff(t)[:n]
    vm = 0.5 * (v[:n] + v[1 : n + 1])  # mean speed on each interval
    p = car.mass * a[:n] * vm + car.mass * car.g * car.c_rr * vm + 0.5 * car.rho * car.cd_a * vm**3
    return float(np.sum(np.maximum(p, 0.0) * dt) / car.eta)
