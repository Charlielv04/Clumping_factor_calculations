from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def age_universe(aexp: float | np.ndarray, hubble0: float, omega_m: float, omega_l: float) -> float | np.ndarray:
    def h0t(a: float, omega_matter: float, omega_lambda: float) -> float:
        return 1.0 / np.sqrt(omega_matter / a + omega_lambda * a**2)

    hubbletime = 1.0 / hubble0 * 3.0856e19 / 3.1536e7 / 1.0e2
    if np.size(aexp) == 1:
        h0time = quad(h0t, 0, float(aexp), args=(omega_m, omega_l))[0]
    else:
        h0time = np.array([quad(h0t, 0, float(a), args=(omega_m, omega_l))[0] for a in np.asarray(aexp)])
    return h0time * hubbletime


def hubble_param(aexp: float, hubble0: float, omega_m: float, omega_l: float) -> float:
    return float(hubble0) * np.sqrt(float(omega_m) / float(aexp) ** 3 + float(omega_l))


def e_z(redshift: float, omega_m: float, omega_l: float) -> float:
    return float(np.sqrt(float(omega_m) * (1.0 + float(redshift)) ** 3 + float(omega_l)))


def dist_to_vel(distance_mpc_h: float | np.ndarray, redshift: float, omega_m: float, omega_l: float) -> float | np.ndarray:
    return np.asarray(distance_mpc_h) / (1.0 + redshift) * 100.0 * e_z(redshift, omega_m, omega_l)


def vel_to_dist(velocity_kms: float | np.ndarray, redshift: float, omega_m: float, omega_l: float) -> float | np.ndarray:
    return np.asarray(velocity_kms) * (1.0 + redshift) / 100.0 / e_z(redshift, omega_m, omega_l)


def length_kms_from_cmpc_h(
    length_cmpc_h: float,
    redshift: float,
    omega_m: float = 0.3,
    omega_l: float = 0.7,
    h: float = 0.7,
) -> float:
    return float(length_cmpc_h) / (1.0 + float(redshift)) * 100.0 * float(h) * e_z(redshift, omega_m, omega_l)
