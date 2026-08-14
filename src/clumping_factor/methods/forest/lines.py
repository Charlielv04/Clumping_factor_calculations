from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .constants import SPEED_OF_LIGHT_CM_S


@dataclass(frozen=True)
class LineParameters:
    wavelength_cm: float
    frequency_hz: float
    oscillator_strength: float
    damping_constant_s: float
    lifetime_s: float
    atomic_mass: float

    def legacy_dict(self) -> dict[str, float]:
        return {
            "lambda": self.wavelength_cm,
            "nu": self.frequency_hz,
            "f": self.oscillator_strength,
            "Gamma": self.damping_constant_s,
            "tau": self.lifetime_s,
            "m_atom": self.atomic_mass,
        }


ATOMIC_MASS = {
    "H": 1.00794,
    "He": 4.002602,
    "Li": 6.941,
    "Be": 9.012182,
    "B": 10.811,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.9984032,
    "Ne": 20.1797,
    "Na": 22.989770,
    "Mg": 24.3050,
    "Al": 26.981538,
    "Si": 28.0855,
    "P": 30.973761,
    "S": 32.065,
    "Cl": 35.453,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955910,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938049,
    "Fe": 55.845,
    "Co": 58.933200,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.409,
}


def default_line_list_path() -> Path:
    return Path(resources.files(__package__) / "data" / "line_list.txt")


def _line_name_to_mass(name: str) -> float:
    if name[:3] == "Ly " or (len(name) > 2 and name[2:4] == "H "):
        return ATOMIC_MASS["H"]
    symbol = name[:2].strip()
    if symbol not in ATOMIC_MASS:
        symbol = name[:1].strip()
    return ATOMIC_MASS[symbol]


def read_line_parameters(line_list_file: str | Path | None = None) -> dict[str, LineParameters]:
    path = Path(line_list_file) if line_list_file is not None else default_line_list_path()
    line_parameters: dict[str, LineParameters] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.startswith("#"):
                continue
            alt_name = raw_line[-12:-1].strip() if raw_line.endswith("\n") else raw_line[-11:].strip()
            if not alt_name:
                continue
            values = raw_line[10:-12].strip().split()
            if len(values) < 3:
                raise ValueError(f"Cannot parse line-list row: {raw_line.rstrip()}")
            wavelength_angstrom = float(values[0])
            gamma = float(values[1])
            oscillator_strength = float(values[2])
            wavelength_cm = wavelength_angstrom * 1.0e-8
            line_parameters[alt_name] = LineParameters(
                wavelength_cm=wavelength_cm,
                frequency_hz=SPEED_OF_LIGHT_CM_S / wavelength_cm,
                oscillator_strength=oscillator_strength,
                damping_constant_s=gamma,
                lifetime_s=1.0 / gamma,
                atomic_mass=_line_name_to_mass(alt_name),
            )
    return line_parameters


def read_legacy_line_parameters(line_list_file: str | Path | None = None) -> dict[str, dict[str, float]]:
    return {name: params.legacy_dict() for name, params in read_line_parameters(line_list_file).items()}
