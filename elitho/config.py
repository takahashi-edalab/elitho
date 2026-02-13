import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import ClassVar, Literal
from functools import cached_property


# # Default parameters
# MX = 4  # X magnification
# # Optical parameters
# is_high_na = True
# if is_high_na:
#     NA = 0.55
#     theta0 = -5.3  # incidence angle(degree) for high-NA
#     MY = 8  # Y magnification
#     NDIVX = 512  # X pitch (nm)
#     NDIVY = 2 * NDIVX  # Y pitch (nm)

# else:
#     NA = 0.33
#     theta0 = -6.0  # incidence angle (degree)
#     MY = 4  # Y magnification
#     NDIVX = 1024  # X pitch (nm)
#     NDIVY = 1024  # Y pitch (nm)


class PolarizationDirection(Enum):
    X = auto()
    Y = auto()


class IlluminationType(Enum):
    CIRCULAR = auto()
    ANNULAR = auto()
    DIPOLE_X = auto()
    DIPOLE_Y = auto()


class Illumination(ABC):
    pass
    # @abstractmethod
    # def description(self) -> str: ...


@dataclass
class CircularIllumination(Illumination):
    type: ClassVar[IlluminationType] = IlluminationType.CIRCULAR
    outer_sigma: float = 0.9


@dataclass
class AnnularIllumination(Illumination):
    type: ClassVar[IlluminationType] = IlluminationType.ANNULAR
    outer_sigma: float = 0.9
    inner_sigma: float = 0.55


@dataclass
class DipoleIllumination(Illumination):
    type: Literal[IlluminationType.DIPOLE_X, IlluminationType.DIPOLE_Y]
    outer_sigma: float = 0.9
    inner_sigma: float = 0.55
    open_angle: float = 90.0

    def __post_init__(self):
        if self.type not in (IlluminationType.DIPOLE_X, IlluminationType.DIPOLE_Y):
            raise ValueError("type must be 'DIPOLE_X' or 'DIPOLE_Y'")


@dataclass
class AbsorberLayers:
    thicknesses: list[float] = field(default_factory=lambda: [60.0])  # nm
    complex_refractive_indices: list[complex] = field(
        default_factory=lambda: [0.9567 + 0.0343j]
    )

    @cached_property
    def dielectric_constants(self) -> list[complex]:
        return [cri**2 for cri in self.complex_refractive_indices]

    @cached_property
    def total_thickness(self) -> float:
        return sum(self.thicknesses)

    @cached_property
    def z_ref_from_abs_top(self) -> float:
        return (
            self.total_thickness + 42.0
        )  # reflection point inside ML from the top of the absorber


@dataclass(frozen=True)
class SimulationConfig:
    wavelength: float = 13.5  # nm
    NA: float = 0.33
    is_high_na: bool = False
    illumination: Illumination = field(default_factory=lambda: CircularIllumination())
    absorber_layers: AbsorberLayers = field(default_factory=lambda: AbsorberLayers())
    mask_width: int = 1024  # nm
    mask_height: int = 1024  # nm
    mask_refinement_factor_x: int = 1
    mask_refinement_factor_y: int = 1
    magnification_x: int = 4
    magnification_y: int = 4
    mesh: float = 0.5
    incidence_angle: float = -6.0  # degree
    azimuthal_angle: float = 0.0  # degree
    central_obscuration: float = 0.2
    defocus_min: float = 0.0  # nm
    defocus_max: float = None
    defocus_step: float = None
    # TODO:
    # cutoff_factor: float = 6.0

    def __str__(self) -> str:
        lines = ["=" * 60, "Simulation Configuration", "=" * 60]

        # Optical parameters
        lines.append("\n[Optical Parameters]")
        lines.append(f"  Wavelength: {self.wavelength} nm")
        lines.append(f"  NA: {self.NA}")
        lines.append(f"  High NA mode: {self.is_high_na}")
        lines.append(f"  Central obscuration: {self.central_obscuration}")

        # Illumination
        lines.append("\n[Illumination]")
        ill_type = type(self.illumination).__name__.replace("Illumination", "")
        lines.append(f"  Type: {ill_type}")
        if hasattr(self.illumination, 'outer_sigma'):
            lines.append(f"  Outer sigma: {self.illumination.outer_sigma}")
        if hasattr(self.illumination, 'inner_sigma'):
            lines.append(f"  Inner sigma: {self.illumination.inner_sigma}")
        if hasattr(self.illumination, 'open_angle'):
            lines.append(f"  Open angle: {self.illumination.open_angle}°")

        # Absorber layers
        lines.append("\n[Absorber Layers]")
        lines.append(f"  Number of layers: {len(self.absorber_layers.thicknesses)}")
        lines.append(f"  Total thickness: {self.absorber_layers.total_thickness:.2f} nm")
        for i, (t, n) in enumerate(zip(self.absorber_layers.thicknesses,
                                        self.absorber_layers.complex_refractive_indices)):
            label = "(top)" if i == 0 else ""
            lines.append(f"    Layer {i+1} {label}: n={n.real:.4f}+{n.imag:.4f}j, t={t:.1f} nm")

        # Mask parameters
        lines.append("\n[Mask Parameters]")
        lines.append(f"  Mask size: {self.mask_width} × {self.mask_height} nm²")
        lines.append(f"  Refinement factor: {self.mask_refinement_factor_x} × {self.mask_refinement_factor_y}")
        lines.append(f"  Magnification: {self.magnification_x} × {self.magnification_y}")
        lines.append(f"  Exposure field: {self.exposure_field_width} × {self.exposure_field_height} nm²")

        # Angle parameters
        lines.append("\n[Angle Parameters]")
        lines.append(f"  Incidence angle: {self.incidence_angle}°")
        lines.append(f"  Azimuthal angle: {self.azimuthal_angle}°")

        # Defocus parameters
        lines.append("\n[Defocus Parameters]")
        if self.defocus_max is None or self.defocus_step is None:
            lines.append(f"  Single defocus: {self.defocus_min} nm")
        else:
            lines.append(f"  Range: {self.defocus_min} to {self.defocus_max} nm")
            lines.append(f"  Step: {self.defocus_step} nm")
            lines.append(f"  Number of points: {len(self.defocus_list)}")

        # Calculation parameters
        lines.append("\n[Calculation Parameters]")
        lines.append(f"  Mesh: {self.mesh}°")
        lines.append(f"  k (wavenumber): {self.k:.6f} nm⁻¹")
        lines.append(f"  kX: {self.kX:.6f} nm⁻¹")
        lines.append(f"  kY: {self.kY:.6f} nm⁻¹")
        lines.append(f"  ndiv: {self.ndivX} × {self.ndivY}")
        lines.append(f"  lsmax: {self.lsmaxX} × {self.lsmaxY}")
        lines.append(f"  lpmax: {self.lpmaxX} × {self.lpmaxY}")

        lines.append("=" * 60)
        return "\n".join(lines)

    @property
    def defocus_list(self) -> list:
        if self.defocus_max is None or self.defocus_step is None:
            return [self.defocus_min]

        assert self.defocus_step > 0, "Invalid defocus step."
        assert self.defocus_max > self.defocus_min, "Invalid defocus range."
        return list(
            np.arange(
                self.defocus_min,
                self.defocus_max + self.defocus_step,
                self.defocus_step,
            )
        )

    @cached_property
    def exposure_field_width(self):
        return self.mask_width // self.magnification_x

    @cached_property
    def exposure_field_height(self):
        return self.mask_height // self.magnification_y

    @cached_property
    def k(self):
        return 2.0 * np.pi / self.wavelength

    @cached_property
    def kX(self):
        return self.k * self.NA / self.magnification_x

    @cached_property
    def kY(self):
        return self.k * self.NA / self.magnification_y

    @cached_property
    def phi0(self):
        return 90.0 - self.azimuthal_angle

    @cached_property
    def kx0(self):
        return (
            self.k
            * np.sin(np.deg2rad(self.incidence_angle))
            * np.cos(np.deg2rad(self.phi0))
        )

    @cached_property
    def ky0(self):
        return (
            self.k
            * np.sin(np.deg2rad(self.incidence_angle))
            * np.sin(np.deg2rad(self.phi0))
        )

    @cached_property
    def ndivX(self):
        return max(
            1, int(180.0 / np.pi * self.wavelength / self.mask_width / self.mesh)
        )

    @cached_property
    def ndivY(self):
        return max(
            1, int(180.0 / np.pi * self.wavelength / self.mask_height / self.mesh)
        )

    @cached_property
    def lsmaxX(self):
        return int(
            self.NA * self.mask_width / self.magnification_x / self.wavelength + 1
        )

    @cached_property
    def lsmaxY(self):
        return int(
            self.NA * self.mask_height / self.magnification_y / self.wavelength + 1
        )

    @cached_property
    def lpmaxX(self):
        return int(
            self.NA * self.mask_width / self.magnification_x * 2 / self.wavelength
            + 0.0001
        )

    @cached_property
    def lpmaxY(self):
        return int(
            self.NA * self.mask_height / self.magnification_y * 2 / self.wavelength
            + 0.0001
        )

    @cached_property
    def nsourceX(self):
        return 2 * self.lsmaxX + 1

    @cached_property
    def nsourceY(self):
        return 2 * self.lsmaxY + 1

    @cached_property
    def noutX(self):
        return 2 * self.lpmaxX + 1

    @cached_property
    def noutY(self):
        return 2 * self.lpmaxY + 1

    @cached_property
    def nsourceXL(self):
        return 2 * self.lsmaxX + 10

    @cached_property
    def nsourceYL(self):
        return 2 * self.lsmaxY + 10

    @cached_property
    def noutXL(self):
        return 2 * self.lpmaxX + 10

    @cached_property
    def noutYL(self):
        return 2 * self.lpmaxY + 10

    @cached_property
    def dkx(self):
        return 2.0 * np.pi / self.mask_width

    @cached_property
    def dky(self):
        return 2.0 * np.pi / self.mask_height


# Mask properties
NML = 40  # number of the multilayer pairs

# complex refractive index
n_mo = 0.9238 + 0.006435j  # Mo layer
n_si = 0.999 + 0.001826j  # Si layer
n_ru = 0.8863 + 0.01706j  # Ru layer
n_mo_si2 = 0.9693 + 0.004333j  # Mo/Si2 mixing layer
n_ru_si = 0.9099 + 0.01547j  # Ru/Si mixing layer
n_si_o2 = 0.978 + 0.01083j  # SiO2 layer

# thickness [nm]
thickness_mo = 2.052
thickness_si = 2.283
thickness_ru = 2.5
thickness_mo_si = 1.661
thickness_si_mo = 1.045
thickness_si_ru = 0.8


# complex permittivity
epsilon_mo = n_mo**2
epsilon_si = n_si**2
epsilon_ru = n_ru**2
epsilon_mo_si2 = n_mo_si2**2
epsilon_ru_si = n_ru_si**2
epsilon_si_o2 = n_si_o2**2
