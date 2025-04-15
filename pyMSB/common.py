import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import cumulative_trapezoid

from pyMSB.constants import MATERIAL_CONSTANTS
from pyMSB.models import (
    DoubletComputable,
    SextetComputable,
    SingletComputable,
    SpectroscopeComputable,
    SpectroscopeGeometry,
    SpectroscopeT,
    SpectrumComputable,
    SpectrumT,
)
from pyMSB.statistics import get_spectrum_background


def line(channels: ArrayLike, amp: int, pos: int, line_width: float) -> np.ndarray:
    """Return Lorentz function evaluated for given channels."""
    return abs(amp) * np.power(line_width / 2, 2) / (np.power(channels - pos, 2) + np.power(line_width / 2, 2))


def singlet_func_free(channels: ArrayLike, amp: int, pos: int, line_width1: float):
    """
    Return singlet function with free line position.

    :param channels: Array of channels to evaluate the function at.
    :param amp: Amplitude of lines 1 and 6.
    :param pos: Absolute position of the line.
    :param line_width1: Line width (fwhm - full width at half maximum) of the line.
    """
    return line(channels, amp, pos, line_width1)


def singlet_func(channels: ArrayLike, singlet: SingletComputable, isomer_shift_ref: float):
    """
    Return singlet function with bounded line position.

    :param channels: Array of channels to evaluate the function at.
    :param singlet: SingletGeneric object containing all parameters needed for a bounded singlet definition.
    :param isomer_shift_ref: Position of energetic zero in channels (Absolute isomer shift of calibration sextet).
    """
    return singlet_func_free(
        channels,
        amp=singlet.amplitude,
        pos=isomer_shift_ref + singlet.isomer_shift,
        line_width1=singlet.line_width1,
    )


def doublet_func_free(
    channels: ArrayLike,
    amp: int,
    pos1: int,
    pos2: int,
    line_width1: float,
    line_width2: float,
):
    """
    Return doublet function with free line positions.

    :param channels: Array of channels to evaluate the function at.
    :param amp: Amplitude of lines 1 and 6.
    :param pos1: Absolute position of line 1 (similarly for pos2).
    :param pos2:
    :param line_width1: Line width (fwhm - full width at half maximum) of line 1 (similarly for line_width2).
    :param line_width2:
    """
    return line(channels, amp, pos1, line_width1) + line(channels, amp, pos2, line_width2)


def doublet_func(channels: ArrayLike, doublet: DoubletComputable, isomer_shift_ref: float):
    """
    Return doublet function with bounded line positions.

    :param channels: Array of channels to evaluate the function at.
    :param doublet: DoubletGeneric object containing all parameters needed for a bounded doublet definition.
    :param isomer_shift_ref: Position of energetic zero in channels (Absolute isomer shift of calibration sextet).
    """
    return doublet_func_free(
        channels,
        amp=doublet.amplitude,
        pos1=isomer_shift_ref + doublet.isomer_shift - 0.5 * doublet.quadrupole_split,
        pos2=isomer_shift_ref + doublet.isomer_shift + 0.5 * doublet.quadrupole_split,
        line_width1=doublet.line_width1,
        line_width2=doublet.line_width2,
    )


def sextet_func_free(
    channels,
    ratio13,
    ratio23,
    amp,
    pos1,
    pos2,
    pos3,
    pos4,
    pos5,
    pos6,
    line_width1,
    line_width2,
    line_width3,
    line_width4,
    line_width5,
    line_width6,
):
    """
    Return sextet function with free line positions.

    :param channels: Array of channels to evaluate the function at.
    :param ratio13: Lines 1 and 3 amplitude ratio.
    :param ratio23: Lines 2 and 3 amplitude ratio.
    :param amp: Amplitude of lines 1 and 6.
    :param pos1: Absolute position of line 1 (similarly for pos2-6).
    :param line_width1: Line width (fwhm - full width at half maximum) of line 1 (similarly for line_width2-6).
    """
    return (
        line(channels, amp, pos1, line_width1)
        + line(channels, amp * ratio23 / ratio13, pos2, line_width2)
        + line(channels, amp / ratio13, pos3, line_width3)
        + line(channels, amp / ratio13, pos4, line_width4)
        + line(channels, amp * ratio23 / ratio13, pos5, line_width5)
        + line(channels, amp, pos6, line_width6)
    )


def sextet_func(
    channels: ArrayLike,
    sextet: SextetComputable,
    isomer_shift_ref: float,
    zeeman_diff_ground: float,
    zeeman_diff_excit: float,
):
    """
    Return sextet function with bounded line positions.

    :param channels: Array of channels to evaluate the function at.
    :param sextet: SextetGeneric object containing all parameters needed for a bounded sextet definition.
    :param isomer_shift_ref: Position of energetic zero in channels (Absolute isomer shift of calibration sextet).
    :param zeeman_diff_ground: Material constant.
    :param zeeman_diff_excit: Material constant.
    """
    return sextet_func_free(
        channels,
        ratio13=sextet.ratio13,
        ratio23=sextet.ratio23,
        amp=sextet.amplitude,
        pos1=(
            isomer_shift_ref
            + sextet.isomer_shift
            - 0.5 * (zeeman_diff_ground + 3 * zeeman_diff_excit) * sextet.magnetic_field
            + 0.5 * sextet.quadrupole_split
        ),
        pos2=(
            isomer_shift_ref
            + sextet.isomer_shift
            - 0.5 * (zeeman_diff_ground + 1 * zeeman_diff_excit) * sextet.magnetic_field
            - 0.5 * sextet.quadrupole_split
        ),
        pos3=(
            isomer_shift_ref
            + sextet.isomer_shift
            - 0.5 * (zeeman_diff_ground - 1 * zeeman_diff_excit) * sextet.magnetic_field
            - 0.5 * sextet.quadrupole_split
        ),
        pos4=(
            isomer_shift_ref
            + sextet.isomer_shift
            + 0.5 * (zeeman_diff_ground - 1 * zeeman_diff_excit) * sextet.magnetic_field
            - 0.5 * sextet.quadrupole_split
        ),
        pos5=(
            isomer_shift_ref
            + sextet.isomer_shift
            + 0.5 * (zeeman_diff_ground + 1 * zeeman_diff_excit) * sextet.magnetic_field
            - 0.5 * sextet.quadrupole_split
        ),
        pos6=(
            isomer_shift_ref
            + sextet.isomer_shift
            + 0.5 * (zeeman_diff_ground + 3 * zeeman_diff_excit) * sextet.magnetic_field
            + 0.5 * sextet.quadrupole_split
        ),
        line_width1=sextet.line_width1,
        line_width2=sextet.line_width2,
        line_width3=sextet.line_width3,
        line_width4=sextet.line_width4,
        line_width5=sextet.line_width5,
        line_width6=sextet.line_width6,
    )


def spectrum_func(
    channels: ArrayLike,
    spectrum: SpectrumT,
    spectroscope: SpectroscopeT,
    geometry: SpectroscopeGeometry = SpectroscopeGeometry.TRANSMISSION,
) -> np.ndarray[float]:
    """Return spectrum function evaluated at given channels."""
    geometry_coeff = -1.0 if geometry == SpectroscopeGeometry.TRANSMISSION else 1.0

    counts = np.ones(shape=(len(channels),), dtype=float) * spectrum.background
    for singlet in spectrum.singlets:
        counts += geometry_coeff * singlet_func(channels, singlet, isomer_shift_ref=spectroscope.isomer_shift_ref)
    for doublet in spectrum.doublets:
        counts += geometry_coeff * doublet_func(channels, doublet, isomer_shift_ref=spectroscope.isomer_shift_ref)
    for sextet in spectrum.sextets:
        counts += geometry_coeff * sextet_func(
            channels,
            sextet,
            isomer_shift_ref=spectroscope.isomer_shift_ref,
            zeeman_diff_excit=MATERIAL_CONSTANTS["Fe"].zeeman_diff_excited * spectroscope.scale,
            zeeman_diff_ground=MATERIAL_CONSTANTS["Fe"].zeeman_diff_ground * spectroscope.scale,
        )
    return counts


def generate_spectrum(
    channels: ArrayLike,
    spectrum: SpectrumComputable,
    spectroscope: SpectroscopeComputable,
    geometry: SpectroscopeGeometry = SpectroscopeGeometry.TRANSMISSION,
    seed: int = None,
) -> ArrayLike:
    """Return a spectrum function with poisson noise."""
    rng = np.random.default_rng(seed=seed)
    return rng.poisson(
        spectrum_func(
            channels=channels,
            spectrum=spectrum,
            spectroscope=spectroscope,
            geometry=geometry,
        )
    )


def find_discrete_fold_point(counts: np.ndarray[int] | list[int]) -> int:
    """
    Return discrete fold point estimate (partially based on MossWinn methodology).

    Estimated as fold point with the minimum sum of squared differences
    between left and right parts of the unfolded spectrum split at said point.
    Note: Spectrum is shifted to have zero background, so it can be padded with zeros.

    :param counts: Array of counts of unfolded spectrum.
    :returns: Number of channels to be included in the left half
    """
    n = len(counts)
    counts = counts.astype(int)  # Ensure that counts are integers not unsigned integers before shifting
    counts_shifted = counts - get_spectrum_background(counts)

    diff_sums = np.empty(n, dtype=np.uint64)
    left = np.zeros(n, dtype=int)
    right = np.zeros(n, dtype=int)
    for i in range(n):
        left[:i], left[i:] = counts_shifted[:i][::-1], 0
        right[: n - i], right[n - i :] = counts_shifted[i:], 0
        diff_sums[i - 1] = np.sum(np.abs(left - right))

    optimal_fold_point = np.argmin(diff_sums) + 1
    return optimal_fold_point


def find_adaptive_points(counts: ArrayLike, n: int = 256, channels: ArrayLike = None) -> ArrayLike:
    """
    Return adaptively spaced points for a given spectrum.

    Reduce the number of points in the spectrum while preserving the shape.
    Based on equidistant division of the cumulative curvature (abs second derivative).

    Intended use:
        - Get spectrum counts from spectrum function and use it as counts parameter.
        - Use adaptive points as channels parameter in spectrum function to get adaptively spaced spectrum.

        >>> channels = np.arange(spectrum.length)
        >>> counts = spectrum_func(channels, spectrum, spectroscope)
        >>> adaptive_channels = find_adaptive_points(counts)
        >>> adaptive_counts = spectrum_func(adaptive_channels, spectrum, spectroscope)

    How it works:
        - calculate abs second derivative - the measure of curvature (= non-linearity)
        - take log - weaken the effect of amplitude (smaller peak still have fewer points but the effect is weaker)
        - integrate and normalize [0, 1] - cumulative curvature
        - combine with equidistant division - ensuring at least some points at close to linear regions
        - inverse function - interpolate to get adaptively spaced points

    :param counts: Array of spectrum function values (counts)
    :param n: Number of points to be spaced adaptively. Defaults to 256
    :param channels: Array of channels at which the spectrum function is evaluated. Defaults to np.arange(len(counts))
    :returns: Array of adaptively spaced channels
    """
    if channels is None:
        channels = np.arange(len(counts))
    second_derivative = abs(np.gradient(np.gradient(counts)))
    second_derivative = np.log(second_derivative + 1)
    cum_curve = cumulative_trapezoid(second_derivative, channels, initial=0)
    cum_curve /= cum_curve[-1] + 1e-5  # NOTE: Added small number to avoid division by zero!
    equidistant = np.linspace(0, 1, len(channels))
    alpha = 0.9
    combined = (1 - alpha) * equidistant + alpha * cum_curve
    return np.interp(np.linspace(0, 1, n), combined, channels), second_derivative
