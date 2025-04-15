import numpy as np
from scipy.signal import find_peaks

from pyMSB.common import (
    doublet_func,
    doublet_func_free,
    find_adaptive_points,
    find_discrete_fold_point,
    generate_spectrum,
    line,
    sextet_func,
    sextet_func_free,
    singlet_func,
    singlet_func_free,
    spectrum_func,
)
from pyMSB.constants import MATERIAL_CONSTANTS
from pyMSB.models import (
    SpectroscopeGeometry,
    SpectrumComputable,
)
from pyMSB.statistics import get_background_std, get_spectrum_background, get_spectrum_span
from pyMSB.tests.conftest import CHANNELS, DOUBLET, SEXTET, SINGLET, SPECTROSCOPE, SPECTRUM


def test_line():
    """Test if line function returns correct values."""
    amplitude = 1000
    position = 512
    line_width = 10
    counts = line(channels=CHANNELS, amp=amplitude, pos=position, line_width=line_width)
    assert len(counts) == len(CHANNELS)
    assert np.max(counts) == amplitude  # only true if pos is int
    assert np.argmax(counts) == position  # only true if pos is int
    # test line_width (=full width at half maximum)
    half_max_indices = np.where(counts > amplitude / 2)[0]
    half_max_lw = half_max_indices.max() - half_max_indices.min()
    assert abs(half_max_lw - line_width) < 3  # difference between desired and observed LWs must be less than 3


def test_empty_spectrum():
    """Test if spectrum_func returns correct values for an empty spectrum."""
    background = 10**5
    empty_spectrum = spectrum_func(
        channels=CHANNELS,
        spectrum=SpectrumComputable(background=background),
        spectroscope=SPECTROSCOPE,
        geometry=SpectroscopeGeometry.TRANSMISSION,
    )
    assert (empty_spectrum == background).all()


def test_singlet_free():
    """Test if singlet_func_free returns correct values."""
    counts = singlet_func_free(channels=CHANNELS, amp=1000, pos=512, line_width1=10)
    assert len(counts) == len(CHANNELS)
    assert 1000 * 0.95 <= np.max(counts) <= 1000 * 1.05
    peaks = find_peaks(counts)[0]
    assert len(peaks) == 1
    assert 511 <= peaks[0] <= 513


def test_singlet_func():
    """Test if singlet_func returns correct values."""
    counts = singlet_func(channels=CHANNELS, singlet=SINGLET, isomer_shift_ref=SPECTROSCOPE.isomer_shift_ref)
    assert len(counts) == len(CHANNELS)
    assert SINGLET.amplitude * 0.95 <= np.max(counts) <= SINGLET.amplitude * 1.05
    peaks = find_peaks(counts)[0]
    assert len(peaks) == 1
    pos = SPECTROSCOPE.isomer_shift_ref + SINGLET.isomer_shift
    assert pos - 3 <= peaks[0] <= pos + 3


def test_doublet_free():
    """Test if doublet_func_free returns correct values."""
    counts = doublet_func_free(channels=CHANNELS, amp=1000, pos1=400, pos2=600, line_width1=10, line_width2=15)
    assert len(counts) == len(CHANNELS)
    assert 1000 * 0.95 <= np.max(counts) <= 1000 * 1.05
    peaks = find_peaks(counts)[0]
    assert len(peaks) == 2
    assert 399 <= peaks[0] <= 401
    assert 599 <= peaks[1] <= 601


def test_doublet_func():
    """Test if doublet_func returns correct values."""
    counts = doublet_func(channels=CHANNELS, doublet=DOUBLET, isomer_shift_ref=SPECTROSCOPE.isomer_shift_ref)
    assert len(counts) == len(CHANNELS)
    assert DOUBLET.amplitude * 0.95 <= np.max(counts) <= DOUBLET.amplitude * 1.05
    peaks = find_peaks(counts)[0]
    assert len(peaks) == 2
    pos1 = SPECTROSCOPE.isomer_shift_ref + DOUBLET.isomer_shift - DOUBLET.quadrupole_split / 2
    pos2 = SPECTROSCOPE.isomer_shift_ref + DOUBLET.isomer_shift + DOUBLET.quadrupole_split / 2
    assert pos1 - 3 <= peaks[0] <= pos1 + 3
    assert pos2 - 3 <= peaks[1] <= pos2 + 3


def test_sextet_free():
    """Test if sextet_func_free returns correct values."""
    counts = sextet_func_free(
        channels=CHANNELS,
        amp=1000,
        ratio13=3,
        ratio23=2,
        pos1=200,
        pos2=300,
        pos3=400,
        pos4=500,
        pos5=600,
        pos6=700,
        line_width1=5,
        line_width2=10,
        line_width3=15,
        line_width4=15,
        line_width5=10,
        line_width6=5,
    )
    assert len(counts) == len(CHANNELS)
    assert 1000 * 0.95 <= np.max(counts) <= 1000 * 1.05
    peaks = find_peaks(counts)[0]
    assert len(peaks) == 6
    assert 199 <= peaks[0] <= 201
    assert 299 <= peaks[1] <= 301
    assert 399 <= peaks[2] <= 401
    assert 499 <= peaks[3] <= 501
    assert 599 <= peaks[4] <= 601
    assert 699 <= peaks[5] <= 701


def test_sextet_func():
    """Test if sextet_func returns correct values."""
    counts = sextet_func(
        channels=CHANNELS,
        sextet=SEXTET,
        isomer_shift_ref=SPECTROSCOPE.isomer_shift_ref,
        zeeman_diff_ground=MATERIAL_CONSTANTS["Fe"].zeeman_diff_ground * SPECTROSCOPE.scale,
        zeeman_diff_excit=MATERIAL_CONSTANTS["Fe"].zeeman_diff_excited * SPECTROSCOPE.scale,
    )
    assert SEXTET.amplitude * 0.95 <= np.max(counts) <= SEXTET.amplitude * 1.05
    peaks = find_peaks(counts)[0]
    assert len(peaks) == 6
    true_line1_position = (
        SPECTROSCOPE.isomer_shift_ref
        + SEXTET.isomer_shift
        - 0.5
        * (
            MATERIAL_CONSTANTS["Fe"].zeeman_diff_ground * SPECTROSCOPE.scale
            + 3 * MATERIAL_CONSTANTS["Fe"].zeeman_diff_excited * SPECTROSCOPE.scale
        )
        * SEXTET.magnetic_field
        + 0.5 * SEXTET.quadrupole_split
    )
    true_line2_position = (
        SPECTROSCOPE.isomer_shift_ref
        + SEXTET.isomer_shift
        - 0.5
        * (
            MATERIAL_CONSTANTS["Fe"].zeeman_diff_ground * SPECTROSCOPE.scale
            + 1 * MATERIAL_CONSTANTS["Fe"].zeeman_diff_excited * SPECTROSCOPE.scale
        )
        * SEXTET.magnetic_field
        - 0.5 * SEXTET.quadrupole_split
    )
    true_line3_position = (
        SPECTROSCOPE.isomer_shift_ref
        + SEXTET.isomer_shift
        - 0.5
        * (
            MATERIAL_CONSTANTS["Fe"].zeeman_diff_ground * SPECTROSCOPE.scale
            - 1 * MATERIAL_CONSTANTS["Fe"].zeeman_diff_excited * SPECTROSCOPE.scale
        )
        * SEXTET.magnetic_field
        - 0.5 * SEXTET.quadrupole_split
    )
    true_line4_position = (
        SPECTROSCOPE.isomer_shift_ref
        + SEXTET.isomer_shift
        + 0.5
        * (
            MATERIAL_CONSTANTS["Fe"].zeeman_diff_ground * SPECTROSCOPE.scale
            - 1 * MATERIAL_CONSTANTS["Fe"].zeeman_diff_excited * SPECTROSCOPE.scale
        )
        * SEXTET.magnetic_field
        - 0.5 * SEXTET.quadrupole_split
    )
    true_line5_position = (
        SPECTROSCOPE.isomer_shift_ref
        + SEXTET.isomer_shift
        + 0.5
        * (
            MATERIAL_CONSTANTS["Fe"].zeeman_diff_ground * SPECTROSCOPE.scale
            + 1 * MATERIAL_CONSTANTS["Fe"].zeeman_diff_excited * SPECTROSCOPE.scale
        )
        * SEXTET.magnetic_field
        - 0.5 * SEXTET.quadrupole_split
    )
    true_line6_position = (
        SPECTROSCOPE.isomer_shift_ref
        + SEXTET.isomer_shift
        + 0.5
        * (
            MATERIAL_CONSTANTS["Fe"].zeeman_diff_ground * SPECTROSCOPE.scale
            + 3 * MATERIAL_CONSTANTS["Fe"].zeeman_diff_excited * SPECTROSCOPE.scale
        )
        * SEXTET.magnetic_field
        + 0.5 * SEXTET.quadrupole_split
    )
    assert true_line1_position - 3 <= peaks[0] <= true_line1_position + 3
    assert true_line2_position - 3 <= peaks[1] <= true_line2_position + 3
    assert true_line3_position - 3 <= peaks[2] <= true_line3_position + 3
    assert true_line4_position - 3 <= peaks[3] <= true_line4_position + 3
    assert true_line5_position - 3 <= peaks[4] <= true_line5_position + 3
    assert true_line6_position - 3 <= peaks[5] <= true_line6_position + 3


def test_spectrum_function():
    """Test if spectrum_func returns correct values for a spectrum with one singlet, doublet, and sextet."""
    counts = spectrum_func(
        channels=CHANNELS, spectrum=SPECTRUM, spectroscope=SPECTROSCOPE, geometry=SpectroscopeGeometry.REFLECTION
    )
    assert SPECTRUM.background * 0.95 <= get_spectrum_background(counts) <= SPECTRUM.background * 1.05
    peaks = find_peaks(counts)[0]
    assert len(peaks) == 9


def test_get_spectrum_sample_background_noise():
    """Test if generate_spectrum adds noise."""
    background = 10**5
    generated_spectrum = generate_spectrum(
        channels=CHANNELS,
        spectrum=SpectrumComputable(
            background=background,
            sextets=[
                SEXTET,
            ],
        ),
        spectroscope=SPECTROSCOPE,
        geometry=SpectroscopeGeometry.REFLECTION,
    )
    theoretical_noise = np.sqrt(background)
    assert theoretical_noise * 0.8 <= get_background_std(generated_spectrum) <= theoretical_noise * 1.2


def test_find_discrete_fold_point():
    """Test if find_discrete_fold_point returns correct value."""
    true_fold = 1000  # artificial unfolding
    folded = spectrum_func(
        channels=CHANNELS,
        spectrum=SpectrumComputable(
            background=10**5,
            sextets=[
                SEXTET,
            ],
        ),
        spectroscope=SPECTROSCOPE,
        geometry=SpectroscopeGeometry.REFLECTION,
    )

    unfolded_spectrum = np.append(folded[:true_fold], folded[true_fold - 1 :: -1])
    # add artificial noise to make left and right part of the spectrum different
    # not necessary, but it makes the test more robust
    unfolded_spectrum = unfolded_spectrum + (np.random.random(len(unfolded_spectrum)) - 0.5) * 0.2 * get_spectrum_span(
        folded
    )
    result = find_discrete_fold_point(unfolded_spectrum)
    assert result == true_fold


def test_find_adaptive_points_returns_correct_number_of_points():
    counts = np.random.poisson(100, 1000)
    adaptive_points, _ = find_adaptive_points(counts, n=256)
    assert len(adaptive_points) == 256

    adaptive_points, _ = find_adaptive_points(counts, n=64)
    assert len(adaptive_points) == 64
