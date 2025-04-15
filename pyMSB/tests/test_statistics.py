import timeit

import pytest

from pyMSB import statistics
from pyMSB.data_utils import read_data_simple_format


@pytest.fixture
def spectrum_data():
    with open("mossbauer/tests/test_spectrum.txt", "rb") as f:
        data = read_data_simple_format(f)
    return data


TEST_SPECTRUM_STATS = {
    "background": 272908,
    "span": 25374,
    "background_span": 2484,
    "background_std": 559.94,
    "mossbauer_effect": 8.39,
    "is_transmission_spectrum": False,
}


def test_get_spectrum_background_computation(spectrum_data):
    """Test if get_spectrum_background returns correct value."""
    result = statistics.get_spectrum_background(spectrum_data)
    assert result == TEST_SPECTRUM_STATS["background"]


def test_get_spectrum_background_speed(spectrum_data):
    """Test if get_spectrum_background is fast enough."""
    elapsed_time = timeit.timeit(lambda: statistics.get_spectrum_background(spectrum_data), number=1000) / 1000
    assert elapsed_time < 0.01


def test_get_spectrum_span_computation(spectrum_data):
    """Test if get_spectrum_span returns correct value."""
    result = statistics.get_spectrum_span(spectrum_data)
    assert result == TEST_SPECTRUM_STATS["span"]


def test_get_spectrum_span_speed(spectrum_data):
    """Test if get_spectrum_span is fast enough."""
    elapsed_time = timeit.timeit(lambda: statistics.get_spectrum_span(spectrum_data), number=1000) / 1000
    assert elapsed_time < 0.01


def test_get_background_span_computation(spectrum_data):
    """Test if get_background_span returns correct value."""
    result = statistics.get_background_span(spectrum_data)
    assert result == TEST_SPECTRUM_STATS["background_span"]


def test_get_background_span_speed(spectrum_data):
    """Test if get_background_span is fast enough."""
    elapsed_time = timeit.timeit(lambda: statistics.get_background_span(spectrum_data), number=1000) / 1000
    assert elapsed_time < 0.01


def test_get_background_std_computation(spectrum_data):
    """Test if get_background_std returns correct value."""
    result = statistics.get_background_std(spectrum_data)
    assert round(result, 2) == TEST_SPECTRUM_STATS["background_std"]


def test_get_background_std_speed(spectrum_data):
    """Test if get_background_std is fast enough."""
    elapsed_time = timeit.timeit(lambda: statistics.get_background_std(spectrum_data), number=1000) / 1000
    assert elapsed_time < 0.01


def test_get_mossbauer_effect_computation(spectrum_data):
    """Test if get_mossbauer_effect returns correct value."""
    result = statistics.get_mossbauer_effect(spectrum_data)
    assert round(result, 2) == TEST_SPECTRUM_STATS["mossbauer_effect"]


def test_get_mossbauer_effect_speed(spectrum_data):
    """Test if get_mossbauer_effect is fast enough."""
    elapsed_time = timeit.timeit(lambda: statistics.get_mossbauer_effect(spectrum_data), number=1000) / 1000
    assert elapsed_time < 0.01


def test_is_transmission_spectrum_computation(spectrum_data):
    """Test if is_transmission_spectrum returns correct value."""
    result = statistics.is_transmission_spectrum(spectrum_data)
    assert result == TEST_SPECTRUM_STATS["is_transmission_spectrum"]


def test_is_transmission_spectrum_computation_opposite(spectrum_data):
    """Test if is_transmission_spectrum returns correct value for a flipped spectrum."""
    data = [-value + max(spectrum_data) for value in spectrum_data]
    result = statistics.is_transmission_spectrum(data)
    assert result is not TEST_SPECTRUM_STATS["is_transmission_spectrum"]


def test_is_transmission_spectrum_speed(spectrum_data):
    """Test if is_transmission_spectrum is fast enough."""
    elapsed_time = timeit.timeit(lambda: statistics.is_transmission_spectrum(spectrum_data), number=1000) / 1000
    assert elapsed_time < 0.01
