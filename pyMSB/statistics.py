"""
General statistical functions for Mossbauer data analysis.

This module contains functions for statistical analysis of Mossbauer spectra data.
Computations placed in this module should be fast and server to the user in real-time. Computations that are slow,
require a lot of resources and are run asynchronously should be placed in different place
"""

import numpy as np
import scipy


def get_spectrum_background(counts: np.ndarray[int] | list[int]) -> float:
    """
    Compute background of Mossbauer spectrum.

    Background is estimated as a mean of first and last 10 % of counts.
    Assumption: no resonance lines are present in this area. (Could be tested by selection distribution skewness?)

    :param counts: Mossbauer spectrum counts
    :return: Background estimate
    """
    n_counts = int(len(counts) * 0.1)
    background_estimate_left = np.mean(counts[0:n_counts])
    background_estimate_right = np.mean(counts[-n_counts:])
    background_estimate = (background_estimate_left + background_estimate_right) / 2
    return round(background_estimate)


def get_spectrum_span(counts: np.ndarray[int] | list[int]) -> int:
    """
    Compute span of Mossbauer spectrum (difference between max and min).

    :param counts: Mossbauer spectrum counts
    :return: Span of the spectrum
    """
    return max(counts) - min(counts)


def get_background_span(counts: np.ndarray[int] | list[int]) -> int:
    """
    Compute span of background in Mossbauer spectrum.

    :param counts: Mossbauer spectrum counts
    :return: Span of the background
    """
    n_counts = int(len(counts) * 0.1)
    side_counts = [*counts[0:n_counts], *counts[-n_counts:]]
    return max(side_counts) - min(side_counts)


def get_background_std(counts: np.ndarray[int] | list[int]) -> float:
    """
    Compute standard deviation of background in Mossbauer spectrum.

    :param counts: Mossbauer spectrum counts
    :return: Standard deviation of the spectrum background
    """
    n_counts = int(len(counts) * 0.1)
    side_counts = [*counts[0:n_counts], *counts[-n_counts:]]
    return round(np.std(side_counts), 2)


def get_mossbauer_effect(counts: np.ndarray[int] | list[int]) -> float:
    """
    Compute Mössbauer effect of a spectrum.

    Mössbauer effect (the probability of Mössbauer interaction)
    is estimated as the ratio of spectrum span to background span.

    :param counts: Mossbauer spectrum counts
    :return: Mössbauer effect
    """
    effect = (get_spectrum_span(counts) - get_background_span(counts)) * 100 / get_spectrum_background(counts)
    return round(effect, 2)


def is_transmission_spectrum(counts: np.ndarray[int] | list[int]) -> bool:
    """
    Estimates if the spectrum represents Transmission (and not Reflection).

    Transmission has peaks under background. Reflection has peaks above background.
    True if skewness is negative (peaks under background), False otherwise.

    :param counts: Mossbauer spectrum counts
    :return: True if Transmission, False if Reflection
    """
    skewness = scipy.stats.skew(counts)
    return bool(skewness <= 0)
