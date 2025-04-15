import numpy as np


def channel_to_velocity(values: np.ndarray[int], scale: float, isomer_shift_ref: float) -> np.ndarray[float]:
    """
    Convert channel values to velocity/energy values.

    General formula: velocity = (channel - isomer_shift_ref) / scale
    Used for plot x-axis conversion.

    :param values: Array of channel values to convert.
    :param scale: Scale factor for conversion.
    :param isomer_shift_ref: Position of energetic zero in channels.
    :returns: Array of velocity/energy values.
    """
    return (values - isomer_shift_ref) / scale


def velocity_to_channel(values: np.ndarray[float], scale: float, isomer_shift_ref: float) -> np.ndarray[float]:
    """
    Convert velocity/energy values to channel values.

    General formula: channel = velocity * scale + isomer_shift_ref
    Used for plot x-axis conversion.

    :param values: Array of velocity/energy values to convert.
    :param scale: Scale factor for conversion.
    :param isomer_shift_ref: Position of energetic zero in channels.
    :returns: Array of channel values.
    """
    return (values * scale) + isomer_shift_ref


def channel_span_to_velocity_span(value: float | np.ndarray[float], scale: float) -> float | np.ndarray[float]:
    """
    Convert channel span to velocity/energy span.

    General formula: velocity_span = channel_span / scale
    Used for isomer_shift, quadrupole split and line width conversion.

    :param value: Channel span value(s).
    :param scale: Scale factor for conversion.
    :returns: Velocity/energy span value(s).
    """
    if value is not None:
        return value / scale


def velocity_span_to_channel_span(value: float | np.ndarray[float], scale: float) -> float | np.ndarray[float]:
    """
    Convert velocity/energy span to channel span.

    General formula: channel_span = velocity_span * scale
    Used for isomer_shift, quadrupole split and line width conversion.

    :param value: Velocity/energy span value(s).
    :param scale: Scale factor for conversion.
    :returns: Channel span value(s).
    """
    if value is not None:
        return value * scale
