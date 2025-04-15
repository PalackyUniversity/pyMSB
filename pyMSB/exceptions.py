class MossbauerExc(Exception):
    """Base class for all Mossbauer exceptions."""

    pass


class ValidationExc(MossbauerExc):
    """
    Exception raised for validation error.

    Usually occurs when the input parameters are not technically OK. For example,
    when the number of channels and counts do not match. Or when supplied with
    data in a wrong format (SingletComputable where SingletSpecs is expected).
    """

    pass


class PhysicsValidationExc(MossbauerExc):
    """
    Exception raised for physics validation error.

    Usually occurs when the input parameters are technically OK, but physically not sensible.
    For example, when trying to fit a calibration spectrum without sextet.
    """

    pass
