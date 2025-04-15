Examples
========

TODO: Finish the examples when the API is stable.

Spectrum definition
-------------------

First, we define individual subspectra. The following code defines three subspectra: Singlet, Doublet, and Sextet.

.. code-block:: python

    singlet = SingletComputable(name='Singlet', amp = 1000, isomer_shift = 0, line_width1 = 10)
    doublet = DoubletComputable(name='Doublet',
                                amp = 1000,
                                isomer_shift = 0,
                                quadrupole_split = 20,
                                line_width1 = 10,
                                line_width2 = 10)
    sextet = SextetComputable(name='Sextet',
                              amp = 1000,
                              isomer_shift = 0,
                              quadrupole_split = 20,
                              line_width1 = 10,
                              line_width2 = 10,
                              line_width3 = 10,
                              line_width4 = 10,
                              line_width5 = 10,
                              line_width6 = 10)


Then, we define a spectrum by combining the subspectra and background.

.. code-block:: python

    spectrum = SpectrumComputable(background = 10 ** 4,
                                  singlets = [singlet],
                                  doublets = [doublet],
                                  sextets = [sextet])

Then, we define a spectroscope.

.. code-block:: python

    spectroscope = SpectroscopeComputable(scale=40,
                                          isomer_shift_reference=512)


Finally, we can evaluate the spectrum function and sample the spectrum counts.

.. code-block:: python

    channels = np.arange(1024)
    spectrum_function = spectrum_func(channels = channels,
                                      spectrum = spectrum,
                                      spectroscope = spectroscope,
                                      geometry = SpectroscopeGeometry.TRANSMISSION)

    spectrum_counts = generate_spectrum(channels = channels,
                                        spectrum = spectrum,
                                        spectroscope = spectroscope,
                                        geometry = SpectroscopeGeometry.TRANSMISSION)

The whole example including the plotting with just Sextet subspectrum.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from pyMSB.models import (SingletComputable,
                                  DoubletComputable,
                                  SextetComputable,
                                  SpectrumComputable,
                                  SpectroscopeComputable,
                                  SpectroscopeGeometry)
    from pyMSB.common import spectrum_func, generate_spectrum

    sextet = SextetComputable(name='Sextet',
                              amp = 1000,
                              isomer_shift = 0,
                              quadrupole_split = 20,
                              line_width1 = 10,
                              line_width2 = 10,
                              line_width3 = 10,
                              line_width4 = 10,
                              line_width5 = 10,
                              line_width6 = 10)
    spectrum = SpectrumComputable(background = 10 ** 4,
                                  singlets = [],
                                  doublets = [],
                                  sextets = [sextet])
    spectroscope = SpectroscopeComputable(scale=40,
                                          isomer_shift_reference=512)
    spectrum_function = spectrum_func(channels = np.arange(1024),
                                      spectrum = spectrum,
                                      spectroscope = spectroscope,
                                      geometry = SpectroscopeGeometry.TRANSMISSION)
    spectrum_counts = generate_spectrum(channels = np.arange(1024),
                                        spectrum = spectrum,
                                        spectroscope = spectroscope,
                                        geometry = SpectroscopeGeometry.TRANSMISSION)
    plt.plot(channels, spectrum_function, label='Spectrum function')
    plt.plot(channels, spectrum_counts, "k+" label='Spectrum counts')
    plt.legend()
    plt.grid()
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    plt.show()

TODO result image

Calibration
-----------
TODO

Analysis
--------
TODO

