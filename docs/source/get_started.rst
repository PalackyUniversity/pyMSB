Installation
============

Create new conda virtual environment with pymc and activate it.

.. code-block:: bash

    conda create -c conda-forge -n pyMSB "pymc~=5.9.0"
    conda activate pyMSB


Clone **pyMSB** `repository <https://github.com/PalackyUniversity/pyMSB>`_ and naviagte inside the library directory:

.. code-block:: bash

    git clone https://github.com/PalackyUniversity/pyMSB
    cd pyMSB

Install requirements from `requiremnts.txt`:

.. code-block:: bash

    pip install -r requirements.txt


Finally, install the **pyMSB** library (make sure you are inside the **pyMSB** library directory):

.. code-block:: bash

    pip install .

Optionally, you may add `-e` flag to install the **pyMSB** library in editable mode.

Aditionaly, you may want to install `ipykernel` to enable Jupyter notebooks.
