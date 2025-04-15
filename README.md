# **pyMSB**

The pyMSB is a Python package for the analysis of Mossbauer spectra featuring both Classical and Bayesian approach. Build and fit Mössbauer models to your datasets with ease and gain valuable insights into the underlying physics.

The pyMSB Python package is a part of the BayMoss project, which aims to provide a comprehensive platform for the analysis of Mossbauer spectra. The project is built on the same methods and models as this package, but wraps them in a user-friendly interface that allows you to easily share and discuss your results with whoever you chose. The goal of the BayMoss project is to accelerate and democratize the Mössbauer methods.

## Getting started

### Installation with Conda

Create new conda virtual environment with pymc and activate it.
```
conda create -c conda-forge -n pyMSB "pymc~=5.9.0"
conda activate pyMSB
```
Clone **pyMSB** [repository](https://github.com/PalackyUniversity/pyMSB) and naviagte inside the library directory:
```
git clone https://github.com/PalackyUniversity/pyMSB
cd pyMSB
```
Install requirements from `requiremnts.txt`:
```
pip install -r requirements.txt
```
Finally, install the **pyMSB** library (make sure you are inside the **pyMSB** library directory):
```
pip install .
```
Optionally, you may add `-e` flag to install the **pyMSB** library in editable mode.

Aditionaly, you may want to install `ipykernel` to enable Jupyter notebooks.
