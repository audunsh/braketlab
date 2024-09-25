<img src="https://raw.githubusercontent.com/audunsh/braketlab/master/graphics/braketlab_logo.png" width = 250px>

# Out-of-the-box outside-of-the-box thinking

BraketLab is a pure Python-code for doing quantum theory in Jupyter Notebooks.

BraketLab is being developed by Audun Skau Hansen (a.s.hansen@kjemi.uio.no) at the Department of Chemistry, Hylleraas Centre for Quantum Molecular Sciences, University of Oslo.

# Documentation (in developement)

<a href="https://audunsh.github.io/braketlab/">Documentation on github pages</a>

# Installation

With PyPI
----

The BraketLab package is available in the Python Package Index (PyPI), and can
be installed from the command line using
```sh
pip install braketlab
```


From a local repository clone
----

The package can also be installed from a local clone of the repository. From
within this directory, simply run
```sh
pip install .
```
Optionally one can install with the `-e` (`--editable`) flag to let changes in
the source code take effect without having to perform a fresh install. The
command for this would be
```sh
pip install -e .
```


Development
====

Installation
----

Install the dependencies from the `requirements.txt` file with the command
```sh
pip install -r requirements.txt
```
This includes more packages than are included in the normal pip install of
BraketLab, like pytest for running the unit tests.

It is not necessary to install BraketLab to work on it, but doing so may
simplify the process by making it easier to deal with the import structure of
various modules. Install it from a local clone of the repository with the
command
```sh
pip install -e .
```
as explained in the Installation section.


Testing
----

We encourage, and try to stick to, the habit of writing unit tests for the code
wherever possible. These tests are stored in the `test` directory. To run all
tests using pytest, run the command
```sh
pytest
```
from within this directory.
