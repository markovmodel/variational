""" Variational Approach for conformation dynamics (VAC)

This package contains basis sets, estimators and solvers for the variational approach for
conformation dynamics, a theory that has been proposed in [1] and was further developed in
[2] and [3]. The variational approach is analogous to the Ritz method [4] that is
employed in computational quantum chemistry. It differs in the way how the involved
matrices are computed and in the meaning of the involved operators, eigenfunctions and
eigenvalues - see [3] for a comparison.

Roughly, the idea of the VAC is as follows: Given a (classical)
molecular dynamics trajectory with configurations {x_1, ..., x_T}, and a
set of basis functions defined on the space of configurations {chi_1(x), ..., chi_n(x)},
we compute the two correlation matrices:

c_ij (0)   = < chi_i(x_t) chi_j(x_t) >_t
c_ij (tau) = < chi_i(x_t) chi_j(x_t+tau) >_t

where < . >_t is average over time t. Of course this can be generalized to many trajectories.
Then we solve the generalized eigenvalue problem

C(tau) r = C(0) r l(tau)

where the eigenvalues l(tau) approximate the dominant eigenvalues of the Markov propagator
or Markov backward propagator of the underlying dynamics. The corresponding eigenfunction
of the backward propagator is approximated by

psi(x) = sum_i r_i chi_i(x)

Package functionalities
-----------------------

This package aims at providing code to help addressing a number of key problems:

1. Basis sets for molecular dynamics (MD), and in particular protein dynamics. See [5] for a
   first approach in this direction.

2. Estimators for the corration matrices C(0), C(tau). The trivial time-average that is usually
   employed has a number of problems especially for many short simulation trajectories that are
   initiated far from the equilibrium distribution (the usual case!).

3. Solvers for accurately solving the eigenvalue problem above, even for huge basis sets.

At this time only a few of the above functionalities are implemented and we will go step by step.
This package will undergo heavy development and there is currently no date for an official
release, so don't be surprised if the API (the look + feel of functions and classes) change.
At the moment this package is purely intended for development purposes, so use it at your own
risk.

Applications
------------
1. The time-lagged independent component analysis (TICA) method originally developed in [6] and
   proposed as an optimal data transformation method for building Markov state models of MD
   in [3,7] is a VAC with mean-free basis functions. Therefore you can easily implement TICA with
   this package.

2. By transforming the internal coordinates such as torsion angles or interatomic distances into
   suitable basis functions, you can approximate experimentally-measurable relaxation timescales
   and determine the corresponding structural rearrangements for peptides and proteins [2,5]

3. ... more will follow.

References
----------
[1] Noe, F. and Nueske, F. (2013): A variational approach to modeling slow processes
    in stochastic dynamical systems. SIAM Multiscale Model. Simul. 11, 635-655.

[2] Nueske, F., Keller, B., Perez-Hernandez, G., Mey, A.S.J.S. and Noe, F. (2014)
    Variational Approach to Molecular Kinetics. J. Chem. Theory Comput. 10, 1739-1752.

[3] Perez-Hernandez, G., Paul, F., Giorgino, T., De Fabritiis, G. and Noe, F. (2013)
    Identification of slow molecular order parameters for Markov model construction.
    J. Chem. Phys. 139, 015102.

[4] Ritz, W. (1909): Ueber eine neue Methode zur Loesung gewisser
    Variationsprobleme der mathematischen Physik. J. Reine Angew. Math., 135, 1-61.

[5] Vitalini, F., Noe, F. and Keller, B. (2015): A basis set for peptides for the
    variational approach to conformational kinetics. (In review).

[6] Molgedey, L. and Schuster H. G. (1994): Phys. Rev. Lett. 72, 3634.

[7] Schwantes, C. R. and Pande, V. S. (2000): J. Chem. Theory Comput. 9, 2000

"""
from __future__ import print_function
import os
from setuptools import setup
from os.path import relpath, join
import subprocess

DOCLINES = __doc__.split("\n")

########################
VERSION = "0.0.0"
ISRELEASED = False
__version__ = VERSION
########################
CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: Open BSD clause 2 (OpenBSD)
Programming Language :: Python
Topic :: Scientific/Engineering :: Bio-Informatics
Topic :: Scientific/Engineering :: Chemistry
Topic :: Scientific/Engineering :: Physics
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

################################################################################
# Writing version control information to the module
################################################################################

def git_version():
    # Return the git revision as a string
    # copied from numpy setup.py
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = 'Unknown'

    return GIT_REVISION


def write_version_py(filename='variational/version.py'):
    cnt = """
# This file is automatically generated by setup.py
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    else:
        GIT_REVISION = 'Unknown'

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()

################################################################################
# USEFUL SUBROUTINES
################################################################################

def find_package_data(data_root, package_root):
    files = []
    for root, dirnames, filenames in os.walk(data_root):
        for fn in filenames:
            files.append(relpath(join(root, fn), package_root))
    return files


################################################################################
# SETUP
################################################################################

write_version_py()

setup(
    name = 'variational',
    author = 'Frank Noe, Fabian Paul and Feliks Nueske',
    author_email = 'frank.noe@fu-berlin.de',
    description = DOCLINES[0],
    long_description = "\n".join(DOCLINES[2:]),
    version=__version__,
    license='OpenBSD',
    url='https://github.com/markovmodel/variational',
    platforms=['Linux', 'Mac OS-X', 'Unix', 'Windows'],
    classifiers=CLASSIFIERS.splitlines(),
    package_dir={'variational': 'variational'},
    packages=['variational', 'variational.basissets', 'variational.estimators', 'variational.solvers'],
    zip_safe=False,
    install_requires=[
        'numpy',
        'scipy',
        ],
    )

