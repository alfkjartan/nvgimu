#!/usr/bin/env python
#
# Copyright (C) 2009-2011 University of Edinburgh
#
# This file is part of IMUSim.
#
# IMUSim is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# IMUSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with IMUSim.  If not, see <http://www.gnu.org/licenses/>.

depsOK = True

try:
    import numpy
except ImportError:
    depsOK = False
    print "NumPy should be installed first from suitable binaries."
    print "See http://numpy.scipy.org/"

try:
    import scipy
except ImportError:
    depsOK = False
    print "SciPy should be installed first from suitable binaries."
    print "See http://www.scipy.org/"

try:
    import matplotlib
except ImportError:
    depsOK = False
    print "Matplotlib should be installed first from suitable binaries."
    print "See http://matplotlib.sf.net/"

try:
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
    from Cython.Build import cythonize

    if depsOK:
        setup(
            name = "nvgimu",
            version = "0.2",
            author = "Kjartan Halvorsen",
            license = "GPLv3",
            url = "",
            install_requires = ["pyparsing",],
            packages = find_packages(),
            include_dirs = [numpy.get_include()],
            ext_modules = cythonize("nvg/maths/*.pyx")
            #     ext_modules = [
            #         Extension("nvg.maths.quaternions",
            #             ['nvg/maths/quaternions.c']),
            #         Extension("nvg.maths.quat_splines",
            #             ['nvg/maths/quat_splines.c']),
            #         Extension("nvg.maths.vectors",['nvg/maths/vectors.c'])]
            #
            )
except ImportError:
    print "Setuptools must be installed - see http://pypi.python.org/pypi/setuptools"
