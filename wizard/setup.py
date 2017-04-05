from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np


ext_modules=[Extension("treejack_cython",["treejack_cython.pyx"],libraries=["m"], include_dirs=[np.get_include()])]

#setup(name="fast_loop",ext_modules=cythonize('fast_loop.pyx'),)
setup(name="treejack_cython",cmdclass={"build_ext":build_ext},ext_modules=ext_modules, include_dirs=[np.get_include()])