from distutils.core import setup
import numpy

setup(name='robot_dataset',
      version='1.0',
      description='',
      author='',
      author_email='',
      url='',
      include_dirs=[numpy.get_include()],
      packages=['robot_dataset',  
            'robot_dataset.data', 
            'robot_dataset.config_parser', 
            'robot_dataset.dtypes',
            ],
      package_dir={'': 'src'},
      zip_safe=False
     )