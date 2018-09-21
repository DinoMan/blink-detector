from setuptools import setup

setup(name='blink',
      version='0.1',
      description='Detects blinks in videos',
      packages=['blink'],
      package_dir={'blink': 'blink'},
      package_data={'blink': ['data/*.dat']},
      install_requires=[
          'dlib',
          'scikit-video',
          'opencv-python',
          'scikit-image',
          'scipy',
          'numpy'
      ],
      zip_safe=False)

