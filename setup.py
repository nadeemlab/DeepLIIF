from setuptools import setup

setup(
    name='deepliif',
    version='0.0.1',
    packages=['deepliif', 'deepliif.data', 'deepliif.models', 'deepliif.util'],
    py_modules=['cli'],
    install_requires=[
        'opencv-python==4.5.3.56',
        'torchvision==0.10.0',
        'scikit-image==0.18.3',
        'dominate==2.6.0',
        'numba==0.53.1',
        'Click==8.0.3',
        'requests==2.26.0',
        'dask==2021.11.2',
        'visdom>=0.1.8.3'
    ],
    entry_points={
        'console_scripts': [
            'deepliif = cli:cli'
        ]
    }
)
