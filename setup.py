from setuptools import setup

setup(
    name='deepliif',
    version='0.0.1',
    packages=['deepliif', 'deepliif.data', 'deepliif.models', 'deepliif.options', 'deepliif.util'],
    install_requires=[
        'opencv-python==4.5.3.56',
        'torchvision==0.10.0',
        'scikit-image==0.18.3',
        'dominate==2.6.0',
        'numba==0.53.1'
    ]
)
