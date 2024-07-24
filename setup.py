from setuptools import setup
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='deepliif',
    version='1.1.11',
    packages=['deepliif', 'deepliif.data', 'deepliif.models', 'deepliif.util', 'deepliif.options'],

    description='DeepLIIF: Deep-Learning Inferred Multiplex Immunofluorescence for Immunohistochemical Image Quantification',
    author='Parmida93',
    author_email='ghahremani.parmida@gmail.com',
    url='https://github.com/nadeemlab/DeepLIIF',
    long_description=README,
    long_description_content_type="text/markdown",
    keywords=['DeepLIIF', 'IHC', 'Segmentation', 'Classification'],
    classifiers=[],

    py_modules=['cli'],
    install_requires=[
        "opencv-python==4.5.3.56",
        "torchvision==0.10.0",
        "scikit-image==0.18.3",
        "dominate==2.6.0",
        "numba==0.57.1",
        "Click==8.0.3",
        "requests==2.26.0",
        "dask==2021.11.2",
        "visdom>=0.1.8.3",
        "python-bioformats>=4.0.6"
    ],
    entry_points={
        'console_scripts': [
            'deepliif = cli:cli'
        ]
    }
)
