from setuptools import setup
setup(name='cami',
version='1.1',
description='Causality toolbox for Python',
url='https://github.com/artvalencio/cami-python',
author='Arthur Valencio, IC-Unicamp, RIDC NeuroMat',
author_email='arthur@liv.ic.unicamp.br',
license='MIT',
packages=['cami'],
classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
python_requires='>=3.6',
install_requires=['pandas','numpy','matplotlib','scipy'],
zip_safe=False)
