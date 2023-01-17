from setuptools import setup, find_packages

setup(
    name='rnareplearn',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
        'rnareplearn = RNARepLearn.command_line:main']
    }
    )
