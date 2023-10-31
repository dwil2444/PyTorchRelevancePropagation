from setuptools import setup, find_packages

setup(
    name='torch_lrp',
    version='0.1.0',
    packages=find_packages(exclude="logger"),
    install_requires=[
        # List your dependencies here
    ],
    entry_points={
        'console_scripts': [
            # Define any command-line scripts here (if applicable)
        ],
    },
)
