from setuptools import setup

setup(
    name='redistributor',
    version='0.2',
    author='Pavol Harar',
    author_email='pavol.harar@gmail.com',
    packages=['redistributor'],
    description='Package for transformation of data from arbitrary distribution to arbitrary distribution.',
    long_description=open('readme.md').read(),
    install_requires=['numpy>=1.14.5', 'scipy>=1.1.0', 'scikit_learn>=0.19.1', 'psutil>=5.4.6'],
    license='MIT',
)
