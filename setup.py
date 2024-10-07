from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='subsampwinner',
    version='0.0.8',
    author='Wei Dai',
    author_email='wdai@gmu.edu',
    description='A package for feature selection using Subsampling Winner Algorithm',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/wdai0/subsamp',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.26.4',
        'scipy>=1.13.0',
        'statsmodels>=0.14.2',
        'mpi4py>=3.1.6',
    ],
)