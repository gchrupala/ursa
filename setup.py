from setuptools import setup, find_packages
setup(
    name='ursa',
    version='0.3.1',
    url='https://github.com/gchrupala/ursa',
    license='Apache License 2.0',
    description='Neural representation analysis with representational similarity',
    packages=find_packages(exclude='test'),
    install_requires = [
        'scikit-learn',
        'scipy',
        'conllu',
        'nltk',
        'numpy',
        'pandas',
        'Levenshtein'
        ]
)

