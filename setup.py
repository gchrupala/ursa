from setuptools import setup
setup(
    name='ursa',
    version='0.3.0',
    url='https://github.com/gchrupala/ursa',
    packages=['ursa'],
    license='Apache License 2.0',
    description='Neural representation analysis with representational similarity',
    install_requires = [
        'scikit-learn',
        'scipy',
        'conllu',
        'nltk',
        'numpy',
        'pandas'
        ]
)

