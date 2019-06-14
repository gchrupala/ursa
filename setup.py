from setuptools import setup
setup(
    name='ursa',
    version='0.2.0',
    url='https://github.com/gchrupala/ursa',
    packages=['ursa'],
    license='Apache License 2.0',
    description='Neural representation analysis with representational similarity',
    install_requires = [
        'scikit-learn ==      0.20.3',
        'scipy        ==      1.2.1',
        'conllu       ==      1.2.3',
        'mypy         ==      0.670',
        'mypy-extensions ==   0.4.1',
        'nltk             ==  3.4',
        'numpy            ==  1.16.2',
        'pandas           ==  0.24.1',
        'pytest           ==  4.3.0',
        'decorator        ==  4.3.2',
        'hypothesis       ==  4.7.19',
        'typing_extensions == 3.7.2'
        ]
)

