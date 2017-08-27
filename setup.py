from setuptools import setup, find_packages

setup(
    name='prada_bayes_opt',
    version='0.3',
    packages = ["prada_bayes_opt"],
    include_package_data = True,
    #py_modules = ['prada_bayes_opt.__init__'],
    description='BayesianOptimization Prada package',
    install_requires=[
        "numpy >= 1.11.0",
        "scipy >= 0.17.0",
        "scikit-learn >= 0.17.1",
    ],
)
