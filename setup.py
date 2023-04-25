from setuptools import find_packages, setup

setup(
    name="custom_source_examples",
    version="1.0",
    description="Sample project for Datahub custom sources",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "acryl-datahub",
    ],
    extras_require={
        "mlflow": ["mlflow"],
        "mlflow-skinny": ["mlflow-skinny"],
    }
)
