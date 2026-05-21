from setuptools import setup, find_packages

setup(
    name="python-assignment",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "sqlalchemy>=2.0",
        "bokeh>=3.0",
        "matplotlib>=3.7",
    ],
    python_requires=">=3.10",
)
