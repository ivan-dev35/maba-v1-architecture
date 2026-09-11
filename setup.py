from setuptools import setup, find_packages

setup(
    name="maba",
    version="1.0.0",
    description="Maba v1 Architecture: 101M parameter linear recurrence and attention model",
    author="Andrew Thompson",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "maba = maba.cli:main",
        ],
    },
)
