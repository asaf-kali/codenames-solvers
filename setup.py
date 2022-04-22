from setuptools import setup

setup(
    name="codenames",
    version="1.0.2",
    description="Codenames game logic and solvers implementation in python.",
    author="Asaf Kali",
    author_email="akali93@gmail.com",
    url="https://github.com/asaf-kali/codenames",
    install_requires=[
        # Core
        "numpy~=1.21",
        "gensim~=4.1",
        "pandas~=1.3",
        "scipy~=1.7",
        "generic-iterative-stemmer~=0.2",
        # Algo
        "networkx~=2.6",
        "python-louvain~=0.15",
        "editdistance~=0.6",
        # Web
        "selenium~=4.1.0",
        # CLI
        "beautifultable~=1.0",
    ],
    include_package_data=True,
)