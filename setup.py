from setuptools import setup

setup(
    name="codenames-solvers",
    version="1.0.0",
    description="Codenames board game solvers implementation in python.",
    author="Asaf Kali",
    author_email="akali93@gmail.com",
    url="https://github.com/asaf-kali/codenames-solvers",
    install_requires=[
        # Core
        "codenames~=1.0",
        "pydantic~=1.9",
        "numpy~=1.21",
        "gensim~=4.1",
        "pandas~=1.3",
        "scipy~=1.7",
        "generic-iterative-stemmer~=0.2",
        # Algo
        "networkx~=2.6",
        "python-louvain~=0.15",
        "editdistance~=0.6",
    ],
    include_package_data=True,
)
