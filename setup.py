from setuptools import setup

BASE_DEPENDENCIES = [
    # Core
    "the-spymaster-util>=2.0",
    "codenames~=1.2",
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
]
GPT_DEPENDENCIES = ["openai~=0.27"]
ALL_DEPENDENCIES = BASE_DEPENDENCIES + GPT_DEPENDENCIES

setup(
    name="codenames-solvers",
    version="1.2.3",
    description="Codenames board game solvers implementation in python.",
    author="Asaf Kali",
    author_email="asaf.kali@mail.huji.ac.il",
    url="https://github.com/asaf-kali/codenames-solvers",
    install_requires=BASE_DEPENDENCIES,
    extras_require={
        "all": ALL_DEPENDENCIES,
        "gpt": GPT_DEPENDENCIES,
    },
    include_package_data=True,
)
