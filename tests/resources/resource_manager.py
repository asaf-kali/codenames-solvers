import os

RESOURCE_DIR = os.path.join(os.path.dirname(__file__))


def get_resource_path(resource_name: str) -> str:
    return os.path.join(RESOURCE_DIR, resource_name)
