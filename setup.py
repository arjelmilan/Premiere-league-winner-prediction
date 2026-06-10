from setuptools import setup, find_packages

def get_requirements(file_path: str) -> list:
    with open(file_path) as file:
        requirements = file.read().splitlines()

    requirements = [req for req in requirements if req != "-e ."]
    return requirements

setup(
    name="pl_ml",
    version="0.0.1",
    author="Milan",
    author_email="milan@example.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)

print(get_requirements("requirements.txt"))