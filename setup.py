from setuptools import find_packages, setup


# Read requirements files
requirements_file = "requirements.txt"
with open(requirements_file) as buffer:
    requirements = buffer.read().splitlines()

requirements = list(set(requirements))
requirements_string = "\n  ".join(requirements)
print(f"Found the following requirements to be installed from {requirements_file}:\n  {requirements_string}")


# Collect packages
packages = find_packages(exclude=("tests", "experiments"))
print("Found the following packages to be created:\n  {}".format("\n  ".join(packages)))


# Get long description from README
with open("README.md", "r") as readme:
    long_description = readme.read()


setup(
    name="scholarscrape",
    version="1.0.0",
    packages=packages,
    python_requires=">=3.10.0",
    install_requires=requirements,
    setup_requires=[],
    ext_modules=[],
    url="https://github.com/JakobHavtorn/scholarscrape",
    author="Jakob D. Havtorn",
    description="Interfacing with Semanticscholar API for greatness",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
