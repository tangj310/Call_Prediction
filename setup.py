# ##
# This is the setup.py file to generate the necessary .egg
# ##

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as requirements_file:
    external_packages = requirements_file.read()

setuptools.setup(
    name="Call_Prediction",
    version="0.0.1 Call Prediction",
    author="JT ML/DS",
    description="Azure ML Implementation of The Call Prediction Pipeline version 0.0.1, including IUM tables and verint"
                "as target feature plus whix data, new verint join logic CONCAT(unit_num,SID) as speech_id",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires = external_packages,
    package_dir={"":"src"},
    packages=setuptools.find_namespace_packages(where="src\\"),
    include_package_data=True,
    python_requires=">=3.7",
)