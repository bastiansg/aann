from setuptools import find_packages, setup

setup(
    name="aann",
    packages=find_packages(),
    version="1.0.0",
    install_requires=[
        "more-itertools==9.0.0",
        "joblib==1.2.0",
        "pytorch-lightning==1.9.2",
        "rich==12.5.1",
        "tqdm==4.64.1",
        "Pillow==9.4.0",
        "opencv-contrib-python-headless==4.7.0.68",
        "torchdata==0.5.1",
        "networkx==3.0",
        "pycairo==1.23.0",
        "matplotlib==3.7.0",
    ],
    package_data={"": ["*.yml", "*.yaml"]},
    include_package_data=True,
    classifiers=["Programming Language :: Python :: 3"],
)
