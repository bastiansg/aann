from setuptools import find_packages, setup

setup(
    name="aann",
    packages=find_packages(),
    version="1.0.0",
    install_requires=[
        "more-itertools==9.1.0",
        "joblib==1.2.0",
        "pytorch-lightning==2.0.1",
        # "rich==12.5.1",
        "rich==13.3.3",
        "Pillow==9.5.0",
        "opencv-contrib-python-headless==4.7.0.72",
        "torchdata==0.6.0",
        "networkx==3.1",
        "pycairo==1.23.0",
        "matplotlib==3.7.1",
    ],
    package_data={"": ["*.yml", "*.yaml"]},
    include_package_data=True,
    classifiers=["Programming Language :: Python :: 3"],
)
