from setuptools import setup, find_packages

setup(
    name="my_langgraph_app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph-cli[inmem]",
    ],
)
