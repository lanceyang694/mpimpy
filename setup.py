import re
from setuptools import setup, find_packages

# read the contents of your README file
with open("README.md", 'r', encoding='utf-8') as f:
    long_description = f.read()

with open("mpimpy/__init__.py") as f:
    version = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)

setup(
    name="mpimpy",
    version=version,
    description='A python algorithm-hardware co-design framework for memristor-based in-memory computing.',
    author="Ling Yang",
    author_email="3299285328@qq.com",
    maintainer="Researchers from Prof. Yi Li's group at HUST",
    url="https://github.com/lanceyang694",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    install_requires=["numpy", "scipy", "matplotlib", "pandas"],
    python_requires=">=3.6",
    
)