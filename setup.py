# Copyright 2024 Chris Odom
# MIT License

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ngest",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "aiofiles",
        "aiohttp",
        "aiolimiter",
        "aiorate",
        "async-generator",
        "clang",  # Note: This requires LLVM to be installed separately
        "python-dotenv",
        "esprima",
        "neo4j",
        "ollama",
        "Pillow",
        "PyPDF2",
        "python-magic",
        "syn",
        "tenacity",
        "torch",
        "tqdm",
        "torchvision",
        "transformers",
        "faiss-cpu",  # or faiss-gpu if you're using GPU
        "numpy",
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-asyncio',
        ],
    },
    
    entry_points={
        'console_scripts': [
            'ngest=ngest.main:run_main',
        ],
    },
    author="Chris Odom",
    author_email="chris@opentransactions.org",
    description="For ingesting source code and other files into a semantic graph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fellowtraveler/ngest",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
