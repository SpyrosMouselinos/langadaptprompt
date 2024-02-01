from setuptools import setup, find_packages

setup(
    name='langadaptprompt',
    version='2.3',
    author='Spyridon Mouselinos',
    author_email='mouselinos.spur.kw@gmail.com',
    packages=find_packages(exclude='test'),
    description='Port of LangChain prompting into pure Python functions.',
    url='https://github.com/SpyrosMouselinos/langadaptprompt',
    install_requires=[
        'matplotlib',
        'openai==0.28',
        'pandas',
        'transformers',
        'sentence_transformers',
        'torch',
        'scipy',
        'python-Levenshtein',
        'Levenshtein',
        'scikit-learn',
        'vertexai',
        'tqdm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
