from setuptools import setup, find_packages

setup(
    name='jobo',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23.5",
        "matplotlib>=3.7.2"
    ],
    # Additional metadata about your package
    author='Johannes Lenfers',
    author_email='j.lenfers@uni-muenster.de',
    description='A Bayesian Optimiation Playground',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # If your README is in Markdown
    url='https://github.com/johanneslenfers/jobo',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    # entry_points={"console_scripts": ["baco = baco.run:main"]},
)
