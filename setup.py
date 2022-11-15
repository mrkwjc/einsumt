import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="einsumt",
    version="0.9.3",
    author="Marek Wojciechowski",
    author_email="mrkwjc@gmail.com",
    description="Multithreaded version of numpy.einsum function",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrkwjc/einsumt",
    packages=setuptools.find_packages(),
    keywords=['numpy', 'einsum', 'hpc'],
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
    py_modules=['einsumt']
)
