import os

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
package_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "noiseprint2")
requirements = os.path.join(package_dir, "requirements.txt")
with open(requirements) as fh:
    install_requires = list(filter(lambda x: len(x), map(lambda x: x.strip(), fh.read().splitlines())))

setuptools.setup(
    name="noiseprint2",
    version="1.0.0",
    author="Francesco Tescari",
    description="Porting of noiseprint to tensorflow 2 and keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/francescotescari/noiseprint2",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    packages=["noiseprint2"],
    package_data={'noiseprint2': ['weights/**/.*', 'weights/**/*', ]},
    python_requires='>=3.5',
)
