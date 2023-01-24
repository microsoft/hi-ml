#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from setuptools import find_namespace_packages, setup  # type: ignore


long_description = Path("README.md").read_text(encoding="utf-8")
version = "0.1.2"
package_name = "hi-ml-multimodal"
install_requires = Path("requirements_run.txt").read_text().splitlines()

description = "Microsoft Health Futures package to work with multi-modal health data"

setup(
    name=package_name,
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/hi-ml",
    author="Biomedical Imaging Team @ Microsoft Health Futures",
    author_email="innereyedev@microsoft.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7"
    ],
    license="MIT License",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=install_requires,
)
