import os
from glob import glob

from setuptools import setup

package_name = "vlfm_nav_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "params"), glob("params/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="tak",
    maintainer_email="t.tamura@openreach.tech",
    description="Robot-agnostic launch and parameters for the VLFM nav stack (mapping, Nav2).",
    license="Apache-2.0",
)
