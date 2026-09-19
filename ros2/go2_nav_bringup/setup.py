import os
from glob import glob

from setuptools import find_packages, setup

package_name = "go2_nav_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="tak",
    maintainer_email="t.tamura@openreach.tech",
    description="Go2 robot layer for the VLFM nav stack: config, launch, sim glue nodes.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "sim_lowstate_bridge = go2_nav_bringup.sim_lowstate_bridge:main",
        ],
    },
)
