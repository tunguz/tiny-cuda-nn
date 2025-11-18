import os
from setuptools import find_packages, setup

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

with open(os.path.join(ROOT_DIR, "CMakeLists.txt"), "r", encoding="utf-8") as cmakelists:
	for line in cmakelists:
		line = line.strip()
		if line.startswith("VERSION"):
			VERSION = line.split("VERSION", 1)[-1].strip()
			break
	else:  # pragma: no cover - fallback
		VERSION = "0.0.0"

with open(os.path.join(SCRIPT_DIR, "README.md"), "r", encoding="utf-8") as readme:
	LONG_DESCRIPTION = readme.read()

setup(
	name="tinycudann-sklearn",
	version=VERSION,
	description="sklearn-style Python bindings for tiny-cuda-nn",
	long_description=LONG_DESCRIPTION,
	long_description_content_type="text/markdown",
	packages=find_packages(),
	python_requires=">=3.8",
	install_requires=["numpy", "torch"],
	extras_require={"cuda": ["tinycudann"]},
	url="https://github.com/NVlabs/tiny-cuda-nn",
	license="BSD-3-Clause",
	classifiers=[
		"Programming Language :: Python",
		"Programming Language :: Python :: 3",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
	],
)
