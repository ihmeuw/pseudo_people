import json
import sys
from pathlib import Path

from packaging.version import parse
from setuptools import find_packages, setup

with open("python_versions.json", "r") as f:
    supported_python_versions = json.load(f)

python_versions = [parse(v) for v in supported_python_versions]
min_version = min(python_versions)
max_version = max(python_versions)
active_version = parse(".".join([str(v) for v in sys.version_info[:2]]))

if not (min_version <= active_version <= max_version):
    py_version = ".".join([str(v) for v in sys.version_info[:3]])
    # Python 3.5 does not support f-strings
    error = (
        "\n----------------------------------------\n"
        "Error: Pseudopeople runs under python {min_version}-{max_version}.\n"
        "You are running python {py_version}\n".format(
            min_version=min_version.base_version,
            max_version=max_version.base_version,
            py_version=py_version,
        )
    )
    print(error, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    src_dir = base_dir / "src"

    about: dict[str, str] = {}
    with (src_dir / "pseudopeople" / "__about__.py").open() as f:
        exec(f.read(), about)

    with (base_dir / "README.rst").open() as f:
        long_description = f.read()

    install_requirements = [
        "pandas",
        "numpy<2.0.0",
        "pyyaml>=5.1",
        "pyarrow",
        "scipy",
        "tqdm",
        "layered_config_tree>=2.1.0",
        "loguru",
        # type stubs
        "pandas-stubs",
        "types-PyYAML",
        "types-docutils",
        "types-tqdm",
        "types-setuptools",
        "pyarrow-stubs",
    ]

    setup_requires = ["setuptools_scm"]

    interactive_requirements = [
        "IPython",
        "ipywidgets",
        "jupyter",
    ]

    dask_requirements = ["dask[distributed,dataframe]"]

    test_requirements = [
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "vivarium_testing_utils",
    ] + dask_requirements

    lint_requirements = [
        "black==22.3.0",
        "isort==5.13.2",
    ]

    doc_requirements = [
        "docutils",
        "sphinx>=4.0",
        "sphinx-rtd-theme>=0.6",
        "sphinx-click",
        "IPython",
        "matplotlib",
    ]

    setup(
        name=about["__title__"],
        description=about["__summary__"],
        long_description=long_description,
        license=about["__license__"],
        url=about["__uri__"],
        author=about["__author__"],
        author_email=about["__email__"],
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Natural Language :: English",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX",
            "Operating System :: POSIX :: BSD",
            "Operating System :: POSIX :: Linux",
            "Operating System :: Microsoft :: Windows",
            "Programming Language :: Python",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Education",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Life",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Software Development :: Libraries",
        ],
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        include_package_data=True,
        install_requires=install_requirements,
        tests_require=test_requirements,
        extras_require={
            "docs": doc_requirements,
            "test": test_requirements,
            "interactive": interactive_requirements,
            "dev": doc_requirements
            + test_requirements
            + interactive_requirements
            + lint_requirements,
            "dask": dask_requirements,
        },
        # entry_points="""
        #         [console_scripts]
        #         simulate=pseudopeople.interface.cli:simulate
        #     """,
        zip_safe=False,
        use_scm_version={
            "write_to": "src/pseudopeople/_version.py",
            "write_to_template": '__version__ = "{version}"\n',
            "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
        },
        setup_requires=setup_requires,
    )
