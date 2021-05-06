import setuptools
from tools import get_requirements, get_readme, get_version

PACKAGE_SHORT_NAME = "oknlp"


def main():
    setuptools.setup(
        name=PACKAGE_SHORT_NAME,
        version=get_version(),
        author="THUNLP",
        author_email="thunlp@gmail.com",
        description="OKNLP",
        long_description=get_readme(),
        long_description_content_type="text/markdown",
        url="https://github.com/thunlp/ink",
        packages=setuptools.find_packages(exclude=("tools",)),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.6",
        setup_requires=["wheel"],
        install_requires=get_requirements(),
        include_package_data=True,
        package_data={
            PACKAGE_SHORT_NAME: [
                "data/resources/*.yaml",
                "data/resources/*.yml"
            ]
        },
        ext_modules=[
            setuptools.Extension("oknlp._C.thulac",
                extra_compile_args=["-O2","-Wall","-std=c++11"],
                sources = ["csrc/thulac/thulac_cls.cc"],
                include_dirs = [
                    "csrc/thulac/include"
                ],
            )
        ]
    )

if __name__ == "__main__":
    main()