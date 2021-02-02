import setuptools
from tools import get_requirements, get_readme, get_version


def main():
    setuptools.setup(
        name="ink",
        version=get_version(),
        author="THUNLP",
        author_email="thunlp@gmail.com",
        description="ink",
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
        install_requires=get_requirements()
    )

if __name__ == "__main__":
    main()