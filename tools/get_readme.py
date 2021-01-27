from pathlib import Path

def get_readme():
    ret = ""
    with open(Path(__file__).parent.parent.joinpath("readme.md")) as frd:
        ret = frd.read()
    return ret