import os
def get_version():
    if "GITHUB_REF" in os.environ:
        # GITHUB ACTION ENV
        ref = os.environ["GITHUB_REF"]
        if ref.startswith("refs/tags/"):
            return ref[len("refs/tags/"):]
    if "GITHUB_SHA" in os.environ:
        return os.environ["GITHUB_SHA"]
    return "UNK"