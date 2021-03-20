
try:
    from .thulac import THUlac
except ModuleNotFoundError:
    class THUlac:
        def __init__(self):
            raise NotImplementedError()
