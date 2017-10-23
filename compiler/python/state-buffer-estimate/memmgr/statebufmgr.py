from arch.arch import Arch

class StateBufferMgr(object):
    def __init__(self, arch):
        assert(isinstance(arch, Arch))
