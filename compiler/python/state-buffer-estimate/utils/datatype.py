from abc             import ABCMeta, abstractmethod

##########################################################
class DataType(object, metaclass = ABCMeta):
    #DATA_TYPE_INT8  = 1
    #DATA_TYPE_INT16 = 2
    #DATA_TYPE_FLOAT16  = 3

    @abstractmethod
    def gSizeInBytes(self):
        assert(False)

    @abstractmethod
    def gName(self):
        assert(False)

    @abstractmethod
    def gTccName(self):
        assert(False)

##########################################################
class DataTypeInt8(DataType):
    def __init__(self):
        super().__init__()

    def gSizeInBytes(self):
        return 1

    def gName(self):
        return "int8"

    def gTccName(self):
        return "INT8"

##########################################################
class DataTypeInt16(DataType):
    def __init__(self):
        super().__init__()

    def gSizeInBytes(self):
        return 2

    def gName(self):
        return "int16"

    def gTccName(self):
        return "INT16"

##########################################################
class DataTypeFloat16(DataType):
    def __init__(self):
        super().__init__()

    def gSizeInBytes(self):
        return 2

    def gName(self):
        return "float16"

    def gTccName(self):
        return "FP16"

