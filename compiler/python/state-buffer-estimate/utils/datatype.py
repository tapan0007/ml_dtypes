from abc             import ABCMeta, abstractmethod

##########################################################
class DataType(object, metaclass = ABCMeta):
    #DATA_TYPE_INT8  = 1
    #DATA_TYPE_INT16 = 2
    #DATA_TYPE_FP16  = 3

    @abstractmethod
    def gSizeInBytes(self):
        assert(False)

    @abstractmethod
    def gName(dataType):
        assert(False)

##########################################################
class DataTypeInt8(DataType):
    def __init__(self):
        super().__init__()

    def gSizeInBytes(self):
        return 1

    def gName(dataType):
        return "int8"

##########################################################
class DataTypeInt16(DataType):
    def __init__(self):
        super().__init__()

    def gSizeInBytes(self):
        return 2

    def gName(dataType):
        return "int16"

##########################################################
class DataTypeFp16(DataType):
    def __init__(self):
        super().__init__()

    def gSizeInBytes(self):
        return 2

    def gName(dataType):
        return "fp16"

