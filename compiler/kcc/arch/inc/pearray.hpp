

##########################################################
class PeArray(object):

    #-----------------------------------------------------------------
    def __init__(self, numberRows, numberColumns):
        assert numberRows > 0 and numberColumns > 0
        if numberRows > numberColumns:
            assert numberRows % numberColumns == 0
        elif numberRows < numberColumns:
            assert numberColumns % numberRows == 0

        self.__NumberRows    = numberRows
        self.__NumberColumns = numberColumns

    #-----------------------------------------------------------------
    def gNumberRows(self):  ## IFMAPs
        return self.__NumberRows

    #-----------------------------------------------------------------
    def gNumberColumns(self):  ## OFMAPs
        return self.__NumberColumns

    #-----------------------------------------------------------------
    def gInstructionRamStartInBytes(self):
        return 0x001D00000

    #-----------------------------------------------------------------
    def gInstructionRamEndInBytes(self):
        return 0x001D03FFF

