

##########################################################
class PeArray(object):

    #-----------------------------------------------------------------
    def __init__(self):
        self.__NumberRows    = 128
        self.__NumberColumns = 64
        
    #-----------------------------------------------------------------
    def gNumberRows(self):  ## IFMAPs
        return self.__NumberRows

    #-----------------------------------------------------------------
    def gNumberColumns(self):  ## OFMAPs
        return self.__NumberColumns

