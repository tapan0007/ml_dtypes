from abc import ABCMeta, abstractmethod

##########################################################
class FmapDesc(object, metaclass = ABCMeta):

    #-----------------------------------------------------------------
    def __init__(self, num_maps, map_size):
        assert(num_maps > 0)
        assert(map_size > 0)
        self.__Num_maps = num_maps
        self.__Map_size = map_size

    #-----------------------------------------------------------------
    @abstractmethod
    def copy(self):
        assert(False)


    #-----------------------------------------------------------------
    def __str__(self):
        return "(" + str(self.gNumMaps()) + "," + str(self.gMapSize()) + ")"

    #-----------------------------------------------------------------
    def __eq__(self, other):
        return (self.gNumMaps() == other.gNumMaps()
            and self.gMapSize() == other.gMapSize())

    #-----------------------------------------------------------------
    def __ne__(self, other):
        return not self.__eq__(other)


    #-----------------------------------------------------------------
    def gNumMaps(self):
        return self.__Num_maps

    #-----------------------------------------------------------------
    def gMapSize(self):
        return self.__Map_size

    #-----------------------------------------------------------------
    def gNumPlanes(self):
        return self.gNumMaps()

    #-----------------------------------------------------------------
    def gPlaneSize(self):
        return self.gMapSize()

##########################################################
class OfmapDesc(FmapDesc):
    #-----------------------------------------------------------------
    def __init__(self, num_ofmaps, ofmap_size):
        super().__init__(num_ofmaps, ofmap_size)

    #-----------------------------------------------------------------
    def copy(self):
        return OfmapDesc(self.gNumMaps(), self.gMapSize())

