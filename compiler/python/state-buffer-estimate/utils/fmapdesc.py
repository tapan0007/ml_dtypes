from abc import ABCMeta, abstractmethod

##########################################################
class FmapDesc(object, metaclass = ABCMeta):

    #-----------------------------------------------------------------
    def __init__(self, num_maps, map_size):
        assert num_maps > 0
        self.__Num_maps = num_maps

        #self.__MapSize = map_size
        try:
            (w, h) = map_size
            assert w > 0 and h > 0
            self.__MapWidth = w
            self.__MapHeight = h
        except:
            assert isinstance(map_size, int)
            assert(map_size > 0)
            self.__MapWidth = map_size
            self.__MapHeight = map_size

    #-----------------------------------------------------------------
    @abstractmethod
    def copy(self):
        assert(False)


    #-----------------------------------------------------------------
    def __str__(self):
        return "(" + str(self.gNumMaps()) + "," 
        + str(self.gMapWidth()) + '*' + str(self.gMapHeight()) 
        + ")"

    #-----------------------------------------------------------------
    def __eq__(self, other):
        return (self.gNumMaps() == other.gNumMaps()
            and self.gMapWidth() == other.gMapWidth()
            and self.gMapHeight() == other.gMapHeight()
            )

    #-----------------------------------------------------------------
    def __ne__(self, other):
        return not self.__eq__(other)


    #-----------------------------------------------------------------
    def gNumMaps(self):
        return self.__Num_maps

    #-----------------------------------------------------------------
    def gNumPlanes(self):
        return self.gNumMaps()

    #-----------------------------------------------------------------
    #def gMapSize(self):
    #    return self.__MapSize

    #def gPlaneSize(self):
    #    return self.gMapSize()
    
    #-----------------------------------------------------------------
    def gMapWidth(self):
        return self.__MapWidth

    #-----------------------------------------------------------------
    def gMapHeight(self):
        return self.__MapHeight

    #-----------------------------------------------------------------
    def gPlaneWidth(self):
        return self.gMapWidth()

    #-----------------------------------------------------------------
    def gPlaneHeight(self):
        return self.gMapHeight()

##########################################################
class OfmapDesc(FmapDesc):
    #-----------------------------------------------------------------
    def __init__(self, num_ofmaps, ofmap_size):
        super().__init__(num_ofmaps, ofmap_size)

    #-----------------------------------------------------------------
    def copy(self):
        return OfmapDesc(self.gNumMaps(), (self.gMapWidth(), self.gMapHeight()))

