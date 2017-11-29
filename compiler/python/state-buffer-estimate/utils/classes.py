
class Final(type):

    def __init__(cls, name, bases, namespace):
        # super(Final, cls)
        super().__init__(name, bases, namespace)
        for cls in bases:
            if isinstance(cls, Final):
                raise TypeError("Base class " + str(cls.__name__) + " is final")


