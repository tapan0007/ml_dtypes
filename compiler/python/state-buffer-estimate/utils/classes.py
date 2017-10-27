
class Final(type):

    def __init__(cls, name, bases, namespace):
        # super(Final, cls)
        super().__init__(name, bases, namespace)
        for klass in bases:
            if isinstance(klass, Final):
                raise TypeError("class " + str(klass.__name__) + " is final")


