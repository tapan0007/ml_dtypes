from .genlayer import GenLayer


##########################################################
class GenDataLayer(GenLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generate(self, layer):
        f = self.gFile()
        ind = self.gIndent()
        s = [ "// " + layer.gName(),
              "compile_read_ifmap(out_binary,",
              ind + "const addr_t ifmap_sb_addr, const char *in_numpy_fname,",
              ind + "const char *numpy_layout);",
            ]

        ss = ""
        for x in s: ss += ind + x + "\n"
        f.write(ss)



