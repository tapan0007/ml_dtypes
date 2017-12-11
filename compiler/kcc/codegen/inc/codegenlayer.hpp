#pragma once

#ifndef KCC_CODEGEN_CODEGENLAYER_H
#define KCC_CODEGEN_CODEGENLAYER_H

namespace kcc {
namespace codegen {

class MacroLayer {
    //----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        self.__MacroInstrGen = macroInstrGen
        self.__Indent = "    "

    //----------------------------------------------------------------
    def gMacroInstrGen(self):
        return self.__MacroInstrGen

    //----------------------------------------------------------------
    def gFile(self):
        return self.__MacroInstrGen.gFile()

    //----------------------------------------------------------------
    def gIndent(self):
        return self.__MacroInstrGen.gIndent()

    //----------------------------------------------------------------
    @abstractmethod
    def generate(self):
        assert(False)

    //----------------------------------------------------------------
    def gLayer(self):
        return self.__Layer

    //----------------------------------------------------------------
    def rLayer(self, layer):
        self.__Layer = layer

}}

#endif // KCC_CODEGEN_CODEGENLAYER_H

