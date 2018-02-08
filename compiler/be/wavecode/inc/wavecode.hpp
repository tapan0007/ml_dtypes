#pragma once

#ifndef KCC_WAVECODE_WAVECODE_H
#define KCC_WAVECODE_WAVECODE_H

#include <string>
#include <cstdio>



#include "tcc.hpp"

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"



namespace kcc {

namespace arch {
    class Arch;
}
namespace nets {
    class Network;
}
namespace layers {
    class Layer;
}
namespace wave {
    class WaveOp;
}

namespace wavecode {
class WaveCodeWaveOp;
class WaveCodeMatMul;
class WaveCodeSbAtom;



class WaveCode {
public:
    //----------------------------------------------------------------
    WaveCode(const nets::Network* network, const arch::Arch& arch);

    ~WaveCode();

    void generate(const char* objFileName);

    template<typename INSTR>
    void writeInstruction(INSTR& instruction)
    {
        fwrite(&instruction, sizeof(instruction), 1, m_ObjFile);
    }

private:
    WaveCode() = delete;
    WaveCode(const WaveCode&) = delete;

private:
    WaveCodeWaveOp& getCodeGen(const wave::WaveOp* waveOp);

private:
    const nets::Network*            m_Network;
    const arch::Arch&               m_Arch;
    FILE*                           m_ObjFile;
    std::unique_ptr<WaveCodeMatMul> m_CodeMatMul;
    std::unique_ptr<WaveCodeSbAtom> m_CodeSbAtom;
};

}}

#endif // KCC_WAVECODE_WAVECODE_H

