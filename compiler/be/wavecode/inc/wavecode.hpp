#pragma once

#ifndef KCC_WAVECODE_WAVECODE_H
#define KCC_WAVECODE_WAVECODE_H

#include <array>
#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"



namespace kcc {

namespace arch {
    class Arch;
}
namespace nets {
    class Network;
}
namespace wave {
    class WaveOp;
}

namespace wavecode {
class WaveCodeWaveOp;
class WaveCodeMatMul;
class WaveCodeSbAtomFile;
class WaveCodeSbAtomSave;
class WaveCodePool;
class WaveCodeActivation;
class WaveCodeResAdd;



class WaveCode {
public:
    class NpyFileInfo {
    public:
        kcc_int64 m_FileDramOffset = -1;
        ARBPRECTYPE m_SimTypeId = INVALID_ARBPRECTYPE;
        std::array<kcc_int32, 4> m_RefFileShape;
    };
public:

    using OneInstrStream = FILE*;
    struct InstrStreams {
        OneInstrStream m_StreamProcInstrStream;
        OneInstrStream m_PeArrayInstrStream;
        OneInstrStream m_PoolEngInstrStream;
        OneInstrStream m_ActEngInstrStream;
    };
public:
    //----------------------------------------------------------------
    WaveCode(const nets::Network* network, const arch::Arch& arch);

    ~WaveCode();

    void generate(const InstrStreams& instrStreams);

    template<typename INSTR>
    void writeInstruction(INSTR& instruction);


    kcc_int64 gCurrentDramAddress(kcc_int64 sizeInBytes);
    kcc_int64 getDramForInputNpyFile(const std::string& fileName);
    kcc_int64 getDramForOutputNpyFile(const std::string& fileName);
    void recordDramForInputNpyFile(const std::string& fileName, kcc_int64 dramOffset);
    void recordDramForOutputNpyFile(const std::string& fileName, const NpyFileInfo& npyFileInfo);

private:
    WaveCode() = delete;
    WaveCode(const WaveCode&) = delete;

private:
    WaveCodeWaveOp& getCodeGen(const wave::WaveOp* waveOp);
    void saveAllNpyFiles();

private:
    const nets::Network*                m_Network;
    const arch::Arch&                   m_Arch;

    const InstrStreams*                 m_InstrStreams;
    std::unique_ptr<WaveCodeMatMul>     m_CodeMatMul;
    std::unique_ptr<WaveCodeSbAtomFile> m_CodeSbAtomFile;
    std::unique_ptr<WaveCodeSbAtomSave> m_CodeSbAtomSave;
    std::unique_ptr<WaveCodePool>       m_CodePool;
    std::unique_ptr<WaveCodeActivation> m_CodeActivation;
    std::unique_ptr<WaveCodeResAdd>     m_CodeResAdd;

    kcc_int64                           m_CurrentDramAddress;
    std::map<std::string, kcc_int64>    m_InputNpyFile2DramAddress;
    std::map<std::string, NpyFileInfo>  m_OutputNpyFile2DramAddress;
};

}}

#endif // KCC_WAVECODE_WAVECODE_H

