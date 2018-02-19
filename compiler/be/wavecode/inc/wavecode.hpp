#pragma once

#ifndef KCC_WAVECODE_WAVECODE_H
#define KCC_WAVECODE_WAVECODE_H

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
namespace layers {
    class Layer;
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



class WaveCode {
public:
    class NpyFileInfo {
    public:
        kcc_int64 m_FileDramOffset = -1;
        ARBPRECTYPE m_SimTypeId = INVALID_ARBPRECTYPE;
        std::array<kcc_int32, 4> m_RefFileShape;
    };
public:
    enum UseStream {
        UseStream_StreamProc,
        UseStream_PeArray,
        UseStream_PoolEng,
        UseStream_ActEng
    };

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
    void writeInstruction(INSTR& instruction, UseStream whichStream)
    {
        OneInstrStream strm;
        switch (whichStream) {
        case UseStream_StreamProc:
            strm = m_InstrStreams->m_StreamProcInstrStream;
            break;
        case UseStream_PeArray:
            strm = m_InstrStreams->m_PeArrayInstrStream;
            break;
        case UseStream_PoolEng:
            strm = m_InstrStreams->m_PoolEngInstrStream;
            break;
        case UseStream_ActEng:
            strm = m_InstrStreams->m_ActEngInstrStream;
            break;
        default:
            assert(false && "Wrong instruction stream type");
            break;
        }
        fwrite(&instruction, sizeof(instruction), 1, strm);
    }


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

    kcc_int64                           m_CurrentDramAddress;
    std::map<std::string, kcc_int64>    m_InputNpyFile2DramAddress;
    std::map<std::string, NpyFileInfo>  m_OutputNpyFile2DramAddress;
};

}}

#endif // KCC_WAVECODE_WAVECODE_H

