#pragma once

#ifndef KCC_WAVECODE_WAVECODE_H
#define KCC_WAVECODE_WAVECODE_H

#include <array>
#include <map>
#include <string>
#include <cstdio>
#include <memory>


#include "shared/inc/tpb_isa.hpp"


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
class WaveCodeSbAtomLoad;
class WaveCodeSbAtomSave;
class WaveCodePool;
class WaveCodeActivation;
class WaveCodeResAdd;



class WaveCode {
public:
    class NpyFileInfo {
    public:
        NpyFileInfo();
    public:
        kcc_int64                   m_FileDramOffset    = -1;
        bool                        m_Dirty             = false;
        ARBPRECTYPE                 m_SimTypeId         = INVALID_ARBPRECTYPE;
        std::array<kcc_int32, 4>    m_RefFileShape;
    };
public:

    using OneInstrStream = FILE*;
    struct InstrStreams {
        OneInstrStream m_StreamProcInstrStream;
        OneInstrStream m_PeArrayInstrStream;
        OneInstrStream m_PoolEngInstrStream;
        OneInstrStream m_ActEngInstrStream;
        OneInstrStream m_DmaInstrStream;
    };
public:
    //----------------------------------------------------------------
    WaveCode(const nets::Network* network, const arch::Arch& arch);

    ~WaveCode();

    void generate(const InstrStreams& instrStreams, bool parallelStreams);

    // Instructions that execute on one engine only: POOL, MATMUL, LDWEIGHTS, etc.
    template<typename INSTR>
    void writeInstruction(const INSTR& instruction);

    // multi-engine instructions: WAIT_EVENT, SET_EVENT, CLEAR_EVENT, WRITE
    template<typename INSTR>
    void writeInstruction(const INSTR& instruction, EngineId engId);


    kcc_int64 gCurrentDramAddress(kcc_int64 sizeInBytes);
    kcc_int64 getDramForNpyFile(const std::string& fileName);
    void recordDramForNpyFile(const std::string& fileName, const NpyFileInfo& npyFileInfo);
    void markDramDirty(const std::string& fileName);

    kcc_uint64 calculateEventAddress(EngineId engId, EventId eventId) const;

    bool qParallelStreams() const {
        return m_ParallelStreams;
    }

private:
    WaveCode() = delete;
    WaveCode(const WaveCode&) = delete;

private:
    WaveCodeWaveOp& getCodeGen(const wave::WaveOp* waveOp);
    void saveAllNpyFiles();

    void checkForNoSync(const TPB_CMD_SYNC&) const;

private:
    const nets::Network*                m_Network;
    const arch::Arch&                   m_Arch;

    const InstrStreams*                 m_InstrStreams;
    std::unique_ptr<WaveCodeMatMul>     m_CodeMatMul;
    std::unique_ptr<WaveCodeSbAtomLoad> m_CodeSbAtomLoad;
    std::unique_ptr<WaveCodeSbAtomSave> m_CodeSbAtomSave;
    std::unique_ptr<WaveCodePool>       m_CodePool;
    std::unique_ptr<WaveCodeActivation> m_CodeActivation;
    std::unique_ptr<WaveCodeResAdd>     m_CodeResAdd;

    kcc_int64                           m_CurrentDramAddress;
    std::map<std::string, NpyFileInfo>  m_NpyFile2DramAddress;
    bool                                m_ParallelStreams = false;
};

}}

#endif // KCC_WAVECODE_WAVECODE_H

