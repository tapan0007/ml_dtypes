#pragma once

#ifndef KCC_WAVECODE_WAVECODE_H
#define KCC_WAVECODE_WAVECODE_H

#include <array>
#include <map>
#include <string>
#include <cstdio>
#include <memory>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "events/inc/events.hpp"


struct TONGA_ISA_TPB_INST_EVENTS;

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
class WaveCodeBarrier;
class WaveCodeNop;



class WaveCode {
public:
    class NpyFileInfo {
    public:
        NpyFileInfo();
    public:
        kcc_int64                   m_FileDramOffset    = -1;
        bool                        m_Dirty             = false;
        TONGA_ISA_TPB_DTYPE         m_SimTypeId         = TONGA_ISA_TPB_DTYPE_INVALID;
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
    WaveCode(nets::Network* network, const arch::Arch& arch);

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

    kcc_uint64 calculateEventAddress(EngineId engId, events::EventId eventId) const;

    bool qParallelStreams() const {
        return m_ParallelStreams;
    }

private:
    WaveCode() = delete;
    WaveCode(const WaveCode&) = delete;

private:
    WaveCodeWaveOp& getCodeGen(const wave::WaveOp* waveOp);
    void saveAllNpyFiles();

    void checkForNoSync(const TONGA_ISA_TPB_INST_EVENTS&) const;

private:
    nets::Network*                m_Network;
    const arch::Arch&                   m_Arch;

    kcc_int32                           m_StreamProcPc = 0;
    kcc_int32                           m_PeArrayPc = 0;
    kcc_int32                           m_PoolEngPc = 0;
    kcc_int32                           m_ActEngPc = 0;
    kcc_int32                           m_DmaPc = 0;

    const InstrStreams*                 m_InstrStreams;
    std::unique_ptr<WaveCodeMatMul>     m_CodeMatMul;
    std::unique_ptr<WaveCodeSbAtomLoad> m_CodeSbAtomLoad;
    std::unique_ptr<WaveCodeSbAtomSave> m_CodeSbAtomSave;
    std::unique_ptr<WaveCodePool>       m_CodePool;
    std::unique_ptr<WaveCodeActivation> m_CodeActivation;
    std::unique_ptr<WaveCodeResAdd>     m_CodeResAdd;
    std::unique_ptr<WaveCodeBarrier>    m_CodeBarrier;
    std::unique_ptr<WaveCodeNop>        m_CodeNop;

    kcc_int64                           m_CurrentDramAddress;
    std::map<std::string, NpyFileInfo>  m_NpyFile2DramAddress;
    bool                                m_ParallelStreams = false;
};

}}

#endif // KCC_WAVECODE_WAVECODE_H

