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
#include "kelf/inc/kelfdmadescription.hpp"


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
        ~InstrStreams();
        void closeAll();

        std::string     m_StreamProcBinFile;
        std::string     m_PeArrayBinFile;
        std::string     m_PoolEngBinFile;
        std::string     m_ActEngBinFile;
        std::string     m_DmaBinFile;
        OneInstrStream  m_StreamProcInstrStream = nullptr;
        OneInstrStream  m_PeArrayInstrStream = nullptr;
        OneInstrStream  m_PoolEngInstrStream = nullptr;
        OneInstrStream  m_ActEngInstrStream = nullptr;
        OneInstrStream  m_DmaInstrStream = nullptr;
    };
public:
    //----------------------------------------------------------------
    WaveCode(nets::Network& network, const arch::Arch& arch);

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

    bool qBinFileSimAngel() const {
        return BinFileType::SimAngel == m_BinFileType;
    }
    bool qBinFileSimKelf() const {
        return BinFileType::SimKelf == m_BinFileType;
    }
    bool qBinFileRuntimeKelf() const {
        return BinFileType::RuntimeKelf == m_BinFileType;
    }
    bool qGenerateKelf() const {
        return qBinFileSimKelf() || qBinFileRuntimeKelf();
    }
    void rBinFileType(BinFileType typ) {
        m_BinFileType = typ;
    }
    kelf::DmaDescription& gDmaDescription() {
        return m_DmaDescription;
    }

private:
    WaveCode() = delete;
    WaveCode(const WaveCode&) = delete;

private:
    WaveCodeWaveOp& getCodeGen(const wave::WaveOp* waveOp);
    void saveAllNpyFiles();

    void checkForNoSync(const TONGA_ISA_TPB_INST_EVENTS&) const;

    void determinePrecSbEdges();

private:
    nets::Network&                      m_Network;
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
    BinFileType                         m_BinFileType = BinFileType::SimAngel;
    
    kelf::DmaDescription                m_DmaDescription;
};

}}

#endif // KCC_WAVECODE_WAVECODE_H

