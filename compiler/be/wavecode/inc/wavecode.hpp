#pragma once

#ifndef KCC_WAVECODE_WAVECODE_H
#define KCC_WAVECODE_WAVECODE_H

#include <array>
#include <map>
#include <string>
#include <cstdio>
#include <memory>
#include <limits>
#include <type_traits>
#include <typeinfo>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "events/inc/events.hpp"

#include "compisa/inc/compisamatmul.hpp"
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
class WaveCodeReciprocal;
class WaveCodeActivation;
class WaveCodeClipByValue;
class WaveCodeTensorTensor;
class WaveCodeTensorScalar;
class WaveCodeBarrier;
class WaveCodeNop;
class WaveCodeTensorTensor;



class WaveCode {
public:
    class NpyFileInfo {
    public:
        NpyFileInfo();
    public:
        kcc_int64                   m_FileDramOffset    = -1;
        bool                        m_Dirty             = false;
        TONGA_ISA_TPB_DTYPE         m_SimTypeId         = TONGA_ISA_TPB_DTYPE_INVALID;
        utils::TensorParams::ShapeType    m_RefFileShape;
    };
public:

    using OneInstrStream = FILE*;
    struct InstrStreams {
        struct OneEngInfo {
            std::string     m_BinFile;
            OneInstrStream  m_InstrStream = nullptr;
            kcc_int32       m_Pc = 0;
        };
        ~InstrStreams();
        void closeAll();

        OneEngInfo      m_StreamProc;
        OneEngInfo      m_PeArray;
        OneEngInfo      m_PoolEng;
        OneEngInfo      m_ActEng;
        OneEngInfo      m_Angel;
    };
public:
    //----------------------------------------------------------------
    WaveCode(nets::Network& network, const arch::Arch& arch, bool useSem);

    ~WaveCode();

    void generate(InstrStreams& instrStreams, bool parallelStreams);
    void DetermineEngines();

    // Instructions that execute on one engine only: POOL, MATMUL, LDWEIGHTS, etc.
    template<typename INSTR>
    void writeInstruction(const INSTR& instruction) const;

    // multi-engine instructions: WAIT_EVENT, SET_EVENT, CLEAR_EVENT, WRITE
    template<typename INSTR>
    void writeInstruction(const INSTR& instruction, EngineId engId) const;


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

    bool gFirstInputDMA_PeArray() const {
        return m_FirstInputDMA_PeArray;
    }
    void rFirstInputDMA_PeArray(bool val) {
        m_FirstInputDMA_PeArray = val;
    }

    bool gFirstInputDMA_ActEng() const {
        return m_FirstInputDMA_ActEng;
    }
    void rFirstInputDMA_ActEng(bool val) {
        m_FirstInputDMA_ActEng = val;
    }

public:
    void writeWaitOrWaitClearInstr(const wave::WaveEdge* edge, EngineId engineId);
    void writeWaitOrWaitClearInstr(events::EventId evntId,
                    events::EventWaitMode waitEventMode,
                    EngineId engineId, const char* const dbgTxt);

public:
    template <int N>
    static void saveName(uint8_t (&res)[N], const char* name)
    {
        for (int i = 0; i < N; ++i) {
            res[i] = '\0';
        }
        strncpy(reinterpret_cast<char*>(&res[0]), name, N - 1);
        res[N-1] = 0;
    }

    template <typename INSTR>
    static void SaveName(INSTR& instr, const char* name)
    {
        saveName(instr.reserved, name);
    }

    static void SaveName(compisa::MatMulInstr& instr, const char* name);


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

    InstrStreams*                       m_InstrStreams;

    std::unique_ptr<WaveCodeMatMul>     m_CodeMatMul;
    std::unique_ptr<WaveCodeSbAtomLoad> m_CodeSbAtomLoad;
    std::unique_ptr<WaveCodeSbAtomSave> m_CodeSbAtomSave;
    std::unique_ptr<WaveCodePool>       m_CodePool;
    std::unique_ptr<WaveCodeReciprocal> m_CodeReciprocal;
    std::unique_ptr<WaveCodeActivation> m_CodeActivation;
    std::unique_ptr<WaveCodeClipByValue> m_CodeClipByValue;
    std::unique_ptr<WaveCodeBarrier>    m_CodeBarrier;
    std::unique_ptr<WaveCodeNop>        m_CodeNop;
    std::unique_ptr<WaveCodeTensorTensor> m_CodeTensorTensor;
    std::unique_ptr<WaveCodeTensorScalar> m_CodeTensorScalar;

    kcc_int64                           m_CurrentDramAddress;
    std::map<std::string, NpyFileInfo>  m_NpyFile2DramAddress;
    bool                                m_ParallelStreams = false;
    BinFileType                         m_BinFileType = BinFileType::SimAngel;
    bool                                m_FirstInputDMA_ActEng  = true;
    bool                                m_FirstInputDMA_PeArray = true;

    kelf::DmaDescription                m_DmaDescription;
};





/* **************************************************************** */
template <typename T, bool IsEnum>
struct UnderlyingType;

template <typename T>
struct UnderlyingType<T, true> {
    using Type = typename std::underlying_type<T>::type;
};

template <typename T>
struct UnderlyingType<T, false> {
    using Type = T;
};

/* **************************************************************** */
template<typename TypeTo, typename TypeFrom>
void
AssignWithSizeCheckTempl(TypeTo& to, const TypeFrom from, const char* fileName, int lineNum)
{
    using UnderlyingTypeFrom = typename UnderlyingType<TypeFrom, std::is_enum<TypeFrom>::value>::Type;
    using UnderlyingTypeTo   = typename UnderlyingType<TypeTo,   std::is_enum<TypeTo>::value>::Type;

    const UnderlyingTypeFrom from1(static_cast<UnderlyingTypeFrom>(from));
    const UnderlyingTypeTo   to1(from1);
    const UnderlyingTypeFrom from2(to1);

    Assert( ((to1 > 0) == (from1 > 0))  && ((from1 > 0) == (from2 > 0)) 
            && (from1 == from2) && (static_cast<long long>(from1) == static_cast<long long>(to1)),
            "File='", fileName, "', line=", lineNum,
            ", TypeTo=", typeid(to).name(), ", TypeFrom=", typeid(from).name(), ", ValueFrom=", from );

    to = from;
}

#define AssignWithSizeCheck(To, From) AssignWithSizeCheckTempl(To, From, __FILE__, __LINE__)



}}

#endif // KCC_WAVECODE_WAVECODE_H

