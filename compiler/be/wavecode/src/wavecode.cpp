#include <map>

#include "address_map.h"
#include "aws_tonga_isa_tpb_common.h"

#include "utils/inc/debug.hpp"

#include "compisa/inc/compisaset.hpp"
#include "compisa/inc/compisawait.hpp"
#include "compisa/inc/compisaclear.hpp"
#include "compisa/inc/compisanop.hpp"
#include "compisa/inc/compisamatmul.hpp"
#include "compisa/inc/compisasimrdnpy.hpp"





#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "nets/inc/network.hpp"


#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/resaddwaveop.hpp"
#include "wave/inc/barrierwaveop.hpp"
#include "wave/inc/nopwaveop.hpp"
#include "wave/inc/waveedge.hpp"

//#include "wavecode/inc/wavecodewaveop.hpp"
#include "wavecode/inc/wavecodesbatomload.hpp"
#include "wavecode/inc/wavecodesbatomsave.hpp"
#include "wavecode/inc/wavecodematmul.hpp"
#include "wavecode/inc/wavecodepool.hpp"
#include "wavecode/inc/wavecodeactivation.hpp"
#include "wavecode/inc/wavecoderesadd.hpp"
#include "wavecode/inc/wavecodebarrier.hpp"
#include "wavecode/inc/wavecodenop.hpp"

#include "wavecode/inc/wavecode.hpp"

namespace kcc {
namespace wavecode {

WaveCode::WaveCode(nets::Network& network, const arch::Arch& arch)
    : m_Network(network)
    , m_Arch(arch)
    , m_DmaDescription(network)
{
    m_CodeMatMul            = std::make_unique<WaveCodeMatMul>(*this);
    m_CodeSbAtomLoad        = std::make_unique<WaveCodeSbAtomLoad>(*this);
    m_CodeSbAtomSave        = std::make_unique<WaveCodeSbAtomSave>(*this);
    m_CodePool              = std::make_unique<WaveCodePool>(*this);
    m_CodeActivation        = std::make_unique<WaveCodeActivation>(*this);
    m_CodeResAdd            = std::make_unique<WaveCodeResAdd>(*this);
    m_CodeBarrier           = std::make_unique<WaveCodeBarrier>(*this);
    m_CodeNop               = std::make_unique<WaveCodeNop>(*this);

    m_CurrentDramAddress    = P_0_DRAM_0_BASE;
}

//----------------------------------------------------------------
WaveCode::~WaveCode() = default;


//----------------------------------------------------------------
void
WaveCode::determinePrecSbEdges()
{
    const std::array<EngineId, 3> engineIds = { {
        EngineId::Pooling,
        EngineId::Activation,
        EngineId::PeArray
    } };

    for (auto waveop : m_Network.gWaveOps()) {
        if (!waveop->qSbAtomWaveOp()) {
            continue;
        }

        const auto sbWop = dynamic_cast<wave::SbAtomWaveOp*>(waveop);
        if (waveop->gPrevWaveEdges().size() == 0) {
            // initial loads
            if (const auto loadWop = dynamic_cast<wave::SbAtomLoadWaveOp*>(sbWop)) {
                if (loadWop->qContainWeights()) {
                    loadWop->rEngineId(EngineId::PeArray);
                } else {
                    loadWop->rEngineId(EngineId::Pooling);
                }
            } else {
                loadWop->rEngineId(EngineId::Pooling);
            }

            continue;
        }

        wave::WaveEdge* chosenPrevEdge = nullptr;
        for (auto engId : engineIds) {
            for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
                if (prevWaveEdge->gFromOp()->gEngineId() == engId) {
                    chosenPrevEdge = prevWaveEdge;
                    break;
                }
            }
            if (chosenPrevEdge) {
                break;
            }
        }
        // Chosen edge exist if on TPB engine.
        // If this is SbAtomSave --> SbAtomLoad, there will not be a chosen edge.
        chosenPrevEdge->rChosenForSuccSbAtom(true);
        sbWop->rEngineId(chosenPrevEdge->gFromOp()->gEngineId());
    }
}

//----------------------------------------------------------------
void
WaveCode::DetermineEngines()
{
    if (qGenerateKelf()) {
        determinePrecSbEdges();
    } else {
        for (auto waveop : m_Network.gWaveOps()) {
            if (auto sbWaveop = dynamic_cast<wave::SbAtomWaveOp*>(waveop)) {
                sbWaveop->rEngineId(EngineId::DmaEng);
            }
        }
    }
}

//----------------------------------------------------------------
void
WaveCode::generate(const InstrStreams& instrStreams, bool parallelStreams)
{
    m_ParallelStreams = parallelStreams;

    m_InstrStreams = &instrStreams;

    /***********************************************************************************
     * Pooling angine waits for inference and signals other engines (PE,Act) immediately.
     * Other engines wait on signal from pool *before* any input DMA initiation is done
     * on each respective engine.
     * If any of the other engines never initiates DMA, 
     ***********************************************************************************/
    if (qBinFileRuntimeKelf()) {
    // Pool wait for start inference at beginning
        writeWaitOrWaitClearInstr(events::EventId_StartInference(), events::EventWaitMode::WaitThenClear,
                        EngineId::Pooling, "Waiting on pooling engine to start inference");
        { // Pool sets event for PeArray to read inputs
            compisa::SetInstr setInstr;
            setInstr.event_idx  = events::EventId_BeforeInputRead_PeArray();
            writeInstruction(setInstr, EngineId::Pooling);
        }
        { // Pool sets event for Act to read inputs
            compisa::SetInstr setInstr;
            setInstr.event_idx  = events::EventId_BeforeInputRead_ActEng();
            writeInstruction(setInstr, EngineId::Pooling);
        }
    }

    // Process waveops
    for (auto waveOp : m_Network.gWaveOps()) {
        auto& codeGen = getCodeGen(waveOp);
        codeGen.generate(waveOp);
    }

    if (qBinFileRuntimeKelf()) {
        // If PeArr never waited on input, wait at end for event from pooling
        if (gFirstInputDMA_PeArray()) {
            rFirstInputDMA_PeArray(false);
            writeWaitOrWaitClearInstr(events::EventId_BeforeInputRead_PeArray(),
                events::EventWaitMode::WaitThenClear,
                EngineId::PeArray,
                "At end of PeArray execution: wait for first input event from Pooling");
        }
        // If Act never waited on input, wait at end for event from pooling
        if (gFirstInputDMA_ActEng()) {
            rFirstInputDMA_ActEng(false);
            writeWaitOrWaitClearInstr(events::EventId_BeforeInputRead_ActEng(),
                events::EventWaitMode::WaitThenClear,
                EngineId::Activation,
                "At end of Act engine execution: wait for first input event from Pooling");
        }
    }

    //**********************************************************************************
    if (! qGenerateKelf()) {
        saveAllNpyFiles();
    }
    if (qBinFileRuntimeKelf()) {
        kelf::DmaDescription& dmaDescr(gDmaDescription());
        dmaDescr.writeDmaDescriptors(m_InstrStreams->m_PeArrayBinFile.c_str(), EngineId::PeArray);
        dmaDescr.writeDmaDescriptors(m_InstrStreams->m_ActEngBinFile.c_str(), EngineId::Activation);
        dmaDescr.writeDmaDescriptors(m_InstrStreams->m_PoolEngBinFile.c_str(), EngineId::Pooling);
        dmaDescr.writeInOutDescriptors();
        dmaDescr.writeDefinitions();
    }
}

WaveCodeWaveOp&
WaveCode::getCodeGen(const wave::WaveOp* waveOp)
{
    if (dynamic_cast<const wave::MatMulWaveOp*>(waveOp)) {
        return *m_CodeMatMul;
    } else if (dynamic_cast<const wave::SbAtomLoadWaveOp*>(waveOp)) {
        return *m_CodeSbAtomLoad;
    } else if (dynamic_cast<const wave::SbAtomSaveWaveOp*>(waveOp)) {
        return *m_CodeSbAtomSave;
    } else if (dynamic_cast<const wave::PoolWaveOp*>(waveOp)) {
        return *m_CodePool;
    } else if (dynamic_cast<const wave::ActivationWaveOp*>(waveOp)) {
        return *m_CodeActivation;
    } else if (dynamic_cast<const wave::ResAddWaveOp*>(waveOp)) {
        return *m_CodeResAdd;
    } else if (dynamic_cast<const wave::BarrierWaveOp*>(waveOp)) {
        return *m_CodeBarrier;
    } else if (dynamic_cast<const wave::NopWaveOp*>(waveOp)) {
        return *m_CodeNop;
    } else {
        assert(false && "WaveCode: Unsupported WaveOp");
    }
    return *m_CodeMatMul;
}

kcc_int64
WaveCode::getDramForNpyFile(const std::string& fileName)
{
    const auto it = m_NpyFile2DramAddress.find(fileName);
    if (m_NpyFile2DramAddress.end() != it) {
        return (*it).second.m_FileDramOffset;
    } else {
        return -1;
    }
}


void
WaveCode::recordDramForNpyFile(const std::string& fileName, const NpyFileInfo& npyFileInfo)
{
    m_NpyFile2DramAddress[fileName] = npyFileInfo;
}


kcc_int64
WaveCode::gCurrentDramAddress(kcc_int64 sizeInBytes)
{
    const kcc_int64 currAddress = m_CurrentDramAddress;
    m_CurrentDramAddress += sizeInBytes;
    return currAddress;
}








/***********************************************************************
 * Misc
***********************************************************************/
void
WaveCode::markDramDirty(const std::string& fileName)
{
    const auto it = m_NpyFile2DramAddress.find(fileName);
    assert(m_NpyFile2DramAddress.end() != it && "Setting dirty flag on non-existant file");
    (*it).second.m_Dirty = true;
}

void
WaveCode::saveAllNpyFiles ()
{

    const auto itE = m_NpyFile2DramAddress.cend();
    auto it = m_NpyFile2DramAddress.cbegin();
    for (; itE != it; ++it) {
        if (! (*it).second.m_Dirty) {
            continue;
        }
        compisa::SimRdNpyInstr dramToNpyInstr;
        dramToNpyInstr.inst_events.wait_event_idx    = 0;
        dramToNpyInstr.inst_events.wait_event_mode   = eventWaitMode2Isa(events::EventWaitMode::DontWait);
        dramToNpyInstr.inst_events.set_event_idx     = 0;
        dramToNpyInstr.inst_events.set_event_mode    = eventSetMode2Isa(events::EventSetMode::DontSet);

        strcpy(dramToNpyInstr.dst_fname, (*it).first.c_str());
        const NpyFileInfo& npyFileInfo((*it).second);
        dramToNpyInstr.src_addr         = npyFileInfo.m_FileDramOffset;
        dramToNpyInstr.dst_ndims        = 4;
        for (int i = 0; i < dramToNpyInstr.dst_ndims; ++i) {
            dramToNpyInstr.dst_dims[i]  = npyFileInfo.m_RefFileShape[i];
        }
        dramToNpyInstr.dtype            = npyFileInfo.m_SimTypeId;

        this->writeInstruction(dramToNpyInstr);
    }
}

kcc_uint64
WaveCode::calculateEventAddress(EngineId engId, events::EventId eventId) const
{
    const arch::Arch& arch(arch::Arch::gArch());

    switch (engId) {
    case EngineId::Pooling:
    case EngineId::PeArray:
    case EngineId::Activation:
        return arch.gTpbEventBase() + eventId; // 1 byte per eventId
        break;

    case EngineId::StreamProc:
        return arch.gSpEventBase()  + eventId;

    case EngineId::DmaEng:
        return arch.gTpbEventBase() + eventId;

    case EngineId::AnyEng:
        Assert(false, "AnyEng not allowed, engine id ", static_cast<int>(engId));

    case EngineId::None:
        Assert(false, "Bad engine id ", static_cast<int>(engId));
    }
    Assert(false, "Bad engine id ", static_cast<int>(engId));
    return 0;
}


void
WaveCode::checkForNoSync(const TONGA_ISA_TPB_INST_EVENTS& inst_events) const
{
    Assert(events::qEventSetModeValid(inst_events.set_event_mode), "Invalid set event mode");
    Assert(events::qEventWaitModeValid(inst_events.wait_event_mode), "Invalid wait event mode");

    if (qParallelStreams()) {
        return;
    }

    Assert(TONGA_ISA_TPB_MODE_SET_NONE == inst_events.set_event_mode,
        "Code generation: set event mode should be NONE in serial execution");

    Assert(TONGA_ISA_TPB_MODE_WAIT_NONE == inst_events.wait_event_mode,
        "Code generation: wait event mode should be NONE in serial execution");
}





WaveCode::NpyFileInfo::NpyFileInfo()
{
    m_FileDramOffset    = -1;
    m_Dirty             = false;
    m_SimTypeId         = TONGA_ISA_TPB_DTYPE_INVALID;
}



WaveCode::InstrStreams::~InstrStreams()
{
    closeAll();
}

void
WaveCode::InstrStreams::closeAll()
{
    if (m_StreamProcInstrStream) {
        fclose(m_StreamProcInstrStream);
        m_StreamProcInstrStream = nullptr;
    }
    if (m_PeArrayInstrStream) {
        fclose(m_PeArrayInstrStream);
        m_PeArrayInstrStream = nullptr;
    }
    if (m_PoolEngInstrStream) {
        fclose(m_PoolEngInstrStream);
        m_PoolEngInstrStream = nullptr;
    }
    if (m_ActEngInstrStream) {
        fclose(m_ActEngInstrStream);
        m_ActEngInstrStream = nullptr;
    }
    if (m_DmaInstrStream) {
        fclose(m_DmaInstrStream);
        m_DmaInstrStream = nullptr;
    }
}

//======================================================================
void
WaveCode::writeWaitOrWaitClearInstr(
                    events::EventId evntId, events::EventWaitMode waitEventMode,
                    EngineId engineId, const char* const dbgTxt)
{
    Assert(waitEventMode == events::EventWaitMode::WaitThenClear
                || waitEventMode == events::EventWaitMode::WaitOnly,
           "Cannot wait on edge with DontWait mode");

    enum { WAIT_CLEAR_MODE, WAIT_PLUS_CLEAR, NOP };

    //switch (WAIT_PLUS_CLEAR)
    switch (WAIT_CLEAR_MODE)
    {
    case WAIT_CLEAR_MODE: {
        // Not sure whether wait_event_mode works in SIM.
        compisa::WaitInstr waitInstr;
        waitInstr.event_idx         = evntId;
        waitInstr.wait_event_mode   = eventWaitMode2Isa(waitEventMode);

        SaveName(waitInstr, dbgTxt);
        writeInstruction(waitInstr, engineId);
        break;
    }
    case NOP: {
        // New Nop instruction can wait and set (should use for barrier too)
        compisa::NopInstr nopInstr;
        nopInstr.inst_events.wait_event_idx   = evntId;
        nopInstr.inst_events.wait_event_mode  = events::eventWaitMode2Isa(waitEventMode);
        nopInstr.inst_events.set_event_idx    = 0;
        nopInstr.inst_events.set_event_mode   = events::eventSetMode2Isa(events::EventSetMode::DontSet);

        SaveName(nopInstr, dbgTxt);
        writeInstruction(nopInstr, engineId);
        break;
    }
    case WAIT_PLUS_CLEAR: {
        {
        // old style: Wait(wait-only); Clear
            compisa::WaitInstr waitInstr;
            waitInstr.event_idx         = evntId;
            waitInstr.wait_event_mode   = eventWaitMode2Isa(events::EventWaitMode::WaitOnly);

            SaveName(waitInstr, dbgTxt);
            writeInstruction(waitInstr, engineId);
        }

        if (waitEventMode == events::EventWaitMode::WaitThenClear) {
            compisa::ClearInstr clearInstr;
            clearInstr.event_idx  = evntId;

            SaveName(clearInstr, dbgTxt);
            writeInstruction(clearInstr, engineId);
        }
        break;
    }
    default:
        Assert(false, "Unknown waiting method");
        break;
    }
}

//======================================================================
void
WaveCode::writeWaitOrWaitClearInstr(const wave::WaveEdge* waveEdge, EngineId engineId)
{
    events::EventWaitMode waitEventMode = waveEdge->gWaitEventMode();
    Assert(waitEventMode == events::EventWaitMode::WaitThenClear
                || waitEventMode == events::EventWaitMode::WaitOnly,
           "Cannot wait on edge with DontWait mode");

    enum { WAIT_CLEAR_MODE, WAIT_PLUS_CLEAR, NOP };

    //switch (WAIT_PLUS_CLEAR)
    const wave::WaveOp* waveop = NULL;
    switch (WAIT_CLEAR_MODE)
    {
    case WAIT_CLEAR_MODE: {
        // Not sure whether wait_event_mode works in SIM.
        waveop = waveEdge->gToOp();
        break;
    }
    case NOP: {
        waveop = waveEdge->gFromOp();
        break;
    }
    case WAIT_PLUS_CLEAR: {
        waveop = waveEdge->gToOp();
        waitEventMode = events::EventWaitMode::WaitOnly;
        break;
    }
    default:
        Assert(false, "Unknown waiting method", events::eventWaitMode2Isa(waitEventMode));
        break;
    }
    std::ostringstream oss;
    oss << waveop->gOrder() << "-" << waveop->gName();
    writeWaitOrWaitClearInstr(waveEdge->gEventId(), waitEventMode,
        engineId, oss.str().c_str());
}


//======================================================================
void
WaveCode::SaveName(compisa::MatMulInstr& instr, const char* name)
{
    saveName(instr.reserved_2, name);
}

}}

