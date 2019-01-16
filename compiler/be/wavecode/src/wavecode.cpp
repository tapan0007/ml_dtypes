#include <map>
#include <array>

#include "tonga/address_map.h"
#include "aws_tonga_isa_tpb_common.h"

#include "utils/inc/debug.hpp"

#include "compisa/inc/compisaset.hpp"
#include "compisa/inc/compisawait.hpp"
#include "compisa/inc/compisaclear.hpp"
#include "compisa/inc/compisanop.hpp"
#include "compisa/inc/compisamatmul.hpp"
#include "compisa/inc/compisasimrdnpy.hpp"





#include "utils/inc/asserter.hpp"
#include "utils/inc/debug.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "nets/inc/network.hpp"


#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/tpbcopywaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/reciprocalwaveop.hpp"
#include "wave/inc/regloadwaveop.hpp"
#include "wave/inc/regstorewaveop.hpp"
#include "wave/inc/regshufflewaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/clipbyvaluewaveop.hpp"
#include "wave/inc/tensortensorwaveop.hpp"
#include "wave/inc/tensorscalarwaveop.hpp"
#include "wave/inc/nopwaveop.hpp"
#include "wave/inc/waveedge.hpp"

//#include "wavecode/inc/wavecodewaveop.hpp"
#include "wavecode/inc/wavecodesbatomload_sim.hpp"
#include "wavecode/inc/wavecodesbatomload_kelf.hpp"
#include "wavecode/inc/wavecodesbatomsave_sim.hpp"
#include "wavecode/inc/wavecodesbatomsave_kelf.hpp"
#include "wavecode/inc/wavecodetpbcopy.hpp"
#include "wavecode/inc/wavecodematmul.hpp"
#include "wavecode/inc/wavecodepool.hpp"
#include "wavecode/inc/wavecodereciprocal.hpp"
#include "wavecode/inc/wavecoderegload.hpp"
#include "wavecode/inc/wavecoderegstore.hpp"
#include "wavecode/inc/wavecoderegshuffle.hpp"
#include "wavecode/inc/wavecodeactivation.hpp"
#include "wavecode/inc/wavecodeclipbyvalue.hpp"
#include "wavecode/inc/wavecodetensortensor.hpp"
#include "wavecode/inc/wavecodetensorscalar.hpp"
#include "wavecode/inc/wavecodenop.hpp"

#include "wavecode/inc/wavecode.hpp"

namespace kcc {
namespace wavecode {

WaveCode::WaveCode(nets::Network& network, const arch::Arch& arch, bool useSem)
    : m_Network(network)
    , m_Arch(arch)
    , m_DmaDescription(network, useSem)
{
    m_CodeMatMul            = std::make_unique<WaveCodeMatMul>(*this);
    m_CodeSbAtomLoadSim     = std::make_unique<WaveCodeSbAtomLoadSim>(*this);
    m_CodeSbAtomLoadKelf    = std::make_unique<WaveCodeSbAtomLoadKelf>(*this);
    m_CodeSbAtomSaveSim     = std::make_unique<WaveCodeSbAtomSaveSim>(*this);
    m_CodeSbAtomSaveKelf    = std::make_unique<WaveCodeSbAtomSaveKelf>(*this);
    m_CodeTpbCopy           = std::make_unique<WaveCodeTpbCopy>(*this);
    m_CodePool              = std::make_unique<WaveCodePool>(*this);
    m_CodeReciprocal        = std::make_unique<WaveCodeReciprocal>(*this);
    m_CodeRegLoad           = std::make_unique<WaveCodeRegLoad>(*this);
    m_CodeRegStore          = std::make_unique<WaveCodeRegStore>(*this);
    m_CodeRegShuffle        = std::make_unique<WaveCodeRegShuffle>(*this);
    m_CodeActivation        = std::make_unique<WaveCodeActivation>(*this);
    m_CodeClipByValue       = std::make_unique<WaveCodeClipByValue>(*this);
    m_CodeNop               = std::make_unique<WaveCodeNop>(*this);
    m_CodeTensorTensor      = std::make_unique<WaveCodeTensorTensor>(*this);
    m_CodeTensorScalar      = std::make_unique<WaveCodeTensorScalar>(*this);

    m_CodeTpbCopy->rWaveCodeSbAtomLoadKelf(m_CodeSbAtomLoadKelf.get());
    m_CodeSbAtomLoadKelf->rWaveCodeTpbCopy(m_CodeTpbCopy.get());

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
        // never put DmaTrigger on PeArray
    } };

    for (auto waveop : m_Network.gWaveOps()) {
        if (waveop->gEngineId() != EngineId::None) {
            continue;
        }
        Assert(waveop->qSbAtomWaveOp() || waveop->qTpbCopyWaveOp(),
                "Waveop without engine must be SbAtom or TpbCopy: ", waveop->gName());

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

        EngineId engId = EngineId::None;
        if (chosenPrevEdge) {
            chosenPrevEdge->rChosenForSuccSbAtom(true);
            engId = chosenPrevEdge->gFromOp()->gEngineId();
        } else {
            engId = EngineId::Pooling;
        }

        if (const auto sbWop = dynamic_cast<wave::SbAtomWaveOp*>(waveop)) {
            if (sbWop->gPrevWaveEdges().size() == 0) {
                // initial loads
                if (const auto loadWop = dynamic_cast<wave::SbAtomLoadWaveOp*>(sbWop)) {
                    loadWop->rEngineId(EngineId::Pooling);
                } else {
                    Assert(false, "Waveop without input edges (", sbWop->gName(), ") is not Load");
                    sbWop->rEngineId(EngineId::Pooling);
                }

                continue;
            } else {
                sbWop->rEngineId(engId);
            }
        } else if (const auto copyWop = dynamic_cast<wave::TpbCopyWaveOp*>(waveop)) {
            copyWop->rEngineId(engId);
        } else {
            Assert(false, "Waveop without engine must be SbAtom or TpbCopy: ", waveop->gName());
        }
    }
}

//----------------------------------------------------------------
void
WaveCode::DetermineEngines()
{
    if (qGenerateKelf()) {
        // First determine edges based on preceeding waveops
        determinePrecSbEdges();
    } else {
        for (auto waveop : m_Network.gWaveOps()) {
            if (auto sbWaveop = dynamic_cast<wave::SbAtomWaveOp*>(waveop)) {
                sbWaveop->rEngineId(EngineId::AngelEng);
            }
        }
    }
}

//----------------------------------------------------------------
void
WaveCode::generate(InstrStreams& instrStreams, bool parallelStreams)
{
    m_ParallelStreams = parallelStreams;

    m_InstrStreams = &instrStreams;


    // Process waveops
    for (auto waveOp : m_Network.gWaveOps()) {
        auto& codeGen = getCodeGen(waveOp);
        codeGen.generate(waveOp);
    }

    //**********************************************************************************
    if (! qGenerateKelf()) {
        saveAllNpyFiles();
    }
    if (qBinFileRuntimeKelf()) {
        kelf::DmaDescription& dmaDescr(gDmaDescription());
        dmaDescr.writeDmaDescriptors(m_InstrStreams->m_PeArray.m_BinFile.c_str(), EngineId::PeArray);
        dmaDescr.writeDmaDescriptors(m_InstrStreams->m_ActEng.m_BinFile.c_str(), EngineId::Activation);
        dmaDescr.writeDmaDescriptors(m_InstrStreams->m_PoolEng.m_BinFile.c_str(), EngineId::Pooling);
        dmaDescr.writeInOutDescriptors();
        dmaDescr.writeDefinitions(m_InstrStreams->m_PeArray.m_BinFile.c_str(),
            m_InstrStreams->m_ActEng.m_BinFile.c_str(), m_InstrStreams->m_PoolEng.m_BinFile.c_str());
    }
}

WaveCodeWaveOp&
WaveCode::getCodeGen(const wave::WaveOp* waveOp)
{
    if (dynamic_cast<const wave::MatMulWaveOp*>(waveOp)) {
        return *m_CodeMatMul;
    } else if (dynamic_cast<const wave::SbAtomLoadWaveOp*>(waveOp)) {
        if (qGenerateKelf()) {
            return *m_CodeSbAtomLoadKelf;
        } else {
            return *m_CodeSbAtomLoadSim;
        }
    } else if (dynamic_cast<const wave::SbAtomSaveWaveOp*>(waveOp)) {
        if (qGenerateKelf()) {
            return *m_CodeSbAtomSaveKelf;
        } else {
            return *m_CodeSbAtomSaveSim;
        }
    } else if (dynamic_cast<const wave::TpbCopyWaveOp*>(waveOp)) {
        return *m_CodeTpbCopy;
    } else if (dynamic_cast<const wave::PoolWaveOp*>(waveOp)) {
        return *m_CodePool;
    } else if (dynamic_cast<const wave::ReciprocalWaveOp*>(waveOp)) {
        return *m_CodeReciprocal;
    } else if (dynamic_cast<const wave::RegLoadWaveOp*>(waveOp)) {
        return *m_CodeRegLoad;
    } else if (dynamic_cast<const wave::RegStoreWaveOp*>(waveOp)) {
        return *m_CodeRegStore;
    } else if (dynamic_cast<const wave::RegShuffleWaveOp*>(waveOp)) {
        return *m_CodeRegShuffle;
    } else if (dynamic_cast<const wave::ActivationWaveOp*>(waveOp)) {
        return *m_CodeActivation;
    } else if (dynamic_cast<const wave::ClipByValueWaveOp*>(waveOp)) {
        return *m_CodeClipByValue;
    } else if (dynamic_cast<const wave::NopWaveOp*>(waveOp)) {
        return *m_CodeNop;
    } else if (dynamic_cast<const wave::TensorTensorWaveOp*>(waveOp)) {
        return *m_CodeTensorTensor;
    } else if (dynamic_cast<const wave::TensorScalarWaveOp*>(waveOp)) {
        return *m_CodeTensorScalar;
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
        compisa::SimRdNpyInstr simDramToNpyInstr;
        AssignWithSizeCheck(simDramToNpyInstr.inst_events.wait_event_idx, 0);
        AssignWithSizeCheck(simDramToNpyInstr.inst_events.wait_event_mode, eventWaitMode2Isa(events::EventWaitMode::DontWait));
        AssignWithSizeCheck(simDramToNpyInstr.inst_events.set_event_idx, 0);
        AssignWithSizeCheck(simDramToNpyInstr.inst_events.set_event_mode, eventSetMode2Isa(events::EventSetMode::DontSet));

        strcpy(simDramToNpyInstr.dst_fname, (*it).first.c_str());
        const NpyFileInfo& npyFileInfo((*it).second);
        AssignWithSizeCheck(simDramToNpyInstr.src_addr, npyFileInfo.m_FileDramOffset);
        AssignWithSizeCheck(simDramToNpyInstr.dst_ndims, npyFileInfo.m_RefFileFormat.size());
        for (int i = 0; i < simDramToNpyInstr.dst_ndims; ++i) {
            AssignWithSizeCheck(simDramToNpyInstr.dst_dims[i], npyFileInfo.m_RefFileShape[i]);
        }
        AssignWithSizeCheck(simDramToNpyInstr.dtype, npyFileInfo.m_SimTypeId);
        this->writeInstruction(simDramToNpyInstr);
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

    case EngineId::AngelEng:
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
    Assert(events::qEventSetModeValid(inst_events.set_event_mode), "Invalid set event mode", inst_events.set_event_mode);
    Assert(events::qEventWaitModeValid(inst_events.wait_event_mode), "Invalid wait event mode", inst_events.wait_event_mode);

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
    if (m_StreamProc.m_InstrStream) {
        fclose(m_StreamProc.m_InstrStream);
        m_StreamProc.m_InstrStream = nullptr;
    }
    if (m_PeArray.m_InstrStream) {
        fclose(m_PeArray.m_InstrStream);
        m_PeArray.m_InstrStream = nullptr;
    }
    if (m_PoolEng.m_InstrStream) {
        fclose(m_PoolEng.m_InstrStream);
        m_PoolEng.m_InstrStream = nullptr;
    }
    if (m_ActEng.m_InstrStream) {
        fclose(m_ActEng.m_InstrStream);
        m_ActEng.m_InstrStream = nullptr;
    }
    if (m_Angel.m_InstrStream) {
        fclose(m_Angel.m_InstrStream);
        m_Angel.m_InstrStream = nullptr;
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
        AssignWithSizeCheck(waitInstr.event_idx, evntId);
        AssignWithSizeCheck(waitInstr.wait_event_mode, eventWaitMode2Isa(waitEventMode));

        SaveName(waitInstr, dbgTxt);
        writeInstruction(waitInstr, engineId);
        break;
    }
    case NOP: {
        // New Nop instruction can wait and set (should use for barrier too)
        compisa::NopInstr nopInstr;
        AssignWithSizeCheck(nopInstr.inst_events.wait_event_idx, evntId);
        AssignWithSizeCheck(nopInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(waitEventMode));
        AssignWithSizeCheck(nopInstr.inst_events.set_event_idx, 0);
        AssignWithSizeCheck(nopInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

        SaveName(nopInstr, dbgTxt);
        writeInstruction(nopInstr, engineId);
        break;
    }
    case WAIT_PLUS_CLEAR: {
        {
        // old style: Wait(wait-only); Clear
            compisa::WaitInstr waitInstr;
            AssignWithSizeCheck(waitInstr.event_idx, evntId);
            AssignWithSizeCheck(waitInstr.wait_event_mode, eventWaitMode2Isa(events::EventWaitMode::WaitOnly));

            SaveName(waitInstr, dbgTxt);
            writeInstruction(waitInstr, engineId);
        }

        if (waitEventMode == events::EventWaitMode::WaitThenClear) {
            compisa::ClearInstr clearInstr;
            AssignWithSizeCheck(clearInstr.event_idx, evntId);

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
    oss << waveEdge->gFromOp()->gOrder() << "->" << waveEdge->gToOp()->gOrder() << ": " << waveop->gName();
    writeWaitOrWaitClearInstr(waveEdge->gEventId(), waitEventMode,
        engineId, oss.str().c_str());
}


//======================================================================
void
WaveCode::SaveName(compisa::MatMulInstr& instr, const char* name)
{
    saveName(instr.reserved, name);
}


}}

