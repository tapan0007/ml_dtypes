#include <set>



#include "utils/inc/asserter.hpp"
#include "utils/inc/debug.hpp"

#include "compisa/inc/compisawait.hpp"
#include "compisa/inc/compisaldweights.hpp"
#include "compisa/inc/compisamatmul.hpp"

#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodematmul.hpp"

namespace kcc {
namespace wavecode {


//************************************************************************
WaveCodeMatMul::WaveCodeMatMul(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



//************************************************************************
void
WaveCodeMatMul::generate(wave::WaveOp* waveOp)
{
    auto matmulWaveOp = dynamic_cast<wave::MatMulWaveOp*>(waveOp);
    assert(matmulWaveOp);

    kcc_int32 numSyncedPrevWeights;
    kcc_int32 numSyncedPrevIfmaps;
    countSyncedWeithsIfmapPred(matmulWaveOp, numSyncedPrevWeights, numSyncedPrevIfmaps);
    m_SyncIfmapOnLdWeigthsInstr = (numSyncedPrevWeights < 1);

    bool SyncPrevWavesOnMatMulInstr = generateLoadWeights(matmulWaveOp);
    generateMatMul(matmulWaveOp, SyncPrevWavesOnMatMulInstr);
}

//************************************************************************
bool
WaveCodeMatMul::qSyncOnLdWeightsInstr(const wave::WaveEdge* prevEdge) const
{
    return m_SyncIfmapOnLdWeigthsInstr || qLoadWeightsWaitsFor(prevEdge);
}

//************************************************************************
bool
WaveCodeMatMul::qSyncOnMatMulInstr(const wave::WaveEdge* prevEdge) const
{
    return ! qSyncOnLdWeightsInstr(prevEdge);
}

//************************************************************************
void
WaveCodeMatMul::countSyncedWeithsIfmapPred(
        wave::MatMulWaveOp* matmulWaveop,
        kcc_int32& numSyncedPrevWeights,
        kcc_int32& numSyncedPrevIfmaps)
{
    kcc_int32 numWeights = 0;
    kcc_int32 numIfmaps = 0;
    for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToImplementSync()) {
            continue;
        }
        if (qLoadWeightsWaitsFor(prevWaveEdge)) {
            ++numWeights;
        } else {
            ++numIfmaps;
        }
    }
    numSyncedPrevWeights = numWeights;
    numSyncedPrevIfmaps = numIfmaps;
}

//************************************************************************
bool
WaveCodeMatMul::generateLoadWeights(wave::MatMulWaveOp* matmulWaveop)
{
    assert(matmulWaveop->verify());
    const arch::PeArray& peArray(arch::Arch::gArch().gPeArray());

    const EngineId engineId = matmulWaveop->gEngineId();
    if (matmulWaveop->gWeightsSbAddress() < 0 && qParallelStreams()) {
        // this MatMul reuses weights, but even though weights are loaded,
        // wait might need to be implemnted
        std::set<events::EventId> eventIds;
        for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
            if (! prevWaveEdge->qNeedToImplementSync()) {
                continue;
            }

            if (! qSyncOnLdWeightsInstr(prevWaveEdge)) {
                continue;
            }
            if (prevWaveEdge->qSyncedWithEvent()) {
                const auto evtId = prevWaveEdge->gEventId();
                Assert(eventIds.find(evtId) == eventIds.end(), "Double event id ", evtId);
                eventIds.insert(evtId);
                m_WaveCode.writeWaitOrWaitClearInstr(prevWaveEdge, engineId);
            } else if (prevWaveEdge->qSyncedWithSemaphore()) {
                GenerateSemaphoreInstr(prevWaveEdge);
            } else {
                Assert(false, "Must sync edge from ", prevWaveEdge->gFromOp()->gName(),
                       " to ", prevWaveEdge->gToOp()->gName());
            }
        }
        return true; // No LdWeights instructions
    }
    //const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    compisa::LdWeightsInstr ldweightsInstr;

    const utils::DataType& inDtype(matmulWaveop->gInDtype());
    AssignWithSizeCheck(ldweightsInstr.in_dtype, inDtype.gSimTypeId());
    const kcc_int64 addressInSbPart     = matmulWaveop->gWeightsSbAddress();

    initMemAccess(ldweightsInstr.src_mem_pattern);
    {
        // if weights are not properly aligned (end of octet), load more weights so that the
        // last weight (first loaded) is at the end of octet. Those extra junk weights will
        // be ignored.
        // Let a be starting address of the weights and N number of weights.
        // Let b be the address of the last real weight, i.e., b = a + (N-1), for dtype size = 1.
        // If  (b%8)==7, we are done. Otherwise we need to add x so that (b+x)%8==7.
        // (b + x) % 8 = [(b%8)+(x%8)] % 8 = [b%8 + x] % 8.
        // So [b%8 + x]%8 = 7. However, b%8 + x < 8, so x = 7 - b%8
        //
        // For size=2 (fp16), b = a + N*2, and we want b%8=6. Similarly, x = 6 - b%8
        //
        // This calculation should also take into account perf-mode for int8, so that the
        // last address for int8 in any perf mode is 8*n+6 as in fp16 mode.

        enum { OCTET_SIZE = 8 };
        const kcc_int32 dtypeSize               = inDtype.gSizeInBytes();
        const kcc_int32 realNumWeights          = matmulWaveop->gNumColumnPartitions();
        const kcc_int32 realLastAddressInSbPart = addressInSbPart + (realNumWeights - 1) * dtypeSize;
        const kcc_int32 deltaAddress            = (OCTET_SIZE - dtypeSize) - (realLastAddressInSbPart % OCTET_SIZE);
        Assert(deltaAddress >= 0, "Delta address for extra weights must be non-negative, but it is ", deltaAddress);
        const kcc_int32 newLastAddressInSbPart  = realLastAddressInSbPart + deltaAddress;
        Assert(newLastAddressInSbPart % OCTET_SIZE == OCTET_SIZE - dtypeSize,
            "(New LdWeights address) % ", OCTET_SIZE, " is ", newLastAddressInSbPart % OCTET_SIZE, " should be ", OCTET_SIZE - dtypeSize);
        const kcc_int32 deltaNumWeights         = deltaAddress / dtypeSize;
        const kcc_int32 newNumWeights           = realNumWeights + deltaNumWeights;
        Assert(newNumWeights <= peArray.gNumberColumns(),
            "To align weights adding extra ", deltaNumWeights, ", but new num weights, ",
            newNumWeights, ", exceeds the number PE columns ", peArray.gNumberColumns(),
            ". Waveop ", matmulWaveop->gName());

        AssignWithSizeCheck(ldweightsInstr.src_mem_pattern.start_addr, newLastAddressInSbPart);
        AssignWithSizeCheck(ldweightsInstr.src_mem_pattern.step_elem[PatDim_X], -1); // last column goes first, so decrement
        AssignWithSizeCheck(ldweightsInstr.src_mem_pattern.num_elem[PatDim_X], newNumWeights);
        AssignWithSizeCheck(ldweightsInstr.num_active_cols, newNumWeights);
    }

    AssignWithSizeCheck(ldweightsInstr.num_active_rows, matmulWaveop->gNumRowPartitions());

    if (utils::DataTypeId::Uint8 == matmulWaveop->gInDtype().gDataTypeId()) {
        uint8_t quantOffsetWeights = static_cast<uint8_t>(matmulWaveop->gQuantOffsetWeights());
        ldweightsInstr.quant_offset_uint8[0] = quantOffsetWeights;
        ldweightsInstr.quant_offset_uint8[1] = quantOffsetWeights;
        ldweightsInstr.perf_opt = static_cast<TONGA_ISA_TPB_PE_PERF_OPT_MODE>(matmulWaveop->gPEPerfOptMode());
    } else if (utils::DataTypeId::Uint16 == matmulWaveop->gInDtype().gDataTypeId()) {
        ldweightsInstr.quant_offset_uint16 = matmulWaveop->gQuantOffsetWeights();
    }

    AssignWithSizeCheck(ldweightsInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(ldweightsInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(ldweightsInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(ldweightsInstr.inst_events.set_event_mode,events::eventSetMode2Isa(events::EventSetMode::DontSet));

    //************************************************************************
    // incoming events
    //************************************************************************
    bool firstEmbEvt = true;
    // Synchronizations to MatMul predecessors are created on MatMul
    // except for dynamic weights.
    const bool SyncPrevWavesOnMatMulInstr = !matmulWaveop->qIsDynamicWeights();

    if (qParallelStreams()) { // incoming events
        std::set<events::EventId> eventIds;

        for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
            if (! prevWaveEdge->qNeedToImplementSync()) {
                continue;
            }

            // For dynamic weights, LoadWeight instructions should wait for all predecessors
            // of MatMul one of which produces dynamic weights. If we have data dependence information,
            // LoadWeight can wait only for a node producing the weights.
            if (SyncPrevWavesOnMatMulInstr && ! qSyncOnLdWeightsInstr(prevWaveEdge)) {
                continue;
            }
            if (prevWaveEdge->qSyncedWithEvent()) {
                const auto evtId = prevWaveEdge->gEventId();
                Assert(eventIds.find(evtId) == eventIds.end(), "Double event id ", evtId);
                eventIds.insert(evtId);

                if (firstEmbEvt) {
                    firstEmbEvt = false;
                    AssignWithSizeCheck(ldweightsInstr.inst_events.wait_event_idx, evtId);
                    AssignWithSizeCheck(ldweightsInstr.inst_events.wait_event_mode, eventWaitMode2Isa(prevWaveEdge->gWaitEventMode()));
                } else {
                    m_WaveCode.writeWaitOrWaitClearInstr(prevWaveEdge, engineId);
                }
            } else if (prevWaveEdge->qSyncedWithSemaphore()) {
                GenerateSemaphoreInstr(prevWaveEdge);
            } else {
                Assert(false, "Must sync edge from ", prevWaveEdge->gFromOp()->gName(),
                       " to ", prevWaveEdge->gToOp()->gName());
            }
        }
    }


    //************************************************************************
    // No outgoing events
    //************************************************************************

    //************************************************************************
    {
        std::ostringstream oss;
        oss << matmulWaveop->gOrder() << "-" <<  matmulWaveop->gName();
        m_WaveCode.SaveName(ldweightsInstr, oss.str().c_str());
    }
    m_WaveCode.writeInstruction(ldweightsInstr);

    return SyncPrevWavesOnMatMulInstr;
}


//************************************************************************
bool
WaveCodeMatMul::qLoadWeightsWaitsFor(const wave::WaveEdge* prevEdge) const
{
    const auto prevWaveop = prevEdge->gFromOp();
    if (auto prevSbAtomLoadWaveop = dynamic_cast<const wave::SbAtomLoadWaveOp*>(prevWaveop)) {
        if (prevSbAtomLoadWaveop->qContainWeights()) {
            return true;
        }
    }
    return false;
}



//************************************************************************
void
WaveCodeMatMul::generateMatMul(wave::MatMulWaveOp* matmulWaveop, bool SyncPrevWavesOnMatMulInstr)
{
    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    //const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    const EngineId engineId = matmulWaveop->gEngineId();
    Assert(EngineId::PeArray == engineId, "Engine id for MatMul should be PeArray");

    compisa::MatMulInstr matmulInstr;
    AssignWithSizeCheck(matmulInstr.in_dtype, matmulWaveop->gInDtype().gSimTypeId());
    AssignWithSizeCheck(matmulInstr.num_active_rows, matmulWaveop->gNumRowPartitions());
    AssignWithSizeCheck(matmulInstr.num_active_cols, matmulWaveop->gNumColumnPartitions());

    initMemAccess(matmulInstr.src_mem_pattern);
    AssignWithSizeCheck(matmulInstr.src_mem_pattern.start_addr, matmulWaveop->gIfmapsSbAddress());

    AssignWithSizeCheck(matmulInstr.src_mem_pattern.num_elem[PatDim_X], matmulWaveop->gFmapXNum());
    AssignWithSizeCheck(matmulInstr.src_mem_pattern.step_elem[PatDim_X], matmulWaveop->gFmapXStep());

    AssignWithSizeCheck(matmulInstr.src_mem_pattern.num_elem[PatDim_Y], matmulWaveop->gFmapYNum());
    AssignWithSizeCheck(matmulInstr.src_mem_pattern.step_elem[PatDim_Y], matmulWaveop->gFmapYStep());
    AssignWithSizeCheck(matmulInstr.src_mem_pattern.num_elem[PatDim_Z], matmulWaveop->gFmapZNum());
    AssignWithSizeCheck(matmulInstr.src_mem_pattern.step_elem[PatDim_Z], matmulWaveop->gFmapZStep());


    initMemAccess(matmulInstr.dst_mem_pattern);
    AssignWithSizeCheck(matmulInstr.dst_mem_pattern.start_addr,
                        psumBuf.gEntryTpbAddress(matmulWaveop->gPsumBankId(),
                                                 matmulWaveop->gPsumBankOffset(),
                                                 matmulWaveop->gOutDtype()));
    AssignWithSizeCheck(matmulInstr.dst_mem_pattern.num_elem[PatDim_X], matmulWaveop->gPsumXNum());
    AssignWithSizeCheck(matmulInstr.dst_mem_pattern.step_elem[PatDim_X], matmulWaveop->gPsumXStep());
    AssignWithSizeCheck(matmulInstr.dst_mem_pattern.num_elem[PatDim_Y], matmulWaveop->gPsumYNum());
    AssignWithSizeCheck(matmulInstr.dst_mem_pattern.step_elem[PatDim_Y], matmulWaveop->gPsumYStep());
    AssignWithSizeCheck(matmulInstr.dst_mem_pattern.num_elem[PatDim_Z], matmulWaveop->gPsumZNum());
    AssignWithSizeCheck(matmulInstr.dst_mem_pattern.step_elem[PatDim_Z], matmulWaveop->gPsumZStep());

    if (utils::DataTypeId::Uint8 == matmulWaveop->gInDtype().gDataTypeId()) {
        uint8_t quantOffsetIfmaps = static_cast<uint8_t>(matmulWaveop->gQuantOffsetIfmaps());
        matmulInstr.quant_offset_uint8[0] = quantOffsetIfmaps;
        matmulInstr.quant_offset_uint8[1] = quantOffsetIfmaps;
        matmulInstr.perf_opt = static_cast<TONGA_ISA_TPB_PE_PERF_OPT_MODE>(matmulWaveop->gPEPerfOptMode());
    } else if (utils::DataTypeId::Uint16 == matmulWaveop->gInDtype().gDataTypeId()) {
        matmulInstr.quant_offset_uint16 = matmulWaveop->gQuantOffsetIfmaps();
    }


    AssignWithSizeCheck(matmulInstr.timing_flags, 0);
    if (matmulWaveop->qStartTensorCalc()) {
        matmulInstr.timing_flags |= TONGA_ISA_TPB_MATMUL_TIMING_FLAG_BEGIN_TENSOR_CALC;
    }
    if (matmulWaveop->qStopTensorCalc()) {
        matmulInstr.timing_flags |= TONGA_ISA_TPB_MATMUL_TIMING_FLAG_END_TENSOR_CALC;
    }

    AssignWithSizeCheck(matmulInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(matmulInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(matmulInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(matmulInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));


    //************************************************************************
    if (SyncPrevWavesOnMatMulInstr && qParallelStreams()) { // incoming events
        std::set<events::EventId> eventIds;
        bool firstEmb = true;

        // Inspect incoming edges/events
        for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
            if (! prevWaveEdge->qNeedToImplementSync()) {
                continue;
            }
            if (! qSyncOnMatMulInstr(prevWaveEdge)) {
                continue;
            }

            if (prevWaveEdge->qSyncedWithEvent()) {
                const auto evtId = prevWaveEdge->gEventId();
                Assert(eventIds.find(evtId) == eventIds.end(), "Double event id ", evtId);
                eventIds.insert(evtId);

                if (firstEmb) {
                    firstEmb = false;
                    AssignWithSizeCheck(matmulInstr.inst_events.wait_event_idx, evtId);
                    AssignWithSizeCheck(matmulInstr.inst_events.wait_event_mode,
                                        eventWaitMode2Isa(prevWaveEdge->gWaitEventMode()));
                } else {
                    m_WaveCode.writeWaitOrWaitClearInstr(prevWaveEdge, engineId);
                }
            } else if (prevWaveEdge->qSyncedWithSemaphore()) {
                GenerateSemaphoreInstr(prevWaveEdge);
            } else {
                Assert(false, "Must sync edge from ", prevWaveEdge->gFromOp()->gName(),
                       " to ", prevWaveEdge->gToOp()->gName());
            }
        }
    } // end incoming events

    //************************************************************************
    AssignWithSizeCheck(matmulInstr.ifmap_replication_num_rows, matmulWaveop->gIfmapReplicationNumRows());
    AssignWithSizeCheck(matmulInstr.ifmap_replication_resolution, matmulWaveop->gIfmapReplicationResolution());
    AssignWithSizeCheck(matmulInstr.ifmap_replication_shift_amnt, matmulWaveop->gIfmapReplicationShiftAmnt());

    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(matmulWaveop, matmulInstr);
    }
    if (! instructionWritten) {
        std::ostringstream oss;
        oss << matmulWaveop->gOrder() << "-" << matmulWaveop->gName();
        m_WaveCode.SaveName(matmulInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(matmulInstr);
    }
}


}}


