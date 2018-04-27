#include <set>



#include "utils/inc/asserter.hpp"

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
#include "wave/inc/barrierwaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodematmul.hpp"

namespace kcc {
namespace wavecode {


WaveCodeMatMul::WaveCodeMatMul(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeMatMul::generate(wave::WaveOp* waveOp)
{
    auto matmulWaveOp = dynamic_cast<wave::MatMulWaveOp*>(waveOp);
    assert(matmulWaveOp);

    generateLoadWeights(matmulWaveOp);
    generateMatMul(matmulWaveOp);
}



void
WaveCodeMatMul::generateLoadWeights(wave::MatMulWaveOp* matmulWaveop)
{
    assert(matmulWaveop->verify());
    if (matmulWaveop->gWeightsSbAddress() < 0) {
        return; // this MatMul reuses weights
    }
    const EngineId engineId = matmulWaveop->gEngineId();
    //const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    //const wave::MatMulWaveOp::WaveId& waveId(matmulWaveop->gWaveId());

    compisa::LdWeightsInstr ldweightsInstr;

    //TPB_CMD_HEADER  hdr;
    const utils::DataType& inDtype(matmulWaveop->gInDtype());
    ldweightsInstr.in_dtype                 = inDtype.gSimTypeId();
    //uint8_t         perf_opt = OPT_NONE;
    //uint8_t         dquant_table_idx  = 0;
    //uint8_t         dquant_in_dsize   = 0;
    //uint8_t         dquant_out_dtype  = INVALID_ARBPRECTYPE;
    //uint8_t         dquant_enable  = 0;
    ///* subtract this from each ldweights on way into PE Array */
    //union {
    //    uint8_t     zero_point_uint8[2];
    //    uint16_t    zero_point_uint16   = 0;
    //} TONGA_PACKED;
    const kcc_int64 addressInSbPart     = matmulWaveop->gWeightsSbAddress();

    initMemAccess(ldweightsInstr.src_mem_pattern);
    if (false) {
        const kcc_int32 dtypeSize                           = inDtype.gSizeInBytes();
        const kcc_int32 numWeights                          = matmulWaveop->gOfmapCount(); 
        const kcc_int32 lastAddressInSbPart                 = addressInSbPart + (numWeights - 1) * dtypeSize;
        ldweightsInstr.src_mem_pattern.start_addr           = lastAddressInSbPart;
        ldweightsInstr.src_mem_pattern.step_elem[PatDim_X]  = -1; // last column goes first, so decrement
        ldweightsInstr.src_mem_pattern.num_elem[PatDim_X]   = numWeights;
        ldweightsInstr.num_active_cols                      = numWeights;
    } else {
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
        const kcc_int32 realNumWeights          = matmulWaveop->gOfmapCount();
        const kcc_int32 realLastAddressInSbPart = addressInSbPart + (realNumWeights - 1) * dtypeSize;
        const kcc_int32 deltaAddress            = (OCTET_SIZE - dtypeSize) - (realLastAddressInSbPart % OCTET_SIZE);
        Assert(deltaAddress >= 0, "Delta address for extra weights must be non-negative, but it is ", deltaAddress);
        const kcc_int32 newLastAddressInSbPart  = realLastAddressInSbPart + deltaAddress;
        Assert(newLastAddressInSbPart % OCTET_SIZE == OCTET_SIZE - dtypeSize,
            "(New LdWeights address) % ", OCTET_SIZE, " is ", newLastAddressInSbPart % OCTET_SIZE, " should be ", OCTET_SIZE - dtypeSize);
        const kcc_int32 deltaNumWeights         = deltaAddress / dtypeSize;
        const kcc_int32 newNumWeights           = realNumWeights + deltaNumWeights;

        ldweightsInstr.src_mem_pattern.start_addr           = newLastAddressInSbPart;
        ldweightsInstr.src_mem_pattern.step_elem[PatDim_X]  = -1; // last column goes first, so decrement
        ldweightsInstr.src_mem_pattern.num_elem[PatDim_X]   = newNumWeights;
        ldweightsInstr.num_active_cols                      = newNumWeights;
    }

    ldweightsInstr.num_active_rows              = matmulWaveop->gIfmapCount();

    ldweightsInstr.inst_events.wait_event_idx   = 0;
    ldweightsInstr.inst_events.wait_event_mode  = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    ldweightsInstr.inst_events.set_event_idx    = 0;
    ldweightsInstr.inst_events.set_event_mode   = events::eventSetMode2Isa(events::EventSetMode::DontSet);

    //************************************************************************
    // incoming events
    //************************************************************************
    bool firstEmbEvt = true;

    if (qParallelStreams()) { // incoming events
        std::set<events::EventId> eventIds;

        for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
            if (! prevWaveEdge->qNeedToImplementSync()) {
                continue;
            }
            if (! qLoadWeightsWaitsFor(prevWaveEdge)) {
                continue;
            }

            const auto evtId = prevWaveEdge->gEventId();
            Assert(eventIds.find(evtId) == eventIds.end(), "Double event id ", evtId);
            eventIds.insert(evtId);

            if (firstEmbEvt) {
                firstEmbEvt = false;
                ldweightsInstr.inst_events.wait_event_idx     = evtId;
                ldweightsInstr.inst_events.wait_event_mode    = eventWaitMode2Isa(prevWaveEdge->gWaitEventMode());
            } else {
                writeWaitOrWaitClearInstr(prevWaveEdge, engineId);
            }
        }
    }


    //************************************************************************
    // No outgoing events
    //************************************************************************

    //************************************************************************
    {
        SaveName(ldweightsInstr, matmulWaveop->gName().c_str());
    }
    m_WaveCode.writeInstruction(ldweightsInstr);
}


bool
WaveCodeMatMul::qLoadWeightsWaitsFor(const wave::WaveEdge* prevEdge) const
{
    const auto prevWaveop = prevEdge->gFromOp();
    if (auto prevSbAtomLoadWaveop = dynamic_cast<const wave::SbAtomLoadWaveOp*>(prevWaveop)) {
        if (prevSbAtomLoadWaveop->qContainWeights()) {
            return true;
        }
    }
    if (dynamic_cast<const wave::BarrierWaveOp*>(prevWaveop)) {
        return true;
    }
    return false;
}



void
WaveCodeMatMul::generateMatMul(wave::MatMulWaveOp* matmulWaveop)
{
    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    //const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    const EngineId engineId = matmulWaveop->gEngineId();
    Assert(EngineId::PeArray == engineId, "Engine id for MatMul should be PeArray");

    compisa::MatMulInstr matmulInstr;
    matmulInstr.in_dtype        = matmulWaveop->gInDtype().gSimTypeId();
    matmulInstr.num_active_rows = matmulWaveop->gNumRowPartitions();
    matmulInstr.num_active_cols = matmulWaveop->gNumColumnPartitions();

    initMemAccess(matmulInstr.src_mem_pattern);
    matmulInstr.src_mem_pattern.start_addr      = matmulWaveop->gIfmapsSbAddress();
    matmulInstr.src_mem_pattern.num_elem[PatDim_X]     = matmulWaveop->gFmapXNum();
    matmulInstr.src_mem_pattern.step_elem[PatDim_X]    = matmulWaveop->gFmapXStep();
    matmulInstr.src_mem_pattern.num_elem[PatDim_Y]     = matmulWaveop->gFmapYNum();
    matmulInstr.src_mem_pattern.step_elem[PatDim_Y]    = matmulWaveop->gFmapYStep();
    matmulInstr.src_mem_pattern.num_elem[PatDim_Z]     = matmulWaveop->gFmapZNum();
    matmulInstr.src_mem_pattern.step_elem[PatDim_Z]    = 1;


    initMemAccess(matmulInstr.dst_mem_pattern);
    matmulInstr.dst_mem_pattern.start_addr         = psumBuf.gEntryTpbAddress(
                                                        matmulWaveop->gPsumBankId(),
                                                        matmulWaveop->gPsumBankOffset(),
                                                        matmulWaveop->gOutDtype());
    matmulInstr.dst_mem_pattern.num_elem[PatDim_X]        = matmulWaveop->gPsumXNum();
    matmulInstr.dst_mem_pattern.step_elem[PatDim_X]       = matmulWaveop->gPsumXStep();
    matmulInstr.dst_mem_pattern.num_elem[PatDim_Y]        = matmulWaveop->gPsumYNum();
    matmulInstr.dst_mem_pattern.step_elem[PatDim_Y]       = matmulWaveop->gPsumYStep();


    matmulInstr.timing_flags = 0;
    if (matmulWaveop->qStartTensorCalc()) {
        matmulInstr.timing_flags |= TONGA_ISA_TPB_MATMUL_TIMING_FLAG_BEGIN_TENSOR_CALC;
    }
    if (matmulWaveop->qStopTensorCalc()) {
        matmulInstr.timing_flags |= TONGA_ISA_TPB_MATMUL_TIMING_FLAG_END_TENSOR_CALC;
    }

    matmulInstr.inst_events.wait_event_idx  = 0;
    matmulInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    matmulInstr.inst_events.set_event_idx   = 0;
    matmulInstr.inst_events.set_event_mode  = events::eventSetMode2Isa(events::EventSetMode::DontSet);

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        std::set<events::EventId> eventIds;
        bool firstEmb = true;

        // Inspect incoming edges/events
        for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
            if (! prevWaveEdge->qNeedToImplementSync()) {
                continue;
            }
            if (qLoadWeightsWaitsFor(prevWaveEdge)) {
                continue;
            }

            const auto evtId = prevWaveEdge->gEventId();
            Assert(eventIds.find(evtId) == eventIds.end(), "Double event id ", evtId);
            eventIds.insert(evtId);

            if (firstEmb) {
                firstEmb = false;
                matmulInstr.inst_events.wait_event_idx     = evtId;
                matmulInstr.inst_events.wait_event_mode    = eventWaitMode2Isa(prevWaveEdge->gWaitEventMode());
            } else {
                writeWaitOrWaitClearInstr(prevWaveEdge, engineId);
            }
        }
    } // end incoming events


    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(matmulWaveop, matmulInstr);
    }
    if (! instructionWritten) {
        SaveName(matmulInstr, matmulWaveop->gName().c_str());
        m_WaveCode.writeInstruction(matmulInstr);
    }
}


}}


