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
    ldweightsInstr.src_mem_pattern.start_addr   = addressInSbPart
                            + (matmulWaveop->gOfmapCount() - 1) * inDtype.gSizeInBytes();
    ldweightsInstr.src_mem_pattern.step_elem[0] = -1; // last column goes first, so decrement
    ldweightsInstr.src_mem_pattern.num_elem[1]  = matmulWaveop->gOfmapCount();

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
            if (! prevWaveEdge->qNeedToImplementWait()) {
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
    matmulInstr.src_mem_pattern.num_elem[0]     = matmulWaveop->gFmapXNum();
    matmulInstr.src_mem_pattern.step_elem[0]    = matmulWaveop->gFmapXStep();
    matmulInstr.src_mem_pattern.num_elem[1]     = matmulWaveop->gFmapYNum();
    matmulInstr.src_mem_pattern.step_elem[1]    = matmulWaveop->gFmapYStep();
    matmulInstr.src_mem_pattern.num_elem[2]     = matmulWaveop->gFmapZNum();
    matmulInstr.src_mem_pattern.step_elem[2]    = 1;


    initMemAccess(matmulInstr.dst_mem_pattern);
    matmulInstr.dst_mem_pattern.start_addr         = psumBuf.gEntryTpbAddress(
                                                        matmulWaveop->gPsumBankId(),
                                                        matmulWaveop->gPsumBankOffset(),
                                                        matmulWaveop->gOutDtype());
    matmulInstr.dst_mem_pattern.num_elem[0]        = matmulWaveop->gPsumXNum();
    matmulInstr.dst_mem_pattern.step_elem[0]       = matmulWaveop->gPsumXStep();
    matmulInstr.dst_mem_pattern.num_elem[1]        = matmulWaveop->gPsumYNum();
    matmulInstr.dst_mem_pattern.step_elem[1]       = matmulWaveop->gPsumYStep();


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
            if (! prevWaveEdge->qNeedToImplementWait()) {
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
        m_WaveCode.writeInstruction(matmulInstr);
    }
}


}}


