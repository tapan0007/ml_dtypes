#include "shared/inc/tpb_isa_wait.hpp"
#include "shared/inc/tpb_isa_ldweights.hpp"
#include "shared/inc/tpb_isa_matmul.hpp"

#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/sbatomfilewaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodematmul.hpp"

namespace kcc {
namespace wavecode {

#define ASSERT_HAS_EVENT(edge, from, to) Assert((edge)->gEventId() != EventId_Invalid, "WaveEdge from waveop ", \
            (from)->gName(), " to waveop ", (to)->gName(), " has no event")

WaveCodeMatMul::WaveCodeMatMul(WaveCode* waveCode)
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
WaveCodeMatMul::generateLoadWeights(wave::MatMulWaveOp* matmulWaveOp)
{
    assert(matmulWaveOp->verify());
    if (matmulWaveOp->gWeightsOffsetInAtom() < 0) {
        return; // this MatMul reuses weights
    }
    //const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    //const wave::MatMulWaveOp::WaveId& waveId(matmulWaveOp->gWaveId());

    LDWEIGHTS ldweightsInstr;

    //TPB_CMD_HEADER  hdr;
    const utils::DataType& dtype(matmulWaveOp->gInDtype());
    ldweightsInstr.dtype                 = dtype.gSimTypeId();
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
    const kcc_int64 addressInSbPart     = matmulWaveOp->gWeightsAtomId() * matmulWaveOp->gWaveAtomSize()
                                            + matmulWaveOp->gWeightsOffsetInAtom();

    ldweightsInstr.start_addr            = addressInSbPart + (matmulWaveOp->gOfmapCount() - 1) * dtype.gSizeInBytes();

    ldweightsInstr.x_step                = -1; // last column goes first, so decrement
    ldweightsInstr.x_num                 = matmulWaveOp->gOfmapCount();
    ldweightsInstr.num_row_partitions    = matmulWaveOp->gIfmapCount();

    //************************************************************************
    // incoming events
    std::vector<const wave::WaveEdge*> prevWeightEdges;
    std::vector<const wave::WaveEdge*> prevIfmapEdges;
    std::vector<const wave::WaveEdge*> prevPoolEdges;
    std::vector<const wave::WaveEdge*> prevActivationEdges;

    for (auto prevWaveEdge : matmulWaveOp->gPrevWaveEdges()) {
        if (prevWaveEdge->gEventId() == EventId_Invalid) {
            continue;
        }
        auto prevWaveop = prevWaveEdge->gFromOp();
        if (auto prevSbAtomLoadWaveop = dynamic_cast<wave::SbAtomFileWaveOp*>(prevWaveop)) {
            ASSERT_HAS_EVENT(prevWaveEdge, prevSbAtomLoadWaveop, matmulWaveOp);
            if (prevSbAtomLoadWaveop->qContainWeights()) {
                prevWeightEdges.push_back(prevWaveEdge);
            } else {
                prevIfmapEdges.push_back(prevWaveEdge);
            }
            continue;
        }
        if (auto prevPoolWaveop = dynamic_cast<wave::PoolWaveOp*>(prevWaveop)) {
            ASSERT_HAS_EVENT(prevWaveEdge, prevPoolWaveop, matmulWaveOp);
            prevPoolEdges.push_back(prevWaveEdge);
            continue;
        }
        if (auto prevActivationWaveop = dynamic_cast<wave::ActivationWaveOp*>(prevWaveop)) {
            ASSERT_HAS_EVENT(prevWaveEdge, prevActivationWaveop, matmulWaveOp);
            prevActivationEdges.push_back(prevWaveEdge);
            continue;
        }
        Assert(false, "Matmul waveop: waveop ", prevWaveop->gName(), " has wrong type ", prevWaveop->gTypeStr());
    }

    Assert(prevWeightEdges.size() <= 1, "Matmul waveop ", matmulWaveOp->gName(), " can have only one weight predecessor");
    if (prevWeightEdges.size() > 0) {
        auto prevWaveEdge = prevWeightEdges[0];
        //auto prevWaveop = prevWaveEdge->gFromOp();
        ldweightsInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
        ldweightsInstr.sync.wait_event_mode    = eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
    }

    //************************************************************************
    // No outgoing events

    //************************************************************************
    // write instruction
    m_WaveCode->writeInstruction(ldweightsInstr);
}



void
WaveCodeMatMul::generateMatMul(wave::MatMulWaveOp* matmulWaveOp)
{
    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    //const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    const EngineId engineId = matmulWaveOp->gEngineId();

    MATMUL matmulInstr;
    matmulInstr.dtype                   = matmulWaveOp->gInDtype().gSimTypeId();
    matmulInstr.num_row_partitions      = matmulWaveOp->gNumRowPartitions();
    matmulInstr.num_column_partitions   = matmulWaveOp->gNumColumnPartitions();

    matmulInstr.fmap_start_addr         = matmulWaveOp->gIfmapsAtomId() * wave::MatMulWaveOp::AtomSize +
                                          matmulWaveOp->gIfmapsOffsetInAtom();
    matmulInstr.fmap_x_num              = matmulWaveOp->gFmapXNum();
    matmulInstr.fmap_x_step             = matmulWaveOp->gFmapXStep();
    matmulInstr.fmap_y_num              = matmulWaveOp->gFmapYNum();
    matmulInstr.fmap_y_step             = matmulWaveOp->gFmapYStep();
    matmulInstr.fmap_z_num              = matmulWaveOp->gFmapZNum();
    matmulInstr.fmap_z_step             = matmulWaveOp->gFmapZStepAtoms() * matmulWaveOp->gIfmapsAtomSize();

    matmulInstr.psum_start_addr         = psumBuf.gEntryTpbAddress(
                                                    matmulWaveOp->gPsumBankId(),
                                                    matmulWaveOp->gPsumBankOffset(),
                                                    matmulWaveOp->gOutDtype());
    matmulInstr.psum_x_num              = matmulWaveOp->gPsumXNum();
    matmulInstr.psum_x_step             = matmulWaveOp->gPsumXStep();
    matmulInstr.psum_y_num              = matmulWaveOp->gPsumYNum();
    matmulInstr.psum_y_step             = matmulWaveOp->gPsumYStep();

    matmulInstr.start_tensor_calc       = matmulWaveOp->qStartTensorCalc();
    matmulInstr.stop_tensor_calc        = matmulWaveOp->qStopTensorCalc();


    //************************************************************************
    { // incoming events
        // Right now all non-weight prev edges are treated equally, but in
        // the future we may change it, so keep separate arrays.
        std::vector<const wave::WaveEdge*> prevWeightEdges;
        std::vector<const wave::WaveEdge*> prevIfmapEdges;
        std::vector<const wave::WaveEdge*> prevPoolEdges;
        std::vector<const wave::WaveEdge*> prevActivationEdges;

        // Inspect incoming edges/events
        for (auto prevWaveEdge : matmulWaveOp->gPrevWaveEdges()) {
            if (prevWaveEdge->gEventId() == EventId_Invalid) {
                continue;
            }
            auto prevWaveop = prevWaveEdge->gFromOp();
            if (prevWaveop->gEngineId() == engineId) {
                continue;
            }
            if (auto prevSbAtomLoadWaveop = dynamic_cast<wave::SbAtomLoadWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevSbAtomLoadWaveop, matmulWaveOp);
                if (prevSbAtomLoadWaveop->qContainWeights()) {
                    prevWeightEdges.push_back(prevWaveEdge);
                } else {
                    prevIfmapEdges.push_back(prevWaveEdge);
                }
                continue;
            }
            if (auto prevPoolWaveop = dynamic_cast<wave::PoolWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevPoolWaveop, matmulWaveOp);
                prevPoolEdges.push_back(prevWaveEdge);
                continue;
            }
            if (auto prevActivationWaveop = dynamic_cast<wave::ActivationWaveOp*>(prevWaveop)) {
                ASSERT_HAS_EVENT(prevWaveEdge, prevActivationWaveop, matmulWaveOp);
                prevActivationEdges.push_back(prevWaveEdge);
                continue;
            }
            Assert(false, "Matmul waveop: predecessor waveop ", prevWaveop->gName(), " has wrong type ", prevWaveop->gTypeStr());
        }

        bool firstEmb = true;
        for (auto prevWaveEdge : prevIfmapEdges) {
            if (firstEmb) {
                matmulInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                matmulInstr.sync.wait_event_mode    = events::eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                firstEmb = false;
            } else {
                WAIT waitInstr;
                waitInstr.event_id                  = prevWaveEdge->gEventId();
                m_WaveCode->writeInstruction(waitInstr, EngineId::PeArray);
            }
        }
        for (auto prevWaveEdge : prevPoolEdges) {
            if (firstEmb) {
                matmulInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                matmulInstr.sync.wait_event_mode    = events::eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                firstEmb = false;
            } else {
                WAIT waitInstr;
                waitInstr.event_id                  = prevWaveEdge->gEventId();
                m_WaveCode->writeInstruction(waitInstr, EngineId::PeArray);
            }
        }
        for (auto prevWaveEdge : prevActivationEdges) {
            if (firstEmb) {
                matmulInstr.sync.wait_event_id      = prevWaveEdge->gEventId();
                matmulInstr.sync.wait_event_mode    = events::eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
                firstEmb = false;
            } else {
                WAIT waitInstr;
                waitInstr.event_id                  = prevWaveEdge->gEventId();
                m_WaveCode->writeInstruction(waitInstr, EngineId::PeArray);
            }
        }
    }




    //************************************************************************
    { // Outgoing events
        // Right now all succ edges are treated equally, but in the future
        // we may change it, so keep separate arrays.
        std::vector<const wave::WaveEdge*> succIfmapEdges;
        std::vector<const wave::WaveEdge*> succPoolEdges;
        std::vector<const wave::WaveEdge*> succActivationEdges;

        for (auto succWaveEdge : matmulWaveOp->gSuccWaveEdges()) {
            if (succWaveEdge->gEventId() == EventId_Invalid) {
                continue;
            }
            auto succWaveop = succWaveEdge->gFromOp();
            if (succWaveop->gEngineId() == engineId) {
                continue;
            }
            if (auto succSbAtomSaveWaveop = dynamic_cast<wave::SbAtomSaveWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, matmulWaveOp, succSbAtomSaveWaveop);
                succIfmapEdges.push_back(succWaveEdge);
                continue;
            }
            if (auto succPoolWaveop = dynamic_cast<wave::PoolWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, matmulWaveOp, succPoolWaveop);
                succPoolEdges.push_back(succWaveEdge);
                continue;
            }
            if (auto succActivationWaveop = dynamic_cast<wave::ActivationWaveOp*>(succWaveop)) {
                ASSERT_HAS_EVENT(succWaveEdge, matmulWaveOp, succActivationWaveop);
                succActivationEdges.push_back(succWaveEdge);
                continue;
            }
            Assert(false, "Matmul waveop: successor waveop ", succWaveop->gName(), " has wrong type ", succWaveop->gTypeStr());
        }

        bool firstEmb = true;
        for (auto succWaveEdge : succIfmapEdges) {
            WRITE writeInstr;
            writeInstr.dst_address  = m_WaveCode->calculateEventAddress(EngineId::DmaEng, succWaveEdge->gEventId());
            writeInstr.data         = ~(0UL);  // writing is for remote event-set. All 1's ensure that bit/byte endianess does not matter.
            writeInstr.nbytes       = 1;

            m_WaveCode->writeInstruction(writeInstr, EngineId::DmaEng);
        }
        for (auto succWaveEdge : succPoolEdges) {
            if (firstEmb) {
                matmulInstr.sync.set_event_id       = succWaveEdge->gEventId();
                matmulInstr.sync.set_event_mode     = events::eventSetMode2Int(succWaveEdge->gSetEventMode());
                firstEmb = false;
            } else {
                SET setEventInstr;
                setEventInstr.event_id          = succWaveEdge->gEventId();
                m_WaveCode->writeInstruction(setEventInstr, EngineId::Pooling);
            }
        }
        for (auto succWaveEdge : succActivationEdges) {
            if (firstEmb) {
                matmulInstr.sync.set_event_id       = succWaveEdge->gEventId();
                matmulInstr.sync.set_event_mode     = events::eventSetMode2Int(succWaveEdge->gSetEventMode());
                firstEmb = false;
            } else {
                SET setEventInstr;
                setEventInstr.event_id          = succWaveEdge->gEventId();
                m_WaveCode->writeInstruction(setEventInstr, EngineId::Activation);
            }
        }

    }

    //************************************************************************
    // write instruction
    m_WaveCode->writeInstruction(matmulInstr);
}


}}


