
#include "compisa/inc/compisamatadd.hpp"



#include "utils/inc/asserter.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/resaddwaveop.hpp"
#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecoderesadd.hpp"

namespace kcc {
namespace wavecode {

WaveCodeResAdd::WaveCodeResAdd(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeResAdd::generate(wave::WaveOp* waveOp)
{
    auto resaddWaveop = dynamic_cast<wave::ResAddWaveOp*>(waveOp);
    assert(resaddWaveop);

    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());

    const EngineId engineId = resaddWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for Pool should be Pooling");

    compisa::MatAddInstr resaddInstr;

    resaddInstr.in_a_dtype          = resaddWaveop->gInADtype().gSimTypeId();
    resaddInstr.in_b_dtype          = resaddWaveop->gInBDtype().gSimTypeId();
    resaddInstr.out_dtype           = resaddWaveop->gOutDtype().gSimTypeId();
    resaddInstr.num_partitions      = resaddWaveop->gNumPartitions();

    // SrcA
    if (resaddWaveop->qSrcAIsPsum()) {
        resaddInstr.src_a_start_addr  = psumBuf.gEntryTpbAddress(resaddWaveop->gSrcAPsumBankId(),
                                                                 resaddWaveop->gSrcAPsumBankOffset(),
                                                                 resaddWaveop->gInADtype());
    } else {
        resaddInstr.src_a_start_addr  = stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * resaddWaveop->gSrcAStartAtMidPart(),
                                                resaddWaveop->gSrcASbAddress());
    }
    resaddInstr.src_a_x_step      = resaddWaveop->gSrcAXStep();
    resaddInstr.src_a_y_step      = resaddWaveop->gSrcAYStep();
    resaddInstr.src_a_z_step      = resaddWaveop->gSrcAZStep();
    resaddInstr.src_a_x_num       = resaddWaveop->gSrcAXNum();
    resaddInstr.src_a_y_num       = resaddWaveop->gSrcAYNum();
    resaddInstr.src_a_z_num       = resaddWaveop->gSrcAZNum();

    // SrcB
    if (resaddWaveop->qSrcBIsPsum()) {
        resaddInstr.src_b_start_addr  = psumBuf.gEntryTpbAddress(resaddWaveop->gSrcBPsumBankId(),
                                                                 resaddWaveop->gSrcBPsumBankOffset(),
                                                                 resaddWaveop->gInBDtype());
    } else {
        resaddInstr.src_b_start_addr  = stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * resaddWaveop->gSrcBStartAtMidPart(),
                                                resaddWaveop->gSrcBSbAddress());
    }
    resaddInstr.src_b_x_step      = resaddWaveop->gSrcBXStep();
    resaddInstr.src_b_y_step      = resaddWaveop->gSrcBYStep();
    resaddInstr.src_b_z_step      = resaddWaveop->gSrcBZStep();
    resaddInstr.src_b_x_num       = resaddWaveop->gSrcBXNum();
    resaddInstr.src_b_y_num       = resaddWaveop->gSrcBYNum();
    resaddInstr.src_b_z_num       = resaddWaveop->gSrcBZNum();

    // Dst
    if (resaddWaveop->qDstIsPsum()) {
        resaddInstr.dst_start_addr  = psumBuf.gEntryTpbAddress(resaddWaveop->gDstPsumBankId(),
                                                                 resaddWaveop->gDstPsumBankOffset(),
                                                                 resaddWaveop->gOutDtype());
    } else {
        resaddInstr.dst_start_addr  = stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * resaddWaveop->gDstStartAtMidPart(),
                                                resaddWaveop->gDstSbAddress());
    }
    resaddInstr.dst_x_step      = resaddWaveop->gDstXStep();
    resaddInstr.dst_y_step      = resaddWaveop->gDstYStep();
    resaddInstr.dst_z_step      = resaddWaveop->gDstZStep();
    resaddInstr.dst_x_num       = resaddWaveop->gDstXNum();
    resaddInstr.dst_y_num       = resaddWaveop->gDstYNum();
    resaddInstr.dst_z_num       = resaddWaveop->gDstZNum();

    resaddInstr.sync.wait_event_id    = 0;
    resaddInstr.sync.wait_event_mode  = events::eventWaitMode2Int(events::EventWaitMode::DontWait);
    resaddInstr.sync.set_event_id    = 0;
    resaddInstr.sync.set_event_mode  = events::eventSetMode2Int(events::EventSetMode::DontSet);

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(resaddWaveop, resaddInstr.sync);
    } // end incoming events

    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(resaddWaveop, resaddInstr);
    }


    if (! instructionWritten) {
        m_WaveCode.writeInstruction(resaddInstr);
    }
}


}}


