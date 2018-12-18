
#include "compisa/inc/compisaregstore.hpp"



#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/regstorewaveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
//#include "wave/inc/sbatomloadwaveop.hpp"
//#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecoderegstore.hpp"

namespace kcc {
namespace wavecode {


WaveCodeRegStore::WaveCodeRegStore(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}


void
WaveCodeRegStore::generate(wave::WaveOp* waveOp)
{
    auto regstoreWaveop = dynamic_cast<wave::RegStoreWaveOp*>(waveOp);
    assert(regstoreWaveop);
    const arch::Arch& arch(arch::Arch::gArch());

    const auto& stateBuf(arch.gStateBuffer());

    const EngineId engineId = regstoreWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for Pool should be Pooling");

    compisa::RegStoreInstr regstoreInstr;

    AssignWithSizeCheck(regstoreInstr.dtype, regstoreWaveop->gOutDtype().gSimTypeId());
    //AssignWithSizeCheck(regstoreInstr.out_dtype, regstoreWaveop->gOutDtype().gSimTypeId());

    initMemAccess(regstoreInstr.dst_mem_pattern);

    AssignWithSizeCheck(regstoreInstr.dst_mem_pattern.start_addr,
        stateBuf.gEntryTpbAddress(
            arch.gNumberPeArrayRows()/2 * regstoreWaveop->gDstStartAtMidPart(),
            regstoreWaveop->gDstSbAddress()));

    AssignWithSizeCheck(regstoreInstr.num_active_channels, regstoreWaveop->gNumPartitions());
    if (regstoreWaveop->qParallelMode()) {
        AssignWithSizeCheck(regstoreInstr.serialization_mode, TONGA_ISA_TPB_REG_SERIALIZATION_MODE_PARALLEL);
    } else {
        AssignWithSizeCheck(regstoreInstr.serialization_mode, TONGA_ISA_TPB_REG_SERIALIZATION_MODE_SERIAL);
    }

    AssignWithSizeCheck(regstoreInstr.dst_mem_pattern.step_elem[PatDim_X], regstoreWaveop->gDstXStep());
    AssignWithSizeCheck(regstoreInstr.dst_mem_pattern.num_elem[PatDim_X], regstoreWaveop->gDstXNum());
    AssignWithSizeCheck(regstoreInstr.dst_mem_pattern.step_elem[PatDim_Y], regstoreWaveop->gDstYStep());
    AssignWithSizeCheck(regstoreInstr.dst_mem_pattern.num_elem[PatDim_Y], regstoreWaveop->gDstYNum());
    AssignWithSizeCheck(regstoreInstr.dst_mem_pattern.step_elem[PatDim_Z], regstoreWaveop->gDstZStep());
    AssignWithSizeCheck(regstoreInstr.dst_mem_pattern.num_elem[PatDim_Z], regstoreWaveop->gDstZNum());


    AssignWithSizeCheck(regstoreInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(regstoreInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(regstoreInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(regstoreInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(regstoreWaveop, regstoreInstr.inst_events);
    } // end incoming events


    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(regstoreWaveop, regstoreInstr);
    }

    if (! instructionWritten) {
        std::ostringstream oss;
        oss << regstoreWaveop->gOrder() << "-" << regstoreWaveop->gName();
        m_WaveCode.SaveName(regstoreInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(regstoreInstr);
    }
}

}}


