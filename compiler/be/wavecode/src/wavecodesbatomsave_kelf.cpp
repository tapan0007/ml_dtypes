#include "utils/inc/asserter.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"


#include "compisa/inc/compisadmatrigger.hpp"

#include "events/inc/events.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "kelf/inc/kelfdmadescription.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomsave_kelf.hpp"

namespace kcc {
namespace wavecode {

//************************************************************************
WaveCodeSbAtomSaveKelf::WaveCodeSbAtomSaveKelf(WaveCodeRef waveCode)
    : WaveCodeSbAtomSave(waveCode)
{}


//************************************************************************
void
WaveCodeSbAtomSaveKelf::generate(wave::WaveOp* waveop)
{
    Assert(qGenerateKelf(), "Must be in Kelf mode to save for Kelf");
    auto sbAtomSaveWaveop = dynamic_cast<wave::SbAtomSaveWaveOp*>(waveop);
    Assert(sbAtomSaveWaveop, "Expecting Save waveop");
    calcOutputSize(sbAtomSaveWaveop);
    generateForKelf(sbAtomSaveWaveop);
}

//************************************************************************
void
WaveCodeSbAtomSaveKelf::generateForKelf(wave::SbAtomSaveWaveOp* sbAtomSaveWaveop)
{
    EngineId chosenEngId;
    std::vector<events::EventId> succEventIds;
    /*const kcc_int32 numSyncs =*/ findSuccEventsAndChosenEngine(sbAtomSaveWaveop,
                                        chosenEngId, succEventIds);

    // TODO: handle debug and off-load SbAtomSaves (variable tmp)
    generateDmaTriggerRuntimeKelf(sbAtomSaveWaveop, chosenEngId, succEventIds);
}


//************************************************************************
void
WaveCodeSbAtomSaveKelf::generateDmaTriggerRuntimeKelf(wave::SbAtomSaveWaveOp* sbAtomSaveWaveop,
                    EngineId chosenEngId, const std::vector<events::EventId>& succEventIds)
{
    Assert(m_WaveCode.qBinFileRuntimeKelf(), "Must be binary for Runtime Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());


    const kcc_int64 numPartitions   = sbAtomSaveWaveop->gNumPartitions();
    const kcc_int64 numBytesPerPart = sbAtomSaveWaveop->gLength();
    const kcc_int64 addressInPart   = sbAtomSaveWaveop->gSbAddress();
    const kcc_int64 stepSize        = sbAtomSaveWaveop->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomSaveWaveop->gStartAtMidPart() ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;

    //************************************************************************
    const bool qOut = sbAtomSaveWaveop-> qFinalLayerOfmap();
    std::ostringstream oss;
    oss << sbAtomSaveWaveop->gOrder() << "-" << sbAtomSaveWaveop->gName();
    const kcc_int32 blockIdx(kelfDma.startNewDmaBlockFromTpb(
                                        sbAtomSaveWaveop->gDmaQueue(), chosenEngId,
                                        qOut, oss.str().c_str()));
    kelf::DmaDescription::DmaBlockFromTpb* dmaBlock = kelfDma.gDmaBlockFromTpb(blockIdx);

    const std::string refFileName(sbAtomSaveWaveop->gRefFileName());

    for (kcc_int32 partIdx = startPart; partIdx < startPart + numPartitions; ++partIdx) {
        const TongaAddress  fileAddress = sbAtomSaveWaveop->gOffsetInFile() + (partIdx * stepSize);
        const TpbAddress    sbAddress = stateBuf.gEntryTpbAddress(partIdx, addressInPart);
        dmaBlock->addDmaDesc(sbAddress, fileAddress, refFileName, numBytesPerPart);
    }
    for (auto eventId : succEventIds) {
        dmaBlock->addTailEventId(eventId);
    }
    //************************************************************************
    // Incoming events were processed in
    // WaveCodeSbAtomSaveKelf::findSuccEventsAndChosenEngine(),
    // so processing them again in processIncomingEdgesForceWait is wrong.
    // The event on the chosen pred edge is replaced by sequential execution of
    // the previous waveop and TRIGGER.
    //
    // Of non-chosen pred edges, one of the Waits can be implemented as embedded.
    // The current code does not do that yet.

    if (false && qParallelStreams()) { // incoming events, adds wait for each
        events::EventId waitEventId = 0; // events::EventId_Invalid();
        events::EventWaitMode waitEventMode = events::EventWaitMode::DontWait;

        processIncomingEdgesForceWait(sbAtomSaveWaveop, chosenEngId, waitEventId, waitEventMode);
    } // end incoming events

    //************************************************************************
    addDmaBarrier(sbAtomSaveWaveop, chosenEngId);
    //************************************************************************
    compisa::DmaTriggerInstr dmaTriggerInstr;
    dmaTriggerInstr.SetDmaQueueName(dmaBlock->gDmaQueue()->gName().c_str());

    AssignWithSizeCheck(dmaTriggerInstr.use_raw_count, 0); // get from JSON
    AssignWithSizeCheck(dmaTriggerInstr.block_id, dmaBlock->gBlockId());

    AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_idx, 0); // succ event is in desc
    AssignWithSizeCheck(dmaTriggerInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));
    {
        const auto eventId = (succEventIds.size() > 0 ?  succEventIds[0] : -1);
        std::ostringstream oss;
        oss << sbAtomSaveWaveop->gOrder() << ":" << eventId << "-" << sbAtomSaveWaveop->gName();
        m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
    }
    m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
}

//************************************************************************

}}



