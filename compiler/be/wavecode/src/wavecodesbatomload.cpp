#include "utils/inc/debug.hpp"
#include "utils/inc/misc.hpp"
#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"

#include "compisa/inc/compisasimwrnpy.hpp"
#include "compisa/inc/compisasimmemcpy.hpp"
#include "compisa/inc/compisadmatrigger.hpp"
#include "compisa/inc/compisasimdmacopy.hpp"


#include "events/inc/events.hpp"


#include "layers/inc/inputlayer.hpp"
#include "layers/inc/constlayer.hpp"
#include "layers/inc/convlayer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"

#include "kelf/inc/kelfdmadescription.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomload.hpp"

namespace kcc {
namespace wavecode {




//************************************************************************
WaveCodeSbAtomLoad::WaveCodeSbAtomLoad(WaveCodeRef waveCode)
    : WaveCodeSbAtom(waveCode)
{}

//************************************************************************
void
WaveCodeSbAtomLoad::generate(wave::WaveOp* waveOp)
{
    const auto sbAtomLoadWaveOp = dynamic_cast<wave::SbAtomLoadWaveOp*>(waveOp);
    assert(sbAtomLoadWaveOp);
    calcInputSize(sbAtomLoadWaveOp);
    const kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());
    if (qGenerateKelf()) {
        if (sbAtomLoadWaveOp->qContainWeights() || m_WaveCode.qBinFileSimKelf()) {
            generateForKelf(sbAtomLoadWaveOp);
        } else {
            if (kelfDma.qHasFile(sbAtomLoadWaveOp->gRefFileName())) {
                generateForKelf(sbAtomLoadWaveOp); // intermediate load
            } else {
                generateInputDma(sbAtomLoadWaveOp);
            }
        }
    } else {
        generateForSim(sbAtomLoadWaveOp);
    }
}

void
WaveCodeSbAtomLoad::generateForSim(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    if (sbAtomLoadWaveop->gIfmapReplicationResolution() == 0) {
        generateForSimNoRepl(sbAtomLoadWaveop);
    } else {
        generateForSimWithRepl(sbAtomLoadWaveop);
    }
}

//************************************************************************
// Suppose predecessors are w0, w1, w2
// Successors are w3, w4, w5
// We want to issue the following instructions:
// WAIT(w1)
// WAIT(w2)
// MEMCPY first partition with embedded WAIT(w0) and with no-set
// MEMCPY middle partitions with no-wait and no-set
// MEMCPY last partition with no-wait and SET(w3)
// SET(w4)
// SET(w5)
//************************************************************************
void
WaveCodeSbAtomLoad::generateForSimNoRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveOp)
{
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    const EngineId engineId = sbAtomLoadWaveOp->gEngineId();
    Assert(EngineId::None != engineId, "Engine id for SbAtomLoad waveop should not be None");

    //************************************************************************
    kcc_int64 npyFileDramOffset = m_WaveCode.getDramForNpyFile(sbAtomLoadWaveOp->gRefFileName());
    if (npyFileDramOffset < 0) { // Load whole numpy file to DRAM
        compisa::SimWrNpyInstr npyToDramInstr;
        npyToDramInstr.inst_events.wait_event_idx   = 0;
        npyToDramInstr.inst_events.wait_event_mode  = eventWaitMode2Isa(events::EventWaitMode::DontWait);
        npyToDramInstr.inst_events.set_event_idx    = 0;
        npyToDramInstr.inst_events.set_event_mode   = eventSetMode2Isa(events::EventSetMode::DontSet);

        const kcc_int64 numPySize = sbAtomLoadWaveOp->gLoadDataSizeInBytes();
        strcpy(npyToDramInstr.src_fname, sbAtomLoadWaveOp->gRefFileName().c_str());
        npyFileDramOffset           = m_WaveCode.gCurrentDramAddress(numPySize);

        npyToDramInstr.dst_addr     = npyFileDramOffset;
        m_WaveCode.writeInstruction(npyToDramInstr);

        WaveCode::NpyFileInfo npyFileInfo;
        npyFileInfo.m_Dirty          = false;
        npyFileInfo.m_FileDramOffset = npyFileDramOffset;
        npyFileInfo.m_SimTypeId = sbAtomLoadWaveOp->gDataType().gSimTypeId();
        npyFileInfo.m_RefFileShape = sbAtomLoadWaveOp->gRefFileShape();
        m_WaveCode.recordDramForNpyFile(sbAtomLoadWaveOp->gRefFileName(), npyFileInfo);
    }

    //************************************************************************
    compisa::SimMemCpyInstr dramToStateBufInstr;
    dramToStateBufInstr.inst_events.wait_event_idx  = 0;
    dramToStateBufInstr.inst_events.wait_event_mode = eventWaitMode2Isa(events::EventWaitMode::DontWait);
    dramToStateBufInstr.inst_events.set_event_idx   = 0;
    dramToStateBufInstr.inst_events.set_event_mode  = eventSetMode2Isa(events::EventSetMode::DontSet);

    events::EventId setEventId = 0; // events::EventId_Invalid();
    events::EventSetMode setEventMode = events::EventSetMode::DontSet;
    events::EventId waitEventId = 0; // events::EventId_Invalid();
    events::EventWaitMode waitEventMode = events::EventWaitMode::DontWait;

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(sbAtomLoadWaveOp, waitEventId, waitEventMode);
    } // end incoming events


    if (qParallelStreams()) { // Find first successor for embedded
        findFirstSetEventIdMode(sbAtomLoadWaveOp, setEventId,  setEventMode);
    }

    //************************************************************************
    // Instruction(s)
    //************************************************************************
    const kcc_int64 numPartitions   = sbAtomLoadWaveOp->gNumPartitions();
    const kcc_int64 numBytesPerPart = sbAtomLoadWaveOp->gLength();
    const kcc_int64 addressInPart   = sbAtomLoadWaveOp->gSbAddress();
    const kcc_int64 stepSize        = sbAtomLoadWaveOp->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomLoadWaveOp->gStartAtMidPart() ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;

    dramToStateBufInstr.nbytes      = numBytesPerPart;

    for (kcc_int32 partIdx = startPart; partIdx < startPart + numPartitions; ++partIdx) {
        if (qParallelStreams()) {
            const bool first = 0 == partIdx;
            const bool last =  numPartitions-1 == partIdx;
            dramToStateBufInstr.inst_events.wait_event_idx      = 0;
            dramToStateBufInstr.inst_events.wait_event_mode     = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
            dramToStateBufInstr.inst_events.set_event_idx       = 0;
            dramToStateBufInstr.inst_events.set_event_mode      = events::eventSetMode2Isa(events::EventSetMode::DontSet);

            if (first) { // only the first reading waits for predecessors
                dramToStateBufInstr.inst_events.wait_event_idx  = waitEventId;
                dramToStateBufInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(waitEventMode);
            }
            if (last) { // only the last reading informs successors
                dramToStateBufInstr.inst_events.set_event_idx   = setEventId;
                dramToStateBufInstr.inst_events.set_event_mode  = events::eventSetMode2Isa(setEventMode);
            }
        }

        dramToStateBufInstr.src_addr = npyFileDramOffset + sbAtomLoadWaveOp->gOffsetInFile() + (partIdx * stepSize);
        dramToStateBufInstr.dst_addr = stateBuf.gEntryTongaAddress(partIdx, addressInPart);

        {
            std::ostringstream oss;
            oss << sbAtomLoadWaveOp->gOrder() << "-"
                << sbAtomLoadWaveOp->gName()  << "-" << partIdx;
            m_WaveCode.SaveName(dramToStateBufInstr, oss.str().c_str());
        }
        m_WaveCode.writeInstruction(dramToStateBufInstr);
    }

    //************************************************************************
    if (qParallelStreams()) { // Write remaining SETs
        processOutgoingEdgesAlreadyEmb(sbAtomLoadWaveOp, setEventId);
    }
}

//************************************************************************
void
WaveCodeSbAtomLoad::setInstructionEvents(compisa::SimMemCpyInstr& dramToStateBufInstr, bool first, bool last,
                    events::EventId waitEventId, events::EventWaitMode waitEventMode,
                    events::EventId setEventId, events::EventSetMode setEventMode)
{
    dramToStateBufInstr.inst_events.wait_event_idx  = 0;
    dramToStateBufInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    dramToStateBufInstr.inst_events.set_event_idx  = 0;
    dramToStateBufInstr.inst_events.set_event_mode = events::eventSetMode2Isa(events::EventSetMode::DontSet);
    if (first) { // only the first reading waits for predecessors
        dramToStateBufInstr.inst_events.wait_event_idx  = waitEventId;
        dramToStateBufInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(waitEventMode);
    }
    if (last) { // only the last reading informs successors
        dramToStateBufInstr.inst_events.set_event_idx   = setEventId;
        dramToStateBufInstr.inst_events.set_event_mode  = events::eventSetMode2Isa(setEventMode);
    }
}

//************************************************************************
void
WaveCodeSbAtomLoad::generateForSimWithRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    const EngineId engineId = sbAtomLoadWaveop->gEngineId();
    Assert(EngineId::None != engineId, "Engine id for SbAtomLoad waveop should not be None");

    const std::string& refFileName(sbAtomLoadWaveop->gRefFileName());
    const utils::DataType&    dtype(sbAtomLoadWaveop->gDataType());
    //************************************************************************
    kcc_int64 npyFileDramOffset = m_WaveCode.getDramForNpyFile(refFileName);
    if (npyFileDramOffset < 0) { // Load whole numpy file to DRAM
        compisa::SimWrNpyInstr npyToDramInstr;
        npyToDramInstr.inst_events.wait_event_idx   = 0;
        npyToDramInstr.inst_events.wait_event_mode  = eventWaitMode2Isa(events::EventWaitMode::DontWait);
        npyToDramInstr.inst_events.set_event_idx    = 0;
        npyToDramInstr.inst_events.set_event_mode   = eventSetMode2Isa(events::EventSetMode::DontSet);

        const kcc_int64 numPySize = sbAtomLoadWaveop->gLoadDataSizeInBytes();
        strcpy(npyToDramInstr.src_fname, refFileName.c_str());
        npyFileDramOffset           = m_WaveCode.gCurrentDramAddress(numPySize);

        npyToDramInstr.dst_addr     = npyFileDramOffset;
        m_WaveCode.writeInstruction(npyToDramInstr);

        WaveCode::NpyFileInfo npyFileInfo;
        npyFileInfo.m_Dirty          = false;
        npyFileInfo.m_FileDramOffset = npyFileDramOffset;
        npyFileInfo.m_SimTypeId = dtype.gSimTypeId();
        npyFileInfo.m_RefFileShape = sbAtomLoadWaveop->gRefFileShape();
        m_WaveCode.recordDramForNpyFile(refFileName, npyFileInfo);
    }

    //************************************************************************
    compisa::SimMemCpyInstr dramToStateBufInstr;
    dramToStateBufInstr.inst_events.wait_event_idx  = 0;
    dramToStateBufInstr.inst_events.wait_event_mode = eventWaitMode2Isa(events::EventWaitMode::DontWait);
    dramToStateBufInstr.inst_events.set_event_idx   = 0;
    dramToStateBufInstr.inst_events.set_event_mode  = eventSetMode2Isa(events::EventSetMode::DontSet);

    events::EventId setEventId = 0; // events::EventId_Invalid();
    events::EventSetMode setEventMode = events::EventSetMode::DontSet;
    events::EventId waitEventId = 0; // events::EventId_Invalid();
    events::EventWaitMode waitEventMode = events::EventWaitMode::DontWait;

    //************************************************************************
    // incoming events
    processIncomingEdges(sbAtomLoadWaveop, waitEventId, waitEventMode);
    // Find first successor for embedded
    findFirstSetEventIdMode(sbAtomLoadWaveop, setEventId,  setEventMode);

    //************************************************************************
    // Instruction(s)
    //************************************************************************
    const bool          qWeights            = sbAtomLoadWaveop->qContainWeights();
    const kcc_int64     numBytesPerPart     = sbAtomLoadWaveop->gLength();
    const TongaAddress  addressInPart       = sbAtomLoadWaveop->gSbAddress();
    const kcc_int64     startPart           = sbAtomLoadWaveop->gStartAtMidPart()
                                              ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;

    const kcc_uint32     replResolution      = sbAtomLoadWaveop->gIfmapReplicationResolution();
    const kcc_uint32     ifmapReplStepBytes  = sbAtomLoadWaveop->gIfmapReplicationStepBytes();
    const kcc_uint32     srcStepElem         = sbAtomLoadWaveop->gSrcStepElem();
    const kcc_uint32     numActiveParts      = sbAtomLoadWaveop->gNumPartitions();
    const kcc_uint32     ifmapReplNumRows    = sbAtomLoadWaveop->gIfmapReplicationNumRows();
    const TpbAddress     partStepBytes       = sbAtomLoadWaveop->gPartitionStepBytes();

    const TongaAddress  sbPartStep          = stateBuf.gEntryTongaAddress(1, 0) - stateBuf.gEntryTongaAddress(0, 0);

    if (qWeights) {
        Assert(1 == srcStepElem, "Load weights should have stride 1 in replication");
        Assert(numBytesPerPart == ifmapReplStepBytes, "Replication step bytes should be equal to length");
        const kcc_uint32 numChans = replResolution;

        dramToStateBufInstr.nbytes   = ifmapReplStepBytes;

        TongaAddress  fileAddress = sbAtomLoadWaveop->gOffsetInFile();
        TongaAddress  sbTongaAddress = stateBuf.gEntryTongaAddress(startPart, addressInPart);

        kcc_int32 part = startPart;
        while (part < startPart + numActiveParts) {
            for (kcc_uint32 c_idx = 0; c_idx < numChans; ++c_idx) {
                const bool first = startPart == part && 0 == c_idx;
                const bool last = startPart + numActiveParts - 1 == part && replResolution-1 == c_idx;
                setInstructionEvents(dramToStateBufInstr, first, last,
                        waitEventId, waitEventMode, setEventId, setEventMode);

                dramToStateBufInstr.src_addr = npyFileDramOffset + fileAddress + c_idx * partStepBytes;
                                                                            // + c_idx * 6272
                dramToStateBufInstr.dst_addr = sbTongaAddress;
                {
                    std::ostringstream oss;
                    oss << sbAtomLoadWaveop->gOrder()
                        << "-" << sbAtomLoadWaveop->gName()
                        << "-p" << part << "c" << c_idx;
                    m_WaveCode.SaveName(dramToStateBufInstr, oss.str().c_str());
                }
                m_WaveCode.writeInstruction(dramToStateBufInstr);


                sbTongaAddress += sbPartStep;   // += 128k
                ++part;
            }
            fileAddress += ifmapReplStepBytes;  // += 128
        }

    } else {
        const kcc_uint32 stride              = srcStepElem;
        const kcc_uint32 numChans            = replResolution / stride;
        Assert(numChans*stride == replResolution, "Src step elem(stride) must equally divide repl resolution");

        // Compute the offset within FMAP to get to odd elements (for stride=2)
        const kcc_uint32 strideNumBytes      = partStepBytes / stride;
        Assert(strideNumBytes * stride == partStepBytes, "Part step bytes not divisible by stride");

        // TODO: use the new ref_file_sz field in waveop
        const utils::TensorParams tensorParams(sbAtomLoadWaveop->gRefFileShape(),
                                               sbAtomLoadWaveop->gRefFileFormat().c_str());
        kcc_uint64 inputSize = dtype.gSizeInBytes();
        for (auto n : tensorParams) {
            inputSize *= n;
        }

        const TongaAddress    fileStartAddress  = sbAtomLoadWaveop->gOffsetInFile();
        TongaAddress    sbGroupTongaAddress = stateBuf.gEntryTongaAddress(startPart, addressInPart);

        // Compute the row offset within FMAP to get to odd rows (for stride=2)
        const kcc_uint32 fmapH        = strideNumBytes / ifmapReplStepBytes;
        const kcc_uint64 strideNumRows = ceil(float(fmapH) / stride);

        // Keep track of group count to determine even/odd row
        // TODO: to be correct for all cases, need to recompute the actual starting row 
        // (in this first layer of ResNet, starting row happens to be even for all tiles)
        kcc_int32 fileGroupCnt = 0;    
        kcc_int32 activePartCnt = startPart;
        while (activePartCnt < startPart + numActiveParts) {
            TongaAddress sbTongaAddress = sbGroupTongaAddress;
            TongaAddress fileGroupOffset = ((fileGroupCnt % stride) * strideNumRows + (fileGroupCnt / stride)) * ifmapReplStepBytes;
            TongaAddress fileGroupAddress = fileStartAddress + fileGroupOffset;

            for (kcc_uint32 strideIdx = 0; strideIdx < stride; ++strideIdx) {
                const kcc_uint64 strideOffset = strideIdx * strideNumBytes;  // {0,1}*(105340/2)
                for (kcc_uint32 c_idx = 0; c_idx < numChans; ++c_idx) {
                    const TongaAddress chanOffset = c_idx * partStepBytes;  // {0,1,2}*105340

                    const bool first = startPart == activePartCnt && 0 == strideIdx && 0 == c_idx;
                    const bool last = activePartCnt + ifmapReplNumRows >= startPart + numActiveParts
                                      && stride-1 == strideIdx && numChans-1 == c_idx;
                    setInstructionEvents(dramToStateBufInstr, first, last,
                            waitEventId, waitEventMode, setEventId, setEventMode);

                    const TongaAddress filePartAddress = fileGroupAddress + chanOffset + strideOffset;

                    if (filePartAddress < inputSize) {
                        kcc_int64 numBytesToWrite;
                        if (filePartAddress + numBytesPerPart <= inputSize) {
                            numBytesToWrite = numBytesPerPart;
                        } else {
                            numBytesToWrite = inputSize - filePartAddress;
                            if (true) {
                                std::cout << "Trimming SbAtomLoad:\n"
                                        << "    input size: " << inputSize << "\n"
                                        << "    file address: " << filePartAddress << "\n"
                                        << "    num bytes per part (requested): " << numBytesPerPart << "\n"
                                        << "    num bytes to write: " << numBytesToWrite << "\n"
                                        << "    part: " << activePartCnt << "\n"
                                        << "    stride idx: " << strideIdx << "\n"
                                        << "    chan idx: " << c_idx << "\n"
                                        ;
                                std::cout << "    Fmap tensor: ";
                                for (auto n : tensorParams) {
                                    std::cout << n << " ";
                                }
                                std::cout << "\n";
                            }
                        }
                        dramToStateBufInstr.nbytes   = numBytesToWrite;

                        dramToStateBufInstr.src_addr = npyFileDramOffset + filePartAddress;
                        dramToStateBufInstr.dst_addr = sbTongaAddress;
                        {
                            std::ostringstream oss;
                            oss << sbAtomLoadWaveop->gOrder()
                                << "-"  << sbAtomLoadWaveop->gName()
                                << "-p" << (activePartCnt + strideIdx*numChans + c_idx)
                                << "s"  << strideIdx << "c"  << c_idx;
                            m_WaveCode.SaveName(dramToStateBufInstr, oss.str().c_str());
                        }
                        m_WaveCode.writeInstruction(dramToStateBufInstr);
                    }

                    sbTongaAddress += sbPartStep;
                }
            }

            //fileGroupAddress    += ifmapReplStepBytes;              // += 230  (in the 1st layer of RN50)
            fileGroupCnt++;
            sbGroupTongaAddress += ifmapReplNumRows * sbPartStep;   // += 21*128k  (in the 1st layer of RN50)
            activePartCnt       += ifmapReplNumRows;                // += 21  (in the 1st layer of RN50)
        }
    }

    processOutgoingEdgesAlreadyEmb(sbAtomLoadWaveop, setEventId);
} // WaveCodeSbAtomLoad::generateForSimWithRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)



//************************************************************************
// Suppose an SbLoad has the following predecessors and successors
// MM1          POOL        |
//   \          /           |
//    \        /            |
//   e1\      /e2           |
//      \    /              |
//       \  /               |
//       SbLoad             |
//       /  \               |
//      /    \              |
//   e3/      \e4           |
//    /        \            |
//  MM2        ACT          |
//
//************************************************************************
// With the above wavegraph, on SIM the instructions would be
//
//  PeEng    PoolEng   ActEng   AngelEng        |
//  ...       ...      ...      ...             |
//  MM1      POOL               ...             |
//  Set e1   Set e2                             |
//           ...                Wait e1         |
//                              Wait e2         |
//           ...                MemCpy          |
//                              Set e3          |
//                              Set e4          |
//  Wait e3  ...      Wait e4   ...             |
//  MM2      ...      ACT       ...             |
//
//************************************************************************
// To execute DMA on one of the calculation engines:
// 1. Pick one of the incoming engines to execute DmaTrigger
// 2. Pick one of the outgoing engines to wait for Dma
//
//************************************************************************
// Suppose we picked PoolEng to start DMA
// PoolEng will wait for all events that Sb
//
//  PeEng         PoolEng        ActEng          |
//  ...            ...           ...            |
//  MM1           POOL                          |
//  Set e1                                      |
//                Wait e1                       |
//                DmaStart(e3,e4)               |
//                 ...                          |
//                               Wait e4        |
//  Wait e3        ...           ACT            |
//  MM2            ...           ...            |
//
// Code for PeEng and ActEng is unchanged
// All waits by SbLoad are transferred to Pool except for
// the wait Pool->SbLoad.
// Pool executes DmaStart with all events that DMA sets.
//
//
//************************************************************************

//======================================================================
kcc_int32
WaveCodeSbAtomLoad::generateForKelf(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    EngineId chosenEngId;
    std::vector<events::EventId> succEventIds;
    const kcc_int32 numSyncs = findSuccEventsAndChosenEngine(sbAtomLoadWaveop,
                                        chosenEngId, succEventIds);

    if (m_WaveCode.qBinFileSimKelf()) {
        generateDmaCopySimKelf(sbAtomLoadWaveop, chosenEngId, succEventIds);
    } else {
        if (sbAtomLoadWaveop->gIfmapReplicationResolution() == 0) {
            generateDmaDescAndTriggerRuntimeKelf(sbAtomLoadWaveop, chosenEngId, succEventIds);
        } else {
            generateDmaDescAndTriggerRuntimeKelfWithReplication(sbAtomLoadWaveop, chosenEngId, succEventIds);
        }

    }

    return numSyncs;
}

//======================================================================
void
WaveCodeSbAtomLoad::generateDmaDescAndTriggerRuntimeKelf(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop,
                    EngineId chosenEngId, const std::vector<events::EventId>& succEventIds)
{
    Assert(m_WaveCode.qBinFileRuntimeKelf(), "Must be binary for Runtime Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    //************************************************************************
    const kcc_int64 numPartitions   = sbAtomLoadWaveop->gNumPartitions();
    const kcc_int64 numBytesPerPart = sbAtomLoadWaveop->gLength();
    const kcc_int64 addressInPart   = sbAtomLoadWaveop->gSbAddress();
    const kcc_int64 stepSize        = sbAtomLoadWaveop->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomLoadWaveop->gStartAtMidPart()
                                        ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;
    const bool weights              = sbAtomLoadWaveop->qContainWeights();

    //Assert(succEventIds.size() == 1, "AtomLoad: only one succ event id");
    //************************************************************************
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());

    std::ostringstream oss;
    oss << sbAtomLoadWaveop->gOrder() << "-" << sbAtomLoadWaveop->gName();
    kelf::DmaDescription::DmaBlockToTpb& dmaBlock(
                        kelfDma.startNewDmaBlockToTpb(chosenEngId, weights, oss.str().c_str()));
    const std::string& refFileName(sbAtomLoadWaveop->gRefFileName());

    for (kcc_int32 partIdx = startPart; partIdx < startPart + numPartitions; ++partIdx) {
        const TongaAddress fileAddress = sbAtomLoadWaveop->gOffsetInFile() + (partIdx * stepSize);
        const TpbAddress sbTpbAddress = stateBuf.gEntryTpbAddress(partIdx, addressInPart);
        dmaBlock.addDmaDesc(fileAddress, refFileName, sbTpbAddress, numBytesPerPart);
    }
    for (auto eventId : succEventIds) {
        dmaBlock.addTailEventId(eventId);
    }
    //************************************************************************
    // Incoming events were processed in
    // WaveCodeSbAtomLoad::findSuccEventsAndChosenEngine(),
    // so processing them again in processIncomingEdgesForceWait is wrong.
    // The event on the chosen pred edge is replaced by sequential execution of
    // the previous waveop and TRIGGER.
    //
    // Of non-chosen pred edges, one of the Waits can be implemented as embedded.
    // The current code does not do that yet.

    if (false && qParallelStreams()) { // incoming events, adds wait for each
        events::EventId waitEventId = 0; // events::EventId_Invalid();
        events::EventWaitMode waitEventMode = events::EventWaitMode::DontWait;

        processIncomingEdgesForceWait(sbAtomLoadWaveop, chosenEngId, waitEventId, waitEventMode);
    } // end incoming events

    //************************************************************************
    addDmaBarrier(chosenEngId);
    //************************************************************************
    compisa::DmaTriggerInstr dmaTriggerInstr;
    strncpy(dmaTriggerInstr.dma_queue_name,
            dmaBlock.gSymbolicQueueName(chosenEngId).c_str(),
            ArraySizeof(dmaTriggerInstr.dma_queue_name) - 1);
    dmaTriggerInstr.use_raw_count = 0; // get from JSON
    dmaTriggerInstr.block_id = dmaBlock.gBlockId();

    dmaTriggerInstr.inst_events.wait_event_idx  = 0;
    dmaTriggerInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    dmaTriggerInstr.inst_events.set_event_idx   = 0; // succ evt is in the descriptor block
    dmaTriggerInstr.inst_events.set_event_mode  = events::eventSetMode2Isa(events::EventSetMode::DontSet);
    {
        std::ostringstream oss;
        oss << sbAtomLoadWaveop->gOrder()
            << ":" << succEventIds[0]
            << "-" << sbAtomLoadWaveop->gName();
        m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
    }
    m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
    addSecondDmaTrigger(dmaTriggerInstr, chosenEngId);
}

//======================================================================
void
WaveCodeSbAtomLoad::generateInputDma(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    if (sbAtomLoadWaveop->gIfmapReplicationResolution() == 0) {
        generateInputDmaNoRepl(sbAtomLoadWaveop);
    } else {
        generateInputDmaRepl(sbAtomLoadWaveop);
    }
}

//======================================================================
void
WaveCodeSbAtomLoad::generateInputDmaRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    Assert(m_WaveCode.qBinFileRuntimeKelf(), "Must be binary for Runtime Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    const utils::DataType& dataType(sbAtomLoadWaveop->gDataType());
    //************************************************************************
    //const kcc_int64 numPartitions   = sbAtomLoadWaveop->gNumPartitions();
    const kcc_int64 numBytesPerPart = sbAtomLoadWaveop->gLength();
    const kcc_int64 addressInPart   = sbAtomLoadWaveop->gSbAddress();
    //const kcc_int64 stepSize        = sbAtomLoadWaveop->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomLoadWaveop->gStartAtMidPart()
                                        ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;
    const kcc_int32 replResolution  = sbAtomLoadWaveop->gIfmapReplicationResolution();
    const kcc_int32 replStepElem    = sbAtomLoadWaveop->gSrcStepElem();
    const kcc_int32 numActiveParts  = sbAtomLoadWaveop->gNumPartitions();
    const kcc_int32 ifmapReplNumRows    = sbAtomLoadWaveop->gIfmapReplicationNumRows();
    const kcc_int32 ifmapReplStepBytes  = sbAtomLoadWaveop->gIfmapReplicationStepBytes();

    EngineId chosenEngId;
    std::vector<events::EventId> succEventIds;

    /*const kcc_int32 numSyncs =*/ findSuccEventsAndChosenEngine(sbAtomLoadWaveop,
                                        chosenEngId, succEventIds);

    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());

    std::ostringstream oss;
    oss << sbAtomLoadWaveop->gOrder() << "-" << sbAtomLoadWaveop->gName();
    //************************************************************************
    kelf::DmaDescription::DmaBlockInput& dmaBlock(kelfDma.startNewDmaBlockInput(oss.str().c_str()));
    {
        const TongaAddress sbPartStep = stateBuf.gEntryTpbAddress(1, 0) - stateBuf.gEntryTpbAddress(0, 0);
        const kcc_int32 stride       = replStepElem;
        const kcc_int32 numInChans    = replResolution / stride;
        Assert(numInChans*stride == replResolution,
                "Num in channels (", numInChans, ") * stride (", stride,
                ") != Replication resolution (", replResolution, ")");
        const TongaAddress  partStepBytes       = sbAtomLoadWaveop->gPartitionStepBytes();

        // Compute the offset within FMAP to get to odd elements (for stride=2)
        const kcc_uint32 strideNumBytes      = partStepBytes / stride;
        Assert(strideNumBytes * stride == partStepBytes, "Part step bytes not divisible by stride");

        // TODO: use the new ref_file_sz field in waveop
        // assume that H*W*dtypeSize == partStepBytes since when replication there is no input chan folding
        //const std::array<kcc_int32,4>& refFileShape(sbAtomLoadWaveop->gRefFileShape());
        const utils::TensorParams tensorParams(sbAtomLoadWaveop->gRefFileShape(),
                                               sbAtomLoadWaveop->gRefFileFormat().c_str());
        kcc_uint64 inputSize = dataType.gSizeInBytes();
        for (auto n : tensorParams) {
            inputSize *= n;
        }

        const TongaAddress    fileStartAddress  = sbAtomLoadWaveop->gOffsetInFile();
        TpbAddress    sbGroupTpbAddress = stateBuf.gEntryTpbAddress(startPart, addressInPart);

        // Compute the row offset within FMAP to get to odd rows (for stride=2)
        const kcc_uint64 fmapH         = strideNumBytes / ifmapReplStepBytes;
        const kcc_uint64 strideNumRows = ceil(float(fmapH) / stride);

        // Keep track of group count to determine even/odd row
        // TODO: to be correct for all cases, need to recompute the actual starting row 
        // (in this first layer of ResNet, starting row happens to be even for all tiles)
        kcc_int32 fileGroupCnt = 0;    
        kcc_int32 activePartCnt = startPart;
        while (activePartCnt < startPart + numActiveParts) {
            TpbAddress sbTpbAddress = sbGroupTpbAddress;
            TongaAddress fileGroupOffset = ((fileGroupCnt % stride) * strideNumRows + (fileGroupCnt / stride)) * ifmapReplStepBytes;
            TongaAddress fileGroupAddress = fileStartAddress + fileGroupOffset;

            for (kcc_int32 strideIdx = 0; strideIdx < stride; ++strideIdx) {
                const kcc_uint64 strideOffset = strideIdx * strideNumBytes;  // {0,1}*(105340/2)
                for (kcc_int32 c_idx = 0; c_idx < numInChans; ++c_idx) {
                    const TongaAddress chanOffset = c_idx * partStepBytes;  // {0,1,2}*105340
                    const TongaAddress filePartAddress = fileGroupAddress + chanOffset + strideOffset;
                    if (filePartAddress < inputSize) {
                        kcc_int64 numBytesToWrite;
                        if (filePartAddress + numBytesPerPart <= inputSize) {
                            numBytesToWrite = numBytesPerPart;
                        } else {
                            numBytesToWrite = inputSize - filePartAddress;
                            if (true) {
                                std::cout << "Trimming SbAtomLoad:\n"
                                        << "    input size: " << inputSize << "\n"
                                        << "    file address: " << filePartAddress << "\n"
                                        << "    num bytes per part (requested): " << numBytesPerPart << "\n"
                                        << "    num bytes to write: " << numBytesToWrite << "\n"
                                        << "    part: " << activePartCnt << "\n"
                                        << "    stride idx: " << strideIdx << "\n"
                                        << "    chan idx: " << c_idx << "\n"
                                        ;
                                std::cout << "    Fmap tensor: ";
                                for (auto n : tensorParams) {
                                    std::cout << n << " ";
                                }
                                std::cout << "\n";
                            }
                        }
                        dmaBlock.addDmaDesc(filePartAddress, sbTpbAddress, numBytesToWrite);
                    }
                    sbTpbAddress += sbPartStep;
                }
            }
            //fileGroupAddress    += ifmapReplStepBytes;              // += 230  (in the 1st layer of RN50)
            fileGroupCnt++;
            sbGroupTpbAddress += ifmapReplNumRows * sbPartStep;   // += 21*128k  (in the 1st layer of RN50)
            activePartCnt     += ifmapReplNumRows;                // += 21  (in the 1st layer of RN50)
        }
    }

    for (auto eventId : succEventIds) {
        dmaBlock.addTailEventId(eventId);
    }

    //************************************************************************
    events::EventId waitEventId = events::EventId_BeforeInputRead_PeArray();
    events::EventWaitMode waitMode = events::EventWaitMode::DontWait;
    switch (chosenEngId) {
    case EngineId::Pooling:
        // do nothing, waiting for interface already happened at the beginning of Pool
        break;
    case EngineId::PeArray:
        if (m_WaveCode.gFirstInputDMA_PeArray()) {
            m_WaveCode.rFirstInputDMA_PeArray(false);
            waitMode = events::EventWaitMode::WaitThenClear;
            waitEventId = events::EventId_BeforeInputRead_PeArray();
        }
        break;
    case EngineId::Activation:
        if (m_WaveCode.gFirstInputDMA_ActEng()) {
            m_WaveCode.rFirstInputDMA_ActEng(false);
            waitMode = events::EventWaitMode::WaitThenClear;
            waitEventId = events::EventId_BeforeInputRead_ActEng();
        }
        break;
    default:
        Assert(false, "Bad engine ID for DMA load: ", static_cast<int>(chosenEngId));
        break;
    }

    //************************************************************************
    addDmaBarrier(chosenEngId);
    compisa::DmaTriggerInstr dmaTriggerInstr;
    strncpy(dmaTriggerInstr.dma_queue_name,
            dmaBlock.gSymbolicInputQueueName().c_str(),
            ArraySizeof(dmaTriggerInstr.dma_queue_name) - 1);
    dmaTriggerInstr.use_raw_count = 0; // get from JSON
    dmaTriggerInstr.block_id = dmaBlock.gBlockId();

    dmaTriggerInstr.inst_events.wait_event_idx  = waitEventId;
    dmaTriggerInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(waitMode);

    dmaTriggerInstr.inst_events.set_event_idx   = 0; // succ evt is in the descriptor block
    dmaTriggerInstr.inst_events.set_event_mode  = events::eventSetMode2Isa(events::EventSetMode::DontSet);
    {
        std::ostringstream oss;
        oss << sbAtomLoadWaveop->gOrder()
            << ":" << succEventIds[0]
            << "-" << sbAtomLoadWaveop->gName();
        m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
    }
    m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
    addSecondDmaTrigger(dmaTriggerInstr, chosenEngId);
} // WaveCodeSbAtomLoad::generateInputDmaRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)


//======================================================================
void
WaveCodeSbAtomLoad::generateInputDmaNoRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    Assert(m_WaveCode.qBinFileRuntimeKelf(), "Must be binary for Runtime Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    //************************************************************************
    const kcc_int64 numPartitions   = sbAtomLoadWaveop->gNumPartitions();
    const kcc_int64 numBytesPerPart = sbAtomLoadWaveop->gLength();
    const kcc_int64 addressInPart   = sbAtomLoadWaveop->gSbAddress();
    const kcc_int64 stepSize        = sbAtomLoadWaveop->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomLoadWaveop->gStartAtMidPart()
                                        ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;
    EngineId chosenEngId;
    std::vector<events::EventId> succEventIds;

    /*const kcc_int32 numSyncs =*/ findSuccEventsAndChosenEngine(sbAtomLoadWaveop,
                                        chosenEngId, succEventIds);
    Assert(succEventIds.size() > 0, "No successor events found for AtomLoad '",
            sbAtomLoadWaveop->gName(), "'");

    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());

    std::ostringstream oss;
    oss << sbAtomLoadWaveop->gOrder() << "-" << sbAtomLoadWaveop->gName();
    //************************************************************************
    kelf::DmaDescription::DmaBlockInput& dmaBlock(kelfDma.startNewDmaBlockInput(oss.str().c_str()));

    for (kcc_int32 partIdx = startPart; partIdx < startPart + numPartitions; ++partIdx) {
        const kcc_uint64 fileAddress = sbAtomLoadWaveop->gOffsetInFile() + (partIdx * stepSize);
        const TpbAddress sbAddress = stateBuf.gEntryTpbAddress(partIdx, addressInPart);
        dmaBlock.addDmaDesc(fileAddress, sbAddress, numBytesPerPart);
    }
    for (auto eventId : succEventIds) {
        dmaBlock.addTailEventId(eventId);
    }

    //************************************************************************
    events::EventId waitEventId = 0; // does not matter which one, but
    events::EventWaitMode waitMode = events::EventWaitMode::DontWait;
    switch (chosenEngId) {
    case EngineId::Pooling:
        // do nothing, waiting for interface already happened at the beginning of Pool
        break;
    case EngineId::PeArray:
        if (m_WaveCode.gFirstInputDMA_PeArray()) {
            m_WaveCode.rFirstInputDMA_PeArray(false);
            waitMode = events::EventWaitMode::WaitThenClear;
            waitEventId = events::EventId_BeforeInputRead_PeArray();
        }
        break;
    case EngineId::Activation:
        if (m_WaveCode.gFirstInputDMA_PeArray()) {
            m_WaveCode.rFirstInputDMA_PeArray(false);
            waitMode = events::EventWaitMode::WaitThenClear;
            waitEventId = events::EventId_BeforeInputRead_ActEng();
        }
        break;
    default:
        Assert(false, "Bad engine ID for DMA load: ", static_cast<int>(chosenEngId));
        break;
    }

    //************************************************************************
    addDmaBarrier(chosenEngId);
    compisa::DmaTriggerInstr dmaTriggerInstr;
    strncpy(dmaTriggerInstr.dma_queue_name,
            dmaBlock.gSymbolicInputQueueName().c_str(),
            ArraySizeof(dmaTriggerInstr.dma_queue_name) - 1);
    dmaTriggerInstr.use_raw_count = 0; // get from JSON
    dmaTriggerInstr.block_id = dmaBlock.gBlockId();

    dmaTriggerInstr.inst_events.wait_event_idx  = waitEventId;
    dmaTriggerInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(waitMode);

    dmaTriggerInstr.inst_events.set_event_idx   = 0; // succ evt in the descr. block
    dmaTriggerInstr.inst_events.set_event_mode  = events::eventSetMode2Isa(events::EventSetMode::DontSet);
    {
        std::ostringstream oss;
        oss << sbAtomLoadWaveop->gOrder()
            << ":" << succEventIds[0]
            << "-" << sbAtomLoadWaveop->gName();
        m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
    }
    m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
    addSecondDmaTrigger(dmaTriggerInstr, chosenEngId);
} // WaveCodeSbAtomLoad::generateInputDmaNoRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)

//======================================================================
// If ifmap_replication_resolution = 2 * 3 (stride * C, where C is # channels), and contains_weights=False, and src_step_elem=2 (stride)
// (only applicable when contains_weights=False), perform:
// 1- sim_dma_copy 3 channels (C) starting at offset_in_file to sb_address (src_step_elem=2 indicates even indices)
// 2- sim_dma_copy 3 channels (C) starting at offset_in_file + 2 bytes (2 bytes for 1 float16 element), to sb_address for channels following #1 (odd indices)
// 3a- increment sb_address by ifmap_replication_num_rows * SB partition size (ifmap_replication_num_rows is 21 for ResNet50 case)
// 3b- increment offset_in_file by ifmap_replication_step_bytes (W *stride)
// 4- if number of channels copied so far < ifmap_count, goto #1
// (we can generalize this better; in particular, stop as soon as number of channels copied so far >= ifmap_count which is number of active rows)
// (2bytes indicates data item size for float16)
//======================================================================
//
// If ifmap_replication_resolution= 3 (C, maybe assert), and contains_weights=True (src_step_elem should be 1 here, but not used, weights do not have stride), perform:
// 1- sim_dma_copy ifmap_replication_resolution channels (C) starting at offset_in_file to sb_address
// 2a- increment sb_address by ifmap_replication_resolution * SB partition size (ifmaps_replication_num_rows is 3 also, different from above, so can also
//     use it instead of ifmap_replication_resolution)
// 2b- increment offset_in_file by ifmap_replication_step_bytes (equals M, maybe assert)
// 3-  if number of channels copied so far < ifmap_count, goto #1
//
// The test for this is 3-1conv0_padvalid_wave.
//======================================================================
// In the previous comment, please ignore ifmap_replication_step_bytes for MatMul waveop.
// Also, for weights SBAtomLoad waveop, ifmap_replication_resolution is different from that in IFMAP SBAtomLoad waveop (3 instead of 6 for ResNet50 first layer).
// Also, step #1 should say:
// 1- sim_dma_copy ifmap_replication_resolution channels starting at offset_in_file to sb_address
//======================================================================


void
WaveCodeSbAtomLoad::generateDmaDescAndTriggerRuntimeKelfWithReplication(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop,
                    EngineId chosenEngId, const std::vector<events::EventId>& succEventIds)
{
    Assert(succEventIds.size() == 1, "AtomLoad: only one succ event id: ", sbAtomLoadWaveop->gName());
    Assert(m_WaveCode.qBinFileRuntimeKelf(), "Must be binary for Runtime Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    //const utils::DataType& dataType(sbAtomLoadWaveop->gDataType());
    //const kcc_int32 dtypeSize = dataType.gSizeInBytes();

   //************************************************************************
    const kcc_int64 addressInPart       = sbAtomLoadWaveop->gSbAddress();
    const kcc_int64 startPart           = sbAtomLoadWaveop->gStartAtMidPart()
                                            ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;
    const bool      qWeights            = sbAtomLoadWaveop->qContainWeights();

    const kcc_int32 replResolution      = sbAtomLoadWaveop->gIfmapReplicationResolution();
    const kcc_int32 ifmapReplStepBytes  = sbAtomLoadWaveop->gIfmapReplicationStepBytes();
    const kcc_int32 replStepElem        = sbAtomLoadWaveop->gSrcStepElem();
    const kcc_int32 numActiveParts      = sbAtomLoadWaveop->gNumPartitions();
    const kcc_int32 ifmapReplNumRows    = sbAtomLoadWaveop->gIfmapReplicationNumRows();
    const kcc_int64 numBytesPerPart     = sbAtomLoadWaveop->gLength();

    const std::string& refFileName(sbAtomLoadWaveop->gRefFileName());
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());

    std::ostringstream oss;
    oss << sbAtomLoadWaveop->gOrder() << "-" << sbAtomLoadWaveop->gName();
    kelf::DmaDescription::DmaBlockToTpb& dmaBlock(kelfDma.startNewDmaBlockToTpb(
                                                    chosenEngId, qWeights, oss.str().c_str()));
    if (qWeights) {
        Assert(1 == replStepElem, "Load weights should have stride 1 in replication");
        Assert(numBytesPerPart == ifmapReplStepBytes, "Replication step bytes should be equal to length");
        const kcc_uint32 numChans    = replResolution;
        const TpbAddress  partStepBytes       = sbAtomLoadWaveop->gPartitionStepBytes();
        const TpbAddress  sbPartStep = stateBuf.gEntryTpbAddress(1, 0) - stateBuf.gEntryTpbAddress(0, 0);

        TongaAddress fileAddress     = sbAtomLoadWaveop->gOffsetInFile();
        TpbAddress   sbTpbAddress    = stateBuf.gEntryTpbAddress(startPart, addressInPart);

        kcc_int32 part = startPart;
        while (part < startPart + numActiveParts) {
            for (kcc_uint32 c_idx = 0; c_idx < numChans; ++c_idx) {
                const TongaAddress currFileAddress = fileAddress + c_idx * partStepBytes;
                dmaBlock.addDmaDesc(currFileAddress, refFileName, sbTpbAddress, ifmapReplStepBytes);

                sbTpbAddress += sbPartStep;   // += 128k
                ++part;
            }
            fileAddress += ifmapReplStepBytes;  // += 128
        }


    } else {
        const TpbAddress    sbPartStep = stateBuf.gEntryTpbAddress(1, addressInPart) - stateBuf.gEntryTpbAddress(0, addressInPart);
        const kcc_int32 numInChans = replResolution / replStepElem;
        Assert(numInChans*replStepElem == replResolution,
                "Num in channels (", numInChans, ") * stride (", replStepElem,
                ") != Replication resolution (", replResolution, ")");

        const kcc_int32     numBytesPerPart     = sbAtomLoadWaveop->gLength();
        const TongaAddress  partStepBytes       = sbAtomLoadWaveop->gPartitionStepBytes();
        TongaAddress        fileGroupAddress    = sbAtomLoadWaveop->gOffsetInFile();
        TpbAddress          sbGroupTpbAddress   = stateBuf.gEntryTpbAddress(startPart, addressInPart);

        // assume that H*W*dtypeSize == partStepBytes since when replication there is no input chan folding
        kcc_int32 activePartCnt = 0;
        while (activePartCnt < numActiveParts) {
            TpbAddress sbTpbAddress = sbGroupTpbAddress;
            TongaAddress fileAddress = fileGroupAddress;
            for (kcc_int32 strideIdx = 0; strideIdx < replStepElem; ++strideIdx) {
                const TongaAddress strideFileOffset = (partStepBytes/replStepElem) * strideIdx;
                for (kcc_int32 chanIdx = 0; chanIdx < numInChans; ++chanIdx) {
                    TongaAddress chanOffset = chanIdx * partStepBytes;
                    TongaAddress filePartAddress = fileAddress + chanOffset + strideFileOffset;
                    dmaBlock.addDmaDesc(filePartAddress, refFileName, sbTpbAddress, numBytesPerPart);
                    sbTpbAddress += sbPartStep;
                }
            }
            activePartCnt       += ifmapReplNumRows;
            fileGroupAddress    += ifmapReplStepBytes;
            sbGroupTpbAddress   += ifmapReplNumRows * sbPartStep;
        }
    }

    for (auto eventId : succEventIds) {
        dmaBlock.addTailEventId(eventId);
    }


    //************************************************************************
    addDmaBarrier(chosenEngId);
    //************************************************************************
    compisa::DmaTriggerInstr dmaTriggerInstr;
    strncpy(dmaTriggerInstr.dma_queue_name,
            dmaBlock.gSymbolicQueueName(chosenEngId).c_str(),
            ArraySizeof(dmaTriggerInstr.dma_queue_name) - 1);
    dmaTriggerInstr.use_raw_count = 0; // get from JSON
    dmaTriggerInstr.block_id = dmaBlock.gBlockId();

    dmaTriggerInstr.inst_events.wait_event_idx  = 0;
    dmaTriggerInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    dmaTriggerInstr.inst_events.set_event_idx   = 0; // succ evt is in the descriptor block
    dmaTriggerInstr.inst_events.set_event_mode  = events::eventSetMode2Isa(events::EventSetMode::DontSet);
    {
        std::ostringstream oss;
        oss << sbAtomLoadWaveop->gOrder() << ":" << succEventIds[0] << "-" << sbAtomLoadWaveop->gName();
        m_WaveCode.SaveName(dmaTriggerInstr, oss.str().c_str());
    }
    m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
    addSecondDmaTrigger(dmaTriggerInstr, chosenEngId);
} // WaveCodeSbAtomLoad::generateDmaDescAndTriggerRuntimeKelfWithReplication(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop,








//======================================================================
void
WaveCodeSbAtomLoad::generateDmaCopySimKelf(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop,
                    EngineId chosenEngId, const std::vector<events::EventId>& succEventIds)
{
    Assert(m_WaveCode.qBinFileSimKelf(), "Must be binary for SIM Kelf");
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());

    const kcc_int64 numPartitions   = sbAtomLoadWaveop->gNumPartitions();
    const kcc_int64 numBytesPerPart = sbAtomLoadWaveop->gLength();
    const kcc_int64 addressInPart   = sbAtomLoadWaveop->gSbAddress();
    const kcc_int64 stepSize        = sbAtomLoadWaveop->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomLoadWaveop->gStartAtMidPart() ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;

    //************************************************************************

    kcc_int64 npyFileDramOffset = m_WaveCode.getDramForNpyFile(sbAtomLoadWaveop->gRefFileName());
    if (npyFileDramOffset < 0) { // Load whole numpy file to DRAM
        compisa::SimWrNpyInstr npyToDramInstr;
        npyToDramInstr.inst_events.wait_event_idx   = 0;
        npyToDramInstr.inst_events.wait_event_mode  = eventWaitMode2Isa(events::EventWaitMode::DontWait);
        npyToDramInstr.inst_events.set_event_idx    = 0;
        npyToDramInstr.inst_events.set_event_mode   = eventSetMode2Isa(events::EventSetMode::DontSet);

        const kcc_int64 numPySize = sbAtomLoadWaveop->gLoadDataSizeInBytes();
        strcpy(npyToDramInstr.src_fname, sbAtomLoadWaveop->gRefFileName().c_str());
        npyFileDramOffset           = m_WaveCode.gCurrentDramAddress(numPySize);

        npyToDramInstr.dst_addr     = npyFileDramOffset;
        m_WaveCode.writeInstruction(npyToDramInstr);

        WaveCode::NpyFileInfo npyFileInfo;
        npyFileInfo.m_Dirty          = false;
        npyFileInfo.m_FileDramOffset = npyFileDramOffset;
        npyFileInfo.m_SimTypeId = sbAtomLoadWaveop->gDataType().gSimTypeId();
        npyFileInfo.m_RefFileShape = sbAtomLoadWaveop->gRefFileShape();
        m_WaveCode.recordDramForNpyFile(sbAtomLoadWaveop->gRefFileName(), npyFileInfo);
    }


    compisa::SimDmaCopyInstr simDmaCopyInstr;
    simDmaCopyInstr.inst_events.wait_event_idx  = 0;
    simDmaCopyInstr.inst_events.wait_event_mode = eventWaitMode2Isa(events::EventWaitMode::DontWait);
    simDmaCopyInstr.inst_events.set_event_idx   = 0;
    simDmaCopyInstr.inst_events.set_event_mode  = eventSetMode2Isa(events::EventSetMode::DontSet);

    events::EventId setEventId = 0; // events::EventId_Invalid();
    events::EventSetMode setEventMode = events::EventSetMode::DontSet;
    events::EventId waitEventId = 0; // events::EventId_Invalid();
    events::EventWaitMode waitEventMode = events::EventWaitMode::DontWait;

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(sbAtomLoadWaveop, waitEventId, waitEventMode);
    } // end incoming events

    simDmaCopyInstr.inst_events.wait_event_idx  = waitEventId;
    simDmaCopyInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(waitEventMode);

    if (qParallelStreams()) { // Find first successor for embedded
        findFirstSetEventIdMode(sbAtomLoadWaveop, setEventId,  setEventMode);
    }

    //************************************************************************
    const kcc_int64 fileAddress = npyFileDramOffset + sbAtomLoadWaveop->gOffsetInFile() + (startPart * stepSize);
    const TongaAddress sbStartTongaAddress = stateBuf.gEntryTongaAddress(startPart, addressInPart);

    // DRAM
    simDmaCopyInstr.src_start_addr   = fileAddress;
    AssignWithSizeCheck(simDmaCopyInstr.src_num_elem[0], numPartitions * numBytesPerPart);
    AssignWithSizeCheck(simDmaCopyInstr.src_step_elem[0], 1);
    AssignWithSizeCheck(simDmaCopyInstr.src_num_elem[1], 1);
    AssignWithSizeCheck(simDmaCopyInstr.src_step_elem[1], 0);

    // SB
    simDmaCopyInstr.dst_start_addr   = sbStartTongaAddress;
    AssignWithSizeCheck(simDmaCopyInstr.dst_num_elem[0], numBytesPerPart);
    AssignWithSizeCheck(simDmaCopyInstr.dst_step_elem[0], 1);
    AssignWithSizeCheck(simDmaCopyInstr.dst_num_elem[1], numPartitions);
    AssignWithSizeCheck(simDmaCopyInstr.dst_step_elem[1], stateBuf.gEntryTongaAddress(1, addressInPart) - stateBuf.gEntryTongaAddress(0, addressInPart));

    // Should we assert that size <= 1?
    if (succEventIds.size() > 0) {
        simDmaCopyInstr.queue_idx        = succEventIds[0];
    } else {
        simDmaCopyInstr.queue_idx    = 0;
    }

    {
        std::ostringstream oss;
        oss << sbAtomLoadWaveop->gOrder()
            << ":" << succEventIds[0]
            << "-" << sbAtomLoadWaveop->gName();
        m_WaveCode.SaveName(simDmaCopyInstr, oss.str().c_str());
    }
    m_WaveCode.writeInstruction(simDmaCopyInstr, chosenEngId);
}

//======================================================================
kcc_int32
WaveCodeSbAtomLoad::findSuccEventsAndChosenEngine(wave::SbAtomLoadWaveOp* sbAtomWaveop,
                        EngineId& chosenEngId,
                        std::vector<events::EventId>& succEventIds)
{
    chosenEngId = sbAtomWaveop->gEngineId();
    Assert(chosenEngId != EngineId::None, "None engine in waveop ", sbAtomWaveop->gName());
    kcc_int32 numSyncs = 0;
    wave::WaveEdge* chosenPrevEdge = nullptr;

    for (auto prevWaveEdge : sbAtomWaveop->gPrevWaveEdges()) {
        if (prevWaveEdge->qChosenForSuccSbAtom()) {
            chosenPrevEdge = prevWaveEdge;
            break;
        }
    }
    if (chosenPrevEdge) {
        Assert(chosenEngId == chosenPrevEdge->gFromOp()->gEngineId(),
            "Engine on chosen edge from ", chosenPrevEdge->gFromOp()->gName(), " to ", sbAtomWaveop->gName(),
            " different than engine id ", utils::engineId2Str(chosenEngId));
    }

    // First wait on all other engines
    for (auto prevWaveEdge : sbAtomWaveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToImplementSync()) {
            continue;
        }
        /*
        */
        if (prevWaveEdge == chosenPrevEdge) {
            continue;
        }
        ++numSyncs;
        m_WaveCode.writeWaitOrWaitClearInstr(prevWaveEdge, chosenEngId);
    }

    for (auto succWaveEdge : sbAtomWaveop->gSuccWaveEdges()) {
        if (succWaveEdge->qNeedToImplementSync()) {
            succEventIds.push_back(succWaveEdge->gEventId());
            ++numSyncs;
        }
    }
    return numSyncs;
}


/***********************************************************************
***********************************************************************/
void
WaveCodeSbAtomLoad::calcInputSize(const wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    if (sbAtomLoadWaveop->qContainWeights()) { // ifmap
        return;
    }
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());
    if (kelfDma.gInputSizeBytes() > 0) { // only look at first non-weight load, other
        return;                          // loads could be from some intermediate saves
    }
    const utils::DataType&    dtype(sbAtomLoadWaveop->gDataType());
    const std::array<kcc_int32,4>& shape(sbAtomLoadWaveop->gRefFileShape ());
    kcc_int64 sz = dtype.gSizeInBytes();
    for (auto n : shape) {
        sz *= n;
    }
    kelfDma.rInputSizeBytes(sz);
}

}}

