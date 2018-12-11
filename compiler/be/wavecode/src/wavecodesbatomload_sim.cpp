#include "utils/inc/debug.hpp"
#include "utils/inc/misc.hpp"
#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"

#include "arch/inc/statebuffer.hpp"
#include "arch/inc/arch.hpp"

#include "compisa/inc/compisasimwrnpy.hpp"
#include "compisa/inc/compisasimmemcpy.hpp"
#include "compisa/inc/compisasimdmacopy.hpp"


#include "events/inc/events.hpp"



#include "wave/inc/waveedge.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodesbatomload_sim.hpp"

namespace kcc {
namespace wavecode {


//************************************************************************
WaveCodeSbAtomLoadSim::WaveCodeSbAtomLoadSim(WaveCodeRef waveCode)
    : WaveCodeSbAtomLoad(waveCode)
{}


//************************************************************************
void
WaveCodeSbAtomLoadSim::generate(wave::WaveOp* waveOp)
{
    Assert(!qGenerateKelf(), "Must be in Sim mode to generate Load for sim");
    const auto sbAtomLoadWaveop = dynamic_cast<wave::SbAtomLoadWaveOp*>(waveOp);
    assert(sbAtomLoadWaveop);


    calcInputSize(sbAtomLoadWaveop);
    generateForSim(sbAtomLoadWaveop);
}



//************************************************************************
void
WaveCodeSbAtomLoadSim::generateForSim(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
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
WaveCodeSbAtomLoadSim::generateForSimNoRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    const EngineId engineId = sbAtomLoadWaveop->gEngineId();
    Assert(EngineId::None != engineId, "Engine id for SbAtomLoad waveop should not be None");

    //************************************************************************
    kcc_int64 npyFileDramOffset = m_WaveCode.getDramForNpyFile(sbAtomLoadWaveop->gRefFileName());
    if (npyFileDramOffset < 0) { // Load whole numpy file to DRAM
        compisa::SimWrNpyInstr simNpyToDramInstr;
        AssignWithSizeCheck(simNpyToDramInstr.inst_events.wait_event_idx, 0);
        AssignWithSizeCheck(simNpyToDramInstr.inst_events.wait_event_mode, eventWaitMode2Isa(events::EventWaitMode::DontWait));
        AssignWithSizeCheck(simNpyToDramInstr.inst_events.set_event_idx, 0);
        AssignWithSizeCheck(simNpyToDramInstr.inst_events.set_event_mode, eventSetMode2Isa(events::EventSetMode::DontSet));

        const kcc_int64 numPySize = sbAtomLoadWaveop->gLoadDataSizeInBytes();
        strcpy(simNpyToDramInstr.src_fname, sbAtomLoadWaveop->gRefFileName().c_str());
        npyFileDramOffset           = m_WaveCode.gCurrentDramAddress(numPySize);

        AssignWithSizeCheck(simNpyToDramInstr.dst_addr, npyFileDramOffset);
        m_WaveCode.writeInstruction(simNpyToDramInstr);

        WaveCode::NpyFileInfo npyFileInfo;
        npyFileInfo.m_Dirty          = false;
        npyFileInfo.m_FileDramOffset = npyFileDramOffset;
        npyFileInfo.m_SimTypeId = sbAtomLoadWaveop->gDataType().gSimTypeId();
        npyFileInfo.m_RefFileShape = sbAtomLoadWaveop->gRefFileShape();
        m_WaveCode.recordDramForNpyFile(sbAtomLoadWaveop->gRefFileName(), npyFileInfo);
    }

    //************************************************************************
    compisa::SimMemCpyInstr simDramToStateBufInstr;
    AssignWithSizeCheck(simDramToStateBufInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(simDramToStateBufInstr.inst_events.wait_event_mode, eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(simDramToStateBufInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(simDramToStateBufInstr.inst_events.set_event_mode, eventSetMode2Isa(events::EventSetMode::DontSet));

    events::EventId setEventId = 0; // events::EventId_Invalid();
    events::EventSetMode setEventMode = events::EventSetMode::DontSet;
    events::EventId waitEventId = 0; // events::EventId_Invalid();
    events::EventWaitMode waitEventMode = events::EventWaitMode::DontWait;

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(sbAtomLoadWaveop, waitEventId, waitEventMode);
    } // end incoming events


    if (qParallelStreams()) { // Find first successor for embedded
        findFirstSetEventIdMode(sbAtomLoadWaveop, setEventId,  setEventMode);
    }

    //************************************************************************
    // Instruction(s)
    //************************************************************************
    const kcc_int64 numPartitions   = sbAtomLoadWaveop->gNumPartitions();
    const kcc_int64 numBytesPerPart = sbAtomLoadWaveop->gLength();
    const kcc_int64 addressInPart   = sbAtomLoadWaveop->gSbAddress();
    const kcc_int64 stepSize        = sbAtomLoadWaveop->gPartitionStepBytes();
    const kcc_int64 startPart       = sbAtomLoadWaveop->gStartAtMidPart() ? arch::Arch::gArch().gNumberPeArrayRows()/2 : 0;

    AssignWithSizeCheck(simDramToStateBufInstr.nbytes, numBytesPerPart);

    for (kcc_int32 partIdx = startPart; partIdx < startPart + numPartitions; ++partIdx) {
        if (qParallelStreams()) {
            const bool first = 0 == partIdx;
            const bool last =  numPartitions-1 == partIdx;
            AssignWithSizeCheck(simDramToStateBufInstr.inst_events.wait_event_idx, 0);
            AssignWithSizeCheck(simDramToStateBufInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
            AssignWithSizeCheck(simDramToStateBufInstr.inst_events.set_event_idx, 0);
            AssignWithSizeCheck(simDramToStateBufInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

            if (first) { // only the first reading waits for predecessors
            AssignWithSizeCheck(simDramToStateBufInstr.inst_events.wait_event_idx, waitEventId);
            AssignWithSizeCheck(simDramToStateBufInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(waitEventMode));
            }
            if (last) { // only the last reading informs successors
            AssignWithSizeCheck(simDramToStateBufInstr.inst_events.set_event_idx, setEventId);
            AssignWithSizeCheck(simDramToStateBufInstr.inst_events.set_event_mode, events::eventSetMode2Isa(setEventMode));
            }
        }

            AssignWithSizeCheck(simDramToStateBufInstr.src_addr, npyFileDramOffset + sbAtomLoadWaveop->gOffsetInFile() + (partIdx * stepSize));
            AssignWithSizeCheck(simDramToStateBufInstr.dst_addr, stateBuf.gEntryTongaAddress(partIdx, addressInPart));

        {
            std::ostringstream oss;
            oss << sbAtomLoadWaveop->gOrder() << "-"
                << sbAtomLoadWaveop->gName()  << "-" << partIdx;
            m_WaveCode.SaveName(simDramToStateBufInstr, oss.str().c_str());
        }
        m_WaveCode.writeInstruction(simDramToStateBufInstr);
    }

    //************************************************************************
    if (qParallelStreams()) { // Write remaining SETs
        processOutgoingEdgesAlreadyEmb(sbAtomLoadWaveop, setEventId);
    }
}

//************************************************************************
void
WaveCodeSbAtomLoadSim::setInstructionEvents(compisa::SimMemCpyInstr& simDramToStateBufInstr, bool first, bool last,
                    events::EventId waitEventId, events::EventWaitMode waitEventMode,
                    events::EventId setEventId, events::EventSetMode setEventMode)
{
    AssignWithSizeCheck(simDramToStateBufInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(simDramToStateBufInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(simDramToStateBufInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(simDramToStateBufInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));
    if (first) { // only the first reading waits for predecessors
        AssignWithSizeCheck(simDramToStateBufInstr.inst_events.wait_event_idx, waitEventId);
        AssignWithSizeCheck(simDramToStateBufInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(waitEventMode));
    }
    if (last) { // only the last reading informs successors
        AssignWithSizeCheck(simDramToStateBufInstr.inst_events.set_event_idx, setEventId);
        AssignWithSizeCheck(simDramToStateBufInstr.inst_events.set_event_mode, events::eventSetMode2Isa(setEventMode));
    }
}

//************************************************************************
void
WaveCodeSbAtomLoadSim::generateForSimWithRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)
{
    const arch::StateBuffer& stateBuf(arch::Arch::gArch().gStateBuffer());
    const EngineId engineId = sbAtomLoadWaveop->gEngineId();
    Assert(EngineId::None != engineId, "Engine id for SbAtomLoad waveop should not be None");

    const std::string& refFileName(sbAtomLoadWaveop->gRefFileName());
    const utils::DataType&    dtype(sbAtomLoadWaveop->gDataType());
    //************************************************************************
    kcc_int64 npyFileDramOffset = m_WaveCode.getDramForNpyFile(refFileName);
    if (npyFileDramOffset < 0) { // Load whole numpy file to DRAM
        compisa::SimWrNpyInstr simNpyToDramInstr;
        AssignWithSizeCheck(simNpyToDramInstr.inst_events.wait_event_idx, 0);
        AssignWithSizeCheck(simNpyToDramInstr.inst_events.wait_event_mode, eventWaitMode2Isa(events::EventWaitMode::DontWait));
        AssignWithSizeCheck(simNpyToDramInstr.inst_events.set_event_idx, 0);
        AssignWithSizeCheck(simNpyToDramInstr.inst_events.set_event_mode, eventSetMode2Isa(events::EventSetMode::DontSet));

        const kcc_int64 numPySize = sbAtomLoadWaveop->gLoadDataSizeInBytes();
        strcpy(simNpyToDramInstr.src_fname, refFileName.c_str());
        npyFileDramOffset           = m_WaveCode.gCurrentDramAddress(numPySize);

        AssignWithSizeCheck(simNpyToDramInstr.dst_addr, npyFileDramOffset);
        m_WaveCode.writeInstruction(simNpyToDramInstr);

        WaveCode::NpyFileInfo npyFileInfo;
        npyFileInfo.m_Dirty          = false;
        npyFileInfo.m_FileDramOffset = npyFileDramOffset;
        npyFileInfo.m_SimTypeId = dtype.gSimTypeId();
        npyFileInfo.m_RefFileShape = sbAtomLoadWaveop->gRefFileShape();
        m_WaveCode.recordDramForNpyFile(refFileName, npyFileInfo);
    }

    //************************************************************************
    compisa::SimMemCpyInstr simDramToStateBufInstr;
    AssignWithSizeCheck(simDramToStateBufInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(simDramToStateBufInstr.inst_events.wait_event_mode, eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(simDramToStateBufInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(simDramToStateBufInstr.inst_events.set_event_mode, eventSetMode2Isa(events::EventSetMode::DontSet));

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

        AssignWithSizeCheck(simDramToStateBufInstr.nbytes, ifmapReplStepBytes);

        TongaAddress  fileAddress = sbAtomLoadWaveop->gOffsetInFile();
        TongaAddress  sbTongaAddress = stateBuf.gEntryTongaAddress(startPart, addressInPart);

        kcc_int32 part = startPart;
        while (part < startPart + numActiveParts) {
            for (kcc_uint32 c_idx = 0; c_idx < numChans; ++c_idx) {
                const bool first = startPart == part && 0 == c_idx;
                const bool last = startPart + numActiveParts - 1 == part && replResolution-1 == c_idx;
                setInstructionEvents(simDramToStateBufInstr, first, last,
                        waitEventId, waitEventMode, setEventId, setEventMode);

                AssignWithSizeCheck(simDramToStateBufInstr.src_addr, npyFileDramOffset + fileAddress + c_idx * partStepBytes);
                                                                            // + c_idx * 6272
                AssignWithSizeCheck(simDramToStateBufInstr.dst_addr, sbTongaAddress);
                {
                    std::ostringstream oss;
                    oss << sbAtomLoadWaveop->gOrder()
                        << "-" << sbAtomLoadWaveop->gName()
                        << "-p" << part << "c" << c_idx;
                    m_WaveCode.SaveName(simDramToStateBufInstr, oss.str().c_str());
                }
                m_WaveCode.writeInstruction(simDramToStateBufInstr);


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
                    setInstructionEvents(simDramToStateBufInstr, first, last,
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
                        AssignWithSizeCheck(simDramToStateBufInstr.nbytes, numBytesToWrite);

                        AssignWithSizeCheck(simDramToStateBufInstr.src_addr, npyFileDramOffset + filePartAddress);
                        AssignWithSizeCheck(simDramToStateBufInstr.dst_addr, sbTongaAddress);
                        {
                            std::ostringstream oss;
                            oss << sbAtomLoadWaveop->gOrder()
                                << "-"  << sbAtomLoadWaveop->gName()
                                << "-p" << (activePartCnt + strideIdx*numChans + c_idx)
                                << "s"  << strideIdx << "c"  << c_idx;
                            m_WaveCode.SaveName(simDramToStateBufInstr, oss.str().c_str());
                        }
                        m_WaveCode.writeInstruction(simDramToStateBufInstr);
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
} // WaveCodeSbAtomLoadSim::generateForSimWithRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveop)



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



/***********************************************************************
***********************************************************************/

}}

