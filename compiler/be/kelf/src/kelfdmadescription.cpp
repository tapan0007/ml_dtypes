#include <fstream>
#include <iostream>


// Json package
#include "nlohmann/json.hpp"



#include "utils/inc/types.hpp"
#include "utils/inc/version.hpp"
#include "utils/inc/asserter.hpp"
#include "utils/inc/misc.hpp"
#include "utils/inc/debug.hpp"

#include "nets/inc/network.hpp"
#include "events/inc/eventmgr.hpp"
#include "kelf/inc/kelfdmadescription.hpp"

namespace kcc {
namespace kelf {

using json = nlohmann::json;

static const char* const singleInputName = "IN";


/***********************************************************************
***********************************************************************/
DmaDescription::DmaDescription(const nets::Network& network, bool useSem)
    : m_Network(network)
    , m_UseSemaphore(useSem)
{
}


/***********************************************************************
***********************************************************************/
kcc_int32
DmaDescription::gBlockIdForQueue(const dma::DmaQueue* que)
{
    kcc_int32 blockId;
    const auto it = m_QueueToBlockId.find(que);
    if (it == m_QueueToBlockId.end()) {
        blockId = 0;
    } else {
        blockId = (*it).second;
    }
    m_QueueToBlockId[que] = blockId + 1;
    return blockId;
}

/***********************************************************************
***********************************************************************/
kcc_int32
DmaDescription::gNumBlockIdsForQueue(const dma::DmaQueue* que) const
{
    const auto it = m_QueueToBlockId.find(que);
    if (it == m_QueueToBlockId.end()) {
        return 0;
    } else {
        return (*it).second;
    }
}


/***********************************************************************
***********************************************************************/
kcc_int32
DmaDescription::startNewDmaBlockToTpb(const dma::DmaQueue* que, EngineId engId, bool qWeights, const char* comment)
{
    DmaBlockToTpb block(this, que, engId, qWeights, comment);
    m_DmaBlocksToTpb.push_back(block);
    return m_DmaBlocksToTpb.size()-1;
}


/***********************************************************************
***********************************************************************/
kcc_int32
DmaDescription::startNewDmaBlockFromTpb(const dma::DmaQueue* que, EngineId engId, bool qOut, const char* comment)
{
    DmaBlockFromTpb block(this, que, engId, qOut, comment);
    m_DmaBlocksFromTpb.push_back(block);
    return m_DmaBlocksFromTpb.size()-1;
}

/***********************************************************************
***********************************************************************/
kcc_int32
DmaDescription::startNewDmaBlockInput(const dma::DmaQueue* que, EngineId engId,  const char* comment)
{
    DmaBlockInput block (this, que, engId, comment);
    m_DmaBlocksInput.push_back(block);
    return m_DmaBlocksInput.size()-1;
}


/***********************************************************************
***********************************************************************/
kcc_int32
DmaDescription::startNewDmaBlockTpbToTpb(const dma::DmaQueue* que, EngineId engId, const char* comment)
{
    DmaBlockTpbToTpb block(this, que, engId, comment);
    m_DmaBlocksTpbToTpb.push_back(block);
    return m_DmaBlocksTpbToTpb.size()-1;
}


/***********************************************************************
***********************************************************************/
const char*
DmaDescription::gJsonFileName(EngineId engId)
{
    switch (engId) {
    case EngineId::PeArray:
        return m_PeJsonFileName;
        break;
    case EngineId::Activation:
        return m_ActJsonFileName;
        break;
    case EngineId::Pooling:
        return m_PoolJsonFileName;
        break;
    default:
        Assert(false, "Wrong engine id ", static_cast<int>(engId));
        return nullptr;
        break;
    }
    return nullptr;
}

/***********************************************************************
***********************************************************************/
const char*
DmaDescription::gEngineName(EngineId engId)
{
    switch (engId) {
    case EngineId::PeArray:
        return "pe";
        break;
    case EngineId::Activation:
        return "act";
        break;
    case EngineId::Pooling:
        return "pool";
        break;
    default:
        Assert(false, "Wrong engine id ", static_cast<int>(engId));
        return nullptr;
        break;
    }
    return nullptr;
}


/***********************************************************************
***********************************************************************/
const char*
DmaDescription::gSymbolicStateBuffer()
{
    return "SB";
}





/***********************************************************************
***********************************************************************/
auto DmaDescription::gFileSymbolicId(const std::string& fileName, FileType fileType)
    -> FileIdType&
{
    auto it = m_FileNameToId.find(fileName);
    if (m_FileNameToId.end() == it) {
        std::ostringstream oss;
        if (FileType::Weight == fileType) {
            oss << "W" << m_WeightFileIdCnt++;
        } else {
            const char simout[] = "-simout.npy";
            auto fileNameCopy(fileName);
            auto pos = fileNameCopy.find(simout);
            if (std::string::npos != pos) {
                fileNameCopy.replace(pos, ArraySizeof(simout)-1, "");
            } else {
                const char dotnpy[] = ".npy";
                pos = fileNameCopy.find(dotnpy);
                Assert(std::string::npos != pos,
                    "Did not find '.npy' in out file ", fileName);
                fileNameCopy.replace(pos, ArraySizeof(dotnpy)-1, "");
            }
            oss << fileNameCopy;
        }
        const FileIdType fileId(oss.str().c_str(), fileName, fileType);
        m_FileNameToId[fileName] = fileId;
        return m_FileNameToId[fileName];
    } else {
        return (*it).second;
    }
}



/***********************************************************************
***********************************************************************/
auto DmaDescription::gFileSymbolicId(const std::string& fileName)
    -> FileIdType&
{
    auto it = m_FileNameToId.find(fileName);
    Assert(m_FileNameToId.end() != it, "Could not find file ", fileName);
    return (*it).second;
}

/***********************************************************************
***********************************************************************/
auto DmaDescription::gFileSymbolicId(const std::string& fileName) const
    -> const FileIdType&
{
    auto it = m_FileNameToId.find(fileName);
    Assert(m_FileNameToId.end() != it, "Could not find file ", fileName);
    return (*it).second;
}

/***********************************************************************
***********************************************************************/
auto
DmaDescription::gInFileSymbolicId(const std::string& refFile) const
    -> const FileIdType&
{
    auto it = m_InFileNameToId.find(refFile);
    Assert(m_InFileNameToId.end() != it, "Failed to find input ", refFile);
    const FileIdType& fileId((*it).second);
    return fileId;
}

auto
DmaDescription::gInFileSymbolicId(const std::string& refFile)
    -> FileIdType&
{
    auto it = m_InFileNameToId.find(refFile);
    Assert(m_InFileNameToId.end() != it, "Failed to find input ", refFile);
    FileIdType& fileId((*it).second);
    return fileId;
}

/***********************************************************************
***********************************************************************/
void
DmaDescription::rOutputSizeBytes(kcc_int64 sz, const std::string& refFileName, bool qOut)
{
    auto& idType(gFileSymbolicId(refFileName, qOut ? FileType::Output : FileType::TmpBuffer));
    idType.m_Size = sz;
}

kcc_int64
DmaDescription::gOutputSizeBytes(const std::string& refFileName, bool qOut)
{
    const auto& idType(gFileSymbolicId(refFileName, qOut ? FileType::Output : FileType::TmpBuffer));
    return idType.m_Size;
}

kcc_int64
DmaDescription::gInputSizeBytes(const std::string& refFileName) const
{
    const FileIdType& fileId(gInFileSymbolicId(refFileName));
    Assert(fileId.m_Size > 0, "Size of input ", refFileName, " is not > 0");
    return fileId.m_Size;
}

void
DmaDescription::rInputSizeBytes(kcc_int64 sz, const std::string& refFileName)
{
    FileIdType& fileId(gInFileSymbolicId(refFileName));
    if (fileId.m_Size > 0) {
        Assert(fileId.m_Size == sz, "Two different values for input ", refFileName,
                ", old=", fileId.m_Size, " and new=", sz);
    } else {
        fileId.m_Size = sz;
    }
}

void
DmaDescription::recordInFile(const std::string& refFileName)
{
    const char dotNpy[] = ".npy";
    auto fileNameCopy(refFileName);
    auto pos = fileNameCopy.find(dotNpy);
    if (std::string::npos != pos) {
        fileNameCopy.replace(pos, ArraySizeof(dotNpy)-1, "");
    }

    FileIdType fileId;
    fileId.m_VarName = fileNameCopy;
    fileId.m_FileName = refFileName;
    fileId.m_FileType = FileType::Input;
    fileId.m_Size = -1;

    m_InFileNameToId[refFileName] = fileId;
}

/***********************************************************************
***********************************************************************/
bool
DmaDescription::qHasFile(const std::string& fileName) const
{
    auto it = m_FileNameToId.find(fileName);
    return m_FileNameToId.end() != it;
}

/***********************************************************************
***********************************************************************/
#if 0
std::string
DmaDescription::gSymbolicQueue(EngineId engId, bool inpt, bool weight) const
{
    const char* engName = nullptr;
    switch (engId) {
    case EngineId::Pooling:
        engName = "pool";
        break;
    case EngineId::Activation:
        engName = "act";
        break;
    case EngineId::PeArray:
        engName = "pe";
        break;
    default:
        Assert(false, "Must be Pool, Act, or PeArray");
        engName = nullptr;
        break;
    }
    std::ostringstream oss;
    if (inpt) {
        oss << "q_" << engName << "_" << (weight ? "in_w" : "in_d");
    } else {
        oss << "q_" << engName << "_out";
    }
    return std::string(oss.str().c_str());
}
#endif


/***********************************************************************
***********************************************************************/
void
DmaDescription::writeDmaDescriptors(
    const char* binFileName,
    const EngineId engId)
{
    const char* const jsonFileName = gJsonFileName(engId);
    const char* name = nullptr;
    switch (engId) {
    case EngineId::Pooling:
        name = "pool_json";
        break;
    case EngineId::Activation:
        name = "act_json";
        break;
    case EngineId::PeArray:
        name = "pe_array_json";
        break;
    default:
        Assert(false, "Must be Pool, Act, or PeArray");
        break;
    }

    json j;
    if (engId == EngineId::Activation) {
        writeActivationFuncs(j);
    }
    j[Keys::gBinFileName()] = binFileName;
    j[Keys::gJsonName()] = name;

    auto jDmaBlocks = json::array();

    for (const auto& dmaBlockToTpb : m_DmaBlocksToTpb) {
        if (dmaBlockToTpb.gTriggerEngineId() != engId) {
            continue;
        }

        json jBlockToTpb;
        jBlockToTpb[Keys::gQueueName()]     = dmaBlockToTpb.gDmaQueue()->gName();
        jBlockToTpb[Keys::gBlockId()]       = dmaBlockToTpb.gBlockId();
        jBlockToTpb[Keys::gHashComment()]   = dmaBlockToTpb.gComment();
        jBlockToTpb[Keys::gHashBlockSize()] = dmaBlockToTpb.size();
        jBlockToTpb[Keys::gHashNumDescs()]  = dmaBlockToTpb.gNumDescs();
        dmaBlockToTpb.setDmaEventField(jBlockToTpb);

        auto jDmaDescs = json::array();
        for (const auto& desc : dmaBlockToTpb.gDescs()) {
            desc.assertAccessCheck();
            json jDmaDesc;
            jDmaDesc[Keys::gFromId()]       = desc.gSrcFileId().m_VarName;
            jDmaDesc[Keys::gFromOffset()]   = desc.gSrcFileAddress();
            jDmaDesc[Keys::gToId()]         = gSymbolicStateBuffer();
            jDmaDesc[Keys::gToOffset()]     = desc.gDstSbAddress();
            jDmaDesc[Keys::gSize()]         = desc.gNumBytes();

            jDmaDescs.push_back(jDmaDesc);
        }
        jBlockToTpb[Keys::gDescriptor()] = jDmaDescs;

        jDmaBlocks.push_back(jBlockToTpb);
    }

    for (const auto& dmaBlockFromTpb : m_DmaBlocksFromTpb) {
        if (dmaBlockFromTpb.gTriggerEngineId() != engId) {
            continue;
        }

        json jBlockFromTpb;
        jBlockFromTpb[Keys::gQueueName()]       = dmaBlockFromTpb.gDmaQueue()->gName();
        jBlockFromTpb[Keys::gBlockId()]         = dmaBlockFromTpb.gBlockId();
        jBlockFromTpb[Keys::gHashComment()]     = dmaBlockFromTpb.gComment();
        jBlockFromTpb[Keys::gHashBlockSize()]   = dmaBlockFromTpb.size();
        jBlockFromTpb[Keys::gHashNumDescs()]    = dmaBlockFromTpb.gNumDescs();
        dmaBlockFromTpb.setDmaEventField(jBlockFromTpb);

        auto jDmaDescs = json::array();
        for (const auto& desc : dmaBlockFromTpb.gDescs()) {
            desc.assertAccessCheck();
            json jDmaDesc;
            jDmaDesc[Keys::gFromId()]       = gSymbolicStateBuffer();
            jDmaDesc[Keys::gFromOffset()]   = desc.gSrcSbAddress();
            jDmaDesc[Keys::gToId()]         = desc.gDstFileId().m_VarName;
            jDmaDesc[Keys::gToOffset()]     = desc.gDstFileAddress();
            jDmaDesc[Keys::gSize()]         = desc.gNumBytes();

            jDmaDescs.push_back(jDmaDesc);
        }
        jBlockFromTpb[Keys::gDescriptor()] = jDmaDescs;

        jDmaBlocks.push_back(jBlockFromTpb);
    }

    for (const auto& dmaBlockTpbToTpb : m_DmaBlocksTpbToTpb) {
        if (dmaBlockTpbToTpb.gTriggerEngineId() != engId) {
            continue;
        }

        json jBlockTpbToTpb;
        jBlockTpbToTpb[Keys::gQueueName()]     = dmaBlockTpbToTpb.gDmaQueue()->gName();
        jBlockTpbToTpb[Keys::gBlockId()]       = dmaBlockTpbToTpb.gBlockId();
        jBlockTpbToTpb[Keys::gHashComment()]   = dmaBlockTpbToTpb.gComment();
        jBlockTpbToTpb[Keys::gHashBlockSize()] = dmaBlockTpbToTpb.size();
        jBlockTpbToTpb[Keys::gHashNumDescs()]  = dmaBlockTpbToTpb.gNumDescs();
        dmaBlockTpbToTpb.setDmaEventField(jBlockTpbToTpb);

        auto jDmaDescs = json::array();
        for (const auto& desc : dmaBlockTpbToTpb.gDescs()) {
            desc.assertAccessCheck();
            json jDmaDesc;
            jDmaDesc[Keys::gFromId()]       = gSymbolicStateBuffer();
            jDmaDesc[Keys::gFromOffset()]   = desc.gSrcSbAddress();
            jDmaDesc[Keys::gToId()]         = gSymbolicStateBuffer();
            jDmaDesc[Keys::gToOffset()]     = desc.gDstSbAddress();
            jDmaDesc[Keys::gSize()]         = desc.gNumBytes();

            jDmaDescs.push_back(jDmaDesc);
        }
        jBlockTpbToTpb[Keys::gDescriptor()] = jDmaDescs;

        jDmaBlocks.push_back(jBlockTpbToTpb);
    }


    j[Keys::gDmaBlock()] = jDmaBlocks;

    std::ofstream o(jsonFileName);
    o << std::setw(4) << j << std::endl;
}


/***********************************************************************
{
    "name" : "the test that does nothing",
    "var" : {
        "W" : { "type":"file", "file_name":"weights.bin"},
        "SB" : { "type": "state-buffer" },
        "IN" : { "type": "input", "size": 1024 },
        "OUT" : { "type": "output", "size": 1024 }
    },
    "dma_queue" : {
        "IN_QID" : { "type": "in" },
        "OUT_QID" : { "type": "out" },
        "W_QID" : { "type" : "data", "owner": "pe" }
    },
    "pe" : "pe.json",
    "act" : "act.json",
    "pool" : "pool.json",
    "host" : "host.json"
    "pe_instr" : "Trivnet-pe.bin"
}
***********************************************************************/





/***********************************************************************
***********************************************************************/
void DmaDescription::writeActivationFuncs(json& j)
{
    auto activationFuncs = json::array();
    for (auto actFunc : m_ActivationFuncs) {
        const json jActFunc(utils::ActivationFunc2Str(actFunc));
        activationFuncs.push_back(jActFunc);
    }
    j[Keys::gActivationFuncs()] = activationFuncs;
}

/**********************************************************************/
void
DmaDescription::writeQueDefinitions(json&j)
{
    std::set<const dma::DmaQueue*> processedQues;
    {
        auto jDmaQueue = json::object();
        {
            std::set<const dma::DmaQueue*> inDmaQueues;
            for (const auto& dmaInBlock : m_DmaBlocksInput) {
                inDmaQueues.insert(dmaInBlock.gDmaQueue());
            }
            for (auto que : inDmaQueues) {
                Assert(processedQues.find(que) == processedQues.end(), "Repeated DMA queue ", que->gName());
                processedQues.insert(que);
                json jInQueDesc;
                jInQueDesc[Keys::gQueueType()] = "in";
                if (qUseSemaphore()) {
                    jInQueDesc[Keys::gSemId()] = que->gSemaphoreId();
                }
                char buf[512];
                sprintf(buf, "# %s", Keys::gOwner());
                jInQueDesc[buf] = gEngineName(que->gEngineId());
                jDmaQueue[que->gName()]  = jInQueDesc;
            }
        }

        {
            std::set<const dma::DmaQueue*> outDmaQueues;
            for (const auto& dmaBlockFromTpb : m_DmaBlocksFromTpb) {
                if (dmaBlockFromTpb.qOut()) {
                    outDmaQueues.insert(dmaBlockFromTpb.gDmaQueue());
                }
            }
            for (auto que : outDmaQueues) {
                Assert(processedQues.find(que) == processedQues.end(), "Repeated DMA queue ", que->gName());
                processedQues.insert(que);
                json jOutQueDesc;
                jOutQueDesc[Keys::gQueueType()] = "out";
                if (qUseSemaphore()) {
                    jOutQueDesc[Keys::gSemId()] = que->gSemaphoreId();
                }
                jOutQueDesc[Keys::gOwner()] = gEngineName(que->gEngineId());
                jDmaQueue[que->gName()]  = jOutQueDesc;
            }
        }

        {
            std::set<const dma::DmaQueue*> outDmaQueues;
            for (const auto& dmaBlockFromTpb : m_DmaBlocksFromTpb) {
                if (! dmaBlockFromTpb.qOut()) {
                    outDmaQueues.insert(dmaBlockFromTpb.gDmaQueue());
                }
            }
            for (auto que : outDmaQueues) {
                Assert(processedQues.find(que) == processedQues.end(), "Repeated DMA queue ", que->gName());
                processedQues.insert(que);
                json jOutQueDesc;
                jOutQueDesc[Keys::gQueueType()] = "data";
                if (qUseSemaphore()) {
                    jOutQueDesc[Keys::gSemId()] = que->gSemaphoreId();
                }
                jOutQueDesc[Keys::gOwner()] = gEngineName(que->gEngineId());
                jDmaQueue[que->gName()]  = jOutQueDesc;
            }
        }

        {
            std::set<const dma::DmaQueue*> dataDmaQueues;
            for (const auto& dmaBlockToTpb : m_DmaBlocksToTpb) {
                dataDmaQueues.insert(dmaBlockToTpb.gDmaQueue());
            }
            for (auto que : dataDmaQueues) {
                Assert(processedQues.find(que) == processedQues.end(), "Repeated DMA queue ", que->gName());
                processedQues.insert(que);
                json jDataQueDesc;
                jDataQueDesc[Keys::gQueueType()] = "data";
                if (qUseSemaphore()) {
                    jDataQueDesc[Keys::gSemId()] = que->gSemaphoreId();
                }
                jDataQueDesc[Keys::gOwner()] = gEngineName(que->gEngineId());
                if (! que->qFirstQueue()) {
                    jDataQueDesc[Keys::gAxiPort()] = 1;
                }
                jDmaQueue[que->gName()]  = jDataQueDesc;
            }
        }

        {
            std::set<const dma::DmaQueue*> tpbToTpbDmaQueues;
            for (const auto& dmaBlockTpbToTpb : m_DmaBlocksTpbToTpb) {
                tpbToTpbDmaQueues.insert(dmaBlockTpbToTpb.gDmaQueue());
            }
            for (auto que : tpbToTpbDmaQueues) {
                Assert(processedQues.find(que) == processedQues.end(), "Repeated DMA queue ", que->gName());
                processedQues.insert(que);
                json jTptToTpbQueDesc;
                jTptToTpbQueDesc[Keys::gQueueType()] = "data";
                if (qUseSemaphore()) {
                    jTptToTpbQueDesc[Keys::gSemId()] = que->gSemaphoreId();
                }
                jTptToTpbQueDesc[Keys::gOwner()] = gEngineName(que->gEngineId());
                jTptToTpbQueDesc[Keys::gAxiPort()] = 1;
                jDmaQueue[que->gName()]  = jTptToTpbQueDesc;
            }
        }

        j[Keys::gDmaQueue()] = jDmaQueue;
    }

}

/**********************************************************************/
void
DmaDescription::writeDefinitions(const char* peInstrFileName,
    const char* actInstrFileName, const char* poolInstrFileName)
{
    std::array<EngineId, 3> engIds = { {EngineId::PeArray, EngineId::Pooling, EngineId::Activation} };
    json j;
    j[Keys::gJsonName()] = "definition";
    std::string version("0.3-");
    version += utils::Git::gShaShort();
    j["version"] = version;

    {
        json jDebugInfo;
        jDebugInfo[Keys::gWavegraph()] = m_WavegraphJson;
        j[Keys::gDebugInfo()] = jDebugInfo;
    }

    for (auto engId : engIds) {
        j[gEngineName(engId)] = gJsonFileName(engId);
    }

    { // "pe_instr" : "Trivnet-pe.bin"
        std::string instrFileKey(gEngineName(EngineId::PeArray));
        instrFileKey += "_";
        instrFileKey += Keys::gBinFileName();
        j[instrFileKey] = peInstrFileName;
    }
    {
        std::string instrFileKey(gEngineName(EngineId::Pooling));
        instrFileKey += "_";
        instrFileKey += Keys::gBinFileName();
        j[instrFileKey] = poolInstrFileName;
    }
    {
        std::string instrFileKey(gEngineName(EngineId::Activation));
        instrFileKey += "_";
        instrFileKey += Keys::gBinFileName();
        j[instrFileKey] = actInstrFileName;
    }


    j["host"] = m_HostJsonFileName;

    writeQueDefinitions(j);
    writeVarDefinitions(j);

    if (false) {
        auto runtimeEventsJson = json::array();
        for (uint8_t e = events::EventMgr::EventId_RunTimeFirst(); e <= events::EventMgr::EventId_RunTimeLast(); e++){
            json e_json = e;
            runtimeEventsJson.push_back(e_json);
        }
        j["runtime_events"] = runtimeEventsJson;
    }

    std::ofstream o(m_DefJsonFileName);
    o << std::setw(4) << j << std::endl;
}

/**********************************************************************/
void
DmaDescription::writeVarDefinitions(json&j)
{
    const char varTypeStateBuffer[] = "state-buffer";
    const char varTypeInput[]       = "input";
    const char varTypeOutput[]      = "output";
    const char varTypeWeight[]      = "weight";
    const char varTypeTmpBuf[]      = "tmp-buf";
    {
        json jVars;
        {
            json varDesc;
            varDesc[Keys::gDescType()] = varTypeStateBuffer;
            jVars[gSymbolicStateBuffer()] = varDesc;
        }


        {
            const kcc_int32 numInputs = m_InFileNameToId.size();
            Assert(numInputs > 0, "The NN model should have at least one input");

            for (const auto& kv : m_InFileNameToId) {
                json varDesc;

                const FileIdType& fileIdType(kv.second);
                Assert(fileIdType.m_FileType == FileType::Input, "Must be Input");
                varDesc[Keys::gHashTransferType()] = varTypeInput;
                varDesc[Keys::gDescType()] = varTypeInput;
                varDesc[Keys::gSize()] = fileIdType.m_Size;

                if (numInputs > 1) {
                    jVars[fileIdType.m_VarName] = varDesc;
                } else {
                    jVars[singleInputName] = varDesc;
                }
            }
        }


        {
            for (const auto& kv : m_FileNameToId) {
                json varDesc;
                const FileIdType& fileIdType(kv.second);
                switch (fileIdType.m_FileType) {
                case FileType::Input:
                    Assert(false, "Cannot be Input");
                    break;
                case FileType::Output:
                    varDesc[Keys::gHashTransferType()] = varTypeOutput;
                    varDesc[Keys::gHashFileName()] = fileIdType.m_FileName;
                    varDesc[Keys::gDescType()] = varTypeOutput;
                    varDesc[Keys::gSize()] = fileIdType.m_Size;
                    break;
                case FileType::Weight:
                    varDesc[Keys::gHashTransferType()] = varTypeWeight;
                    varDesc[Keys::gDescType()] = "file";
                    varDesc[Keys::gWeightFileName()] = fileIdType.m_FileName;
                    break;
                case FileType::TmpBuffer:
                    varDesc[Keys::gHashTransferType()] = varTypeTmpBuf;
                    varDesc[Keys::gHashFileName()] = fileIdType.m_FileName;
                    varDesc[Keys::gDescType()] = varTypeTmpBuf;
                    varDesc[Keys::gSize()] = fileIdType.m_Size;
                    break;
                default:
                    Assert(false, "Wrong file type");
                    break;
                }
                jVars[fileIdType.m_VarName] = varDesc;
            }
        }

        j["var"] = jVars;
    }
}

/***********************************************************************
***********************************************************************/
void
DmaDescription::writeInOutDescriptors()
{
    json j;

    j[Keys::gJsonName()] = "host_json";
    auto jDmaBlocks = json::array();

    for (const auto& dmaInBlock : m_DmaBlocksInput) {
        json jDmaBlock;
        jDmaBlock[Keys::gQueueName()]  = dmaInBlock.gDmaQueue()->gName();
        jDmaBlock[Keys::gBlockId()]     = dmaInBlock.gBlockId();
        jDmaBlock[Keys::gHashComment()] = dmaInBlock.gComment();
        jDmaBlock[Keys::gHashBlockSize()] = dmaInBlock.size();
        jDmaBlock[Keys::gHashNumDescs()] = dmaInBlock.gNumDescs();
        dmaInBlock.setDmaEventField(jDmaBlock);

        const kcc_int32 numInputs = m_InFileNameToId.size();
        auto jDmaDescs = json::array();
        for (const auto& desc : dmaInBlock.gDescs()) {
            desc.assertAccessCheck();
            json jDmaDesc;
            if (numInputs > 1) {
                jDmaDesc[Keys::gFromId()]         = desc.gSrcFileId().m_VarName;
            } else {
                jDmaDesc[Keys::gFromId()]         = singleInputName;
            }
            jDmaDesc[Keys::gFromOffset()]     = desc.gSrcFileAddress();
            jDmaDesc[Keys::gToId()]           = gSymbolicStateBuffer();
            jDmaDesc[Keys::gToOffset()]       = desc.gDstSbAddress();
            jDmaDesc[Keys::gSize()]         = desc.gNumBytes();

            jDmaDescs.push_back(jDmaDesc);
        }
        jDmaBlock[Keys::gDescriptor()] = jDmaDescs;
        jDmaBlocks.push_back(jDmaBlock);
    }


    j[Keys::gDmaBlock()] = jDmaBlocks;

    std::ofstream o(m_HostJsonFileName);
    o << std::setw(4) << j << std::endl;
}

/***********************************************************************
***********************************************************************/
kcc_uint64
DmaDescription::DmaBlockInput::size() const
{
    kcc_uint64 numInBytes = 0;
    for (const auto& desc : m_Descs) {
        numInBytes += desc.gNumBytes();
    }
    return numInBytes;
}

/***********************************************************************
***********************************************************************/
kcc_uint32
DmaDescription::DmaBlockInput::gNumDescs() const
{
    return m_Descs.size();
}



/***********************************************************************
***********************************************************************/
kcc_uint64
DmaDescription::DmaBlockFromTpb::size() const
{
    kcc_uint64 numInBytes = 0;
    for (const auto& desc : m_Descs) {
        numInBytes += desc.gNumBytes();
    }
    return numInBytes;
}

/***********************************************************************
***********************************************************************/
kcc_uint32
DmaDescription::DmaBlockFromTpb::gNumDescs() const
{
    return m_Descs.size();
}



/***********************************************************************
***********************************************************************/
kcc_uint64
DmaDescription::DmaBlockToTpb::size() const
{
    kcc_uint64 numBytes = 0;
    for (const auto& desc : m_Descs) {
        numBytes += desc.gNumBytes();
    }
    return numBytes;
}

/***********************************************************************
***********************************************************************/
kcc_uint64
DmaDescription::DmaBlockTpbToTpb::size() const
{
    kcc_uint64 numBytes = 0;
    for (const auto& desc : m_Descs) {
        numBytes += desc.gNumBytes();
    }
    return numBytes;
}

/***********************************************************************
***********************************************************************/
kcc_uint32
DmaDescription::DmaBlockToTpb::gNumDescs() const
{
    return m_Descs.size();
}

/***********************************************************************
***********************************************************************/
kcc_uint32
DmaDescription::DmaBlockTpbToTpb::gNumDescs() const
{
    return m_Descs.size();
}

/***********************************************************************
***********************************************************************/
void
DmaDescription::addActivationFunc(ActivationFunc actFunc)
{
    m_ActivationFuncs.insert(actFunc);
}


DmaDescription::FileIdType::FileIdType(const FileIdType&) = default;

const char* DmaDescription::Keys::gBinFileName()
{
    return "instr";
}

const char* DmaDescription::Keys::gJsonName()
{
    return "name";
}

const char* DmaDescription::Keys::gQueueName()
{
    return "queue";
}

const char* DmaDescription::Keys::gFromId()
{
    return "from";
}

const char* DmaDescription::Keys::gFromOffset()
{
    return "from_off";
}

const char* DmaDescription::Keys::gToId()
{
    return "to";
}

const char* DmaDescription::Keys::gToOffset()
{
    return "to_off";
}

const char* DmaDescription::Keys::gSize()
{
    return "size";
}

const char* DmaDescription::Keys::gDescriptor()
{
    return "desc";
}

const char* DmaDescription::Keys::gBlockId()
{
    return "id";
}

const char* DmaDescription::Keys::gDescType()
{
    return "type";
}

const char* DmaDescription::Keys::gSemId()
{
    return "semaphore";
}

const char* DmaDescription::Keys::gQueueType()
{
    return "type";
}

const char* DmaDescription::Keys::gOwner()
{
    return "owner";
}

const char* DmaDescription::Keys::gAxiPort()
{
    return "axi_port";
}

const char* DmaDescription::Keys::gDmaBlock()
{
    return "dma";
}

const char* DmaDescription::Keys::gDmaQueue()
{
    return "dma_queue";
}

const char*
DmaDescription::Keys::gWeightFileName()
{
    return "file_name";
}


// Hash (comments)

const char* DmaDescription::Keys::gHashComment()
{
    return "#comment";
}

const char* DmaDescription::Keys::gHashBlockSize()
{
    return "#block_num_bytes";
}

const char* DmaDescription::Keys::gHashNumDescs()
{
    return "#block_num_descs";
}
const char* DmaDescription::Keys::gHashTransferType()
{
    return "#transfer-type";
}

const char* DmaDescription::Keys::gHashFileName()
{
    return "#file-name";
}

const char* DmaDescription::Keys::gDebugInfo()
{
    return "debug_info";
}

const char* DmaDescription::Keys::gWavegraph()
{
    return "wavegraph";
}

const char* DmaDescription::Keys::gActivationFuncs()
{
    return "activation_functions";
}

}}

