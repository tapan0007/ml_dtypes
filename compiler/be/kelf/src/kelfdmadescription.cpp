#include <fstream>
#include <iostream>


// Json package
#include "nlohmann/json.hpp"



#include "utils/inc/types.hpp"
#include "utils/inc/version.hpp"
#include "utils/inc/asserter.hpp"
#include "utils/inc/misc.hpp"

#include "nets/inc/network.hpp"
#include "events/inc/eventmgr.hpp"
#include "kelf/inc/kelfdmadescription.hpp"

namespace kcc {
namespace kelf {

using json = nlohmann::json;

/***********************************************************************
***********************************************************************/
DmaDescription::DmaDescription(const nets::Network& network)
    : m_Network(network)
    , m_InputSizeBytes(-1)
    , m_OutputSizeBytes(-1)
{
}


/***********************************************************************
***********************************************************************/
kcc_int32
DmaDescription::gBlockIdForQueue(const std::string& queName)
{
    kcc_int32 blockId;
    const auto it = m_QueueToBlockId.find(queName);
    if (it == m_QueueToBlockId.end()) {
        blockId = 0;
    } else {
        blockId = (*it).second;
    }
    m_QueueToBlockId[queName] = blockId + 1;
    return blockId;
}

/***********************************************************************
***********************************************************************/
kcc_int32
DmaDescription::gNumBlockIdsForQueue(const std::string& queName) const
{
    const auto it = m_QueueToBlockId.find(queName);
    if (it == m_QueueToBlockId.end()) {
        return 0;
    } else {
        return (*it).second;
    }
}


/***********************************************************************
***********************************************************************/
auto
DmaDescription::startNewDmaBlockToTpb(EngineId engId, bool qWeights, const char* comment)
    -> DmaBlockToTpb&
{
    m_DmaBlocksToTpb.push_back(DmaBlockToTpb(*this, engId, qWeights, comment));
    return m_DmaBlocksToTpb[m_DmaBlocksToTpb.size()-1];
}


/***********************************************************************
***********************************************************************/
auto
DmaDescription::startNewDmaBlockFromTpb(EngineId engId, bool qOut, const char* comment)
    -> DmaBlockFromTpb&
{
    m_DmaBlocksFromTpb.push_back(DmaBlockFromTpb(*this, engId, qOut, comment));
    return m_DmaBlocksFromTpb[m_DmaBlocksFromTpb.size()-1];
}

/***********************************************************************
***********************************************************************/
auto
DmaDescription::startNewDmaBlockInput( const char* comment)
    -> DmaBlockInput&
{
    m_DmaBlocksInput.push_back(DmaBlockInput(*this, comment));
    return m_DmaBlocksInput[ m_DmaBlocksInput.size()-1];
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
const char*
DmaDescription::gSymbolicInput()
{
    return "IN";
}

/***********************************************************************
***********************************************************************/
const char*
DmaDescription::gSymbolicInQueue()
{
    return "IN_QUE";
}

/***********************************************************************
***********************************************************************/


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
void
DmaDescription::rOutputSizeBytes(kcc_int64 sz, const std::string& refFileName)
{
    auto& idType(gFileSymbolicId(refFileName, FileType::LoadSave));
    idType.m_Size = sz;
}

kcc_int64
DmaDescription::gOutputSizeBytes(const std::string& refFileName)
{
    const auto& idType(gFileSymbolicId(refFileName, FileType::LoadSave));
    return idType.m_Size;
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

    std::vector<json> jDmaBlocks;

    for (const auto& dmaBlockToTpb : m_DmaBlocksToTpb) {
        if (dmaBlockToTpb.gTriggerEngineId() != engId) {
            continue;
        }

        json jBlockToTpb;
        jBlockToTpb[Keys::gQueueName()]    = dmaBlockToTpb.gQueueName();
        jBlockToTpb[Keys::gBlockId()]       = dmaBlockToTpb.gBlockId();
        jBlockToTpb[Keys::gHashComment()] = dmaBlockToTpb.gComment();
        jBlockToTpb[Keys::gHashBlockSize()] = dmaBlockToTpb.size();
        dmaBlockToTpb.setDmaEventField(jBlockToTpb);

        std::vector<json> jDmaDescs;
        for (const auto& desc : dmaBlockToTpb.gDescs()) {
            desc.assertAccessCheck();
            json jDmaDesc;
            jDmaDesc[Keys::gFromId()]         = desc.gSrcFileId().m_VarName;
            jDmaDesc[Keys::gFromOffset()]     = desc.gSrcFileAddress();
            jDmaDesc[Keys::gToId()]           = gSymbolicStateBuffer();
            jDmaDesc[Keys::gToOffset()]       = desc.gDstSbAddress();
            jDmaDesc[Keys::gSize()]         = desc.gNumBytes();

            jDmaDescs.push_back(jDmaDesc);
        }
        jBlockToTpb[Keys::gDescriptor()] = jDmaDescs;

        jDmaBlocks.push_back(jBlockToTpb);
    }

    for (const auto& dmaBlockFromTpb : m_DmaBlocksFromTpb) {
        if (false && dmaBlockFromTpb.qOut()) {
            continue;
        }
        if (dmaBlockFromTpb.gTriggerEngineId() != engId) {
            continue;
        }

        json jBlockFromTpb;
        jBlockFromTpb[Keys::gQueueName()]      = dmaBlockFromTpb.gQueueName();
        jBlockFromTpb[Keys::gBlockId()]         = dmaBlockFromTpb.gBlockId();
        jBlockFromTpb[Keys::gHashComment()]   = dmaBlockFromTpb.gComment();
        jBlockFromTpb[Keys::gHashBlockSize()]  = dmaBlockFromTpb.size();
        dmaBlockFromTpb.setDmaEventField(jBlockFromTpb);

        std::vector<json> jDmaDescs;
        for (const auto& desc : dmaBlockFromTpb.gDescs()) {
            desc.assertAccessCheck();
            json jDmaDesc;
            jDmaDesc[Keys::gFromId()]         = gSymbolicStateBuffer();
            jDmaDesc[Keys::gFromOffset()]     = desc.gSrcSbAddress();
            jDmaDesc[Keys::gToId()]           = desc.gDstFileId().m_VarName;
            jDmaDesc[Keys::gToOffset()]       = desc.gDstFileAddress();
            jDmaDesc[Keys::gSize()]         = desc.gNumBytes();

            jDmaDescs.push_back(jDmaDesc);
        }
        jBlockFromTpb[Keys::gDescriptor()] = jDmaDescs;

        jDmaBlocks.push_back(jBlockFromTpb);
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
void DmaDescription::writeActivationFuncs(json& j)
{
    auto activationFuncs = json::array();
    for (auto actFunc : m_ActivationFuncs) {
        const json jActFunc(utils::ActivationFunc2Str(actFunc));
        activationFuncs.push_back(jActFunc);
    }
    j[Keys::gActivationFuncs()] = activationFuncs;
}

void
DmaDescription::writeDefinitions(const char* peInstrFileName,
    const char* actInstrFileName, const char* poolInstrFileName)
{
    std::array<EngineId, 3> engIds = { {EngineId::PeArray, EngineId::Pooling, EngineId::Activation} };
    json j;
    j[Keys::gJsonName()] = "definition";
    std::string version("0.2-");
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
    {
        json jDmaQueue;
        json queDesc;

        queDesc[Keys::gDescType()] = "in";
        jDmaQueue[gSymbolicInQueue()] = queDesc;

        for (auto engId : engIds) {
            const std::string queName = gSymbolicQueue(engId, false, false);
            if (gNumBlockIdsForQueue(queName) > 0) {
                queDesc[Keys::gDescType()] = "out";
                queDesc[Keys::gOwner()] = gEngineName(engId);
                jDmaQueue[queName]  = queDesc; // output
            }
        }

        for (auto engId : engIds) {
            std::string queName = gSymbolicQueue(engId, true, true);
            if (gNumBlockIdsForQueue(queName) > 0) {
                queDesc[Keys::gQueueType()] = "data";
                queDesc[Keys::gOwner()] = gEngineName(engId);
                jDmaQueue[queName]    = queDesc; // input for weights
            }

            queName = gSymbolicQueue(engId, true, false);
            if (gNumBlockIdsForQueue(queName) > 0) {
                queDesc[Keys::gQueueType()] = "data";
                queDesc[Keys::gOwner()] = gEngineName(engId);
                jDmaQueue[queName]   = queDesc; // input for data
            }

        }

        j[Keys::gDmaQueue()] = jDmaQueue;
    }
    {
        json jVars;
        {
            json varDesc;
            varDesc[Keys::gDescType()] = "state-buffer";
            jVars[gSymbolicStateBuffer()] = varDesc;
        }


        { // input
            json varDesc;

            varDesc[Keys::gHashTransferType()] = "input";
            varDesc[Keys::gDescType()] = "input";
            Assert(gInputSizeBytes() > 0, "Number of input bytes must be positive");
            varDesc[Keys::gSize()] = gInputSizeBytes();
            if (false) { // to be used by RT to verify incoming requests
                varDesc["tensor_dtype"]         = m_Network.gInDataType().gName();
                varDesc["tensor_format"]        = m_Network.gInTensorFormat();
                varDesc["tensor_dimensions"]    = m_Network.gInTensorDimensions();
                varDesc["data_shuffle"]         = m_Network.gInLayerStride();
            }
            jVars[gSymbolicInput()]         = varDesc;
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
                    varDesc[Keys::gHashTransferType()] = "output";
                    varDesc[Keys::gHashFileName()] = fileIdType.m_FileName;
                    varDesc[Keys::gDescType()] = "output";
                    varDesc[Keys::gSize()] = fileIdType.m_Size;
                    break;
                case FileType::Weight:
                    varDesc[Keys::gHashTransferType()] = "weight";
                    varDesc[Keys::gDescType()] = "file";
                    varDesc[Keys::gWeightFileName()] = fileIdType.m_FileName;
                    break;
                case FileType::LoadSave:
                    varDesc[Keys::gHashTransferType()] = "load_save";
                    varDesc[Keys::gHashFileName()] = fileIdType.m_FileName;
                    varDesc[Keys::gDescType()] = "output";
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

/***********************************************************************
***********************************************************************/
void
DmaDescription::writeInOutDescriptors()
{
    json j;

    j[Keys::gJsonName()] = "host_json";
    std::vector<json> jDmaBlocks;

    for (const auto& dmaBlock : m_DmaBlocksInput) {
        json jDmaBlock;
        jDmaBlock[Keys::gQueueName()]  = gSymbolicInQueue();
        jDmaBlock[Keys::gBlockId()]     = dmaBlock.gBlockId();
        jDmaBlock[Keys::gHashComment()] = dmaBlock.gComment();
        jDmaBlock[Keys::gHashBlockSize()] = dmaBlock.size();
        dmaBlock.setDmaEventField(jDmaBlock);

        std::vector<json> jDmaDescs;
        for (const auto& desc : dmaBlock.gDescs()) {
            desc.assertAccessCheck();
            json jDmaDesc;
            jDmaDesc[Keys::gFromId()]         = desc.gSrcFileId().m_VarName;
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
kcc_uint64
DmaDescription::DmaBlockFromTpb::size() const
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
DmaDescription::DmaBlockToTpb::size() const
{
    kcc_uint64 numBytes = 0;
    for (const auto& desc : m_Descs) {
        numBytes += desc.gNumBytes();
    }
    return numBytes;
}

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

const char* DmaDescription::Keys::gQueueType()
{
    return "type";
}

const char* DmaDescription::Keys::gOwner()
{
    return "owner";
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
    return "#block_size";
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

