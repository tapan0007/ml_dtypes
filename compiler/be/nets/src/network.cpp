#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <map>


#include "utils/inc/asserter.hpp"
#include "arch/inc/arch.hpp"


#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/tpbcopywaveop.hpp"
#include "wave/inc/waveedge.hpp"

#include "nets/inc/loadsplitter.hpp"
#include "nets/inc/network.hpp"
#include "nets/inc/network_load.hpp"
#include "nets/inc/network_save.hpp"

namespace kcc {

/*
namespace wave {
    class SbAtomLoadWaveOp;
    class SbAtomSaveWaveOp;
    class MatMulWaveOp;
}
*/

namespace nets {

//--------------------------------------------------------
Network::Network(const arch::Arch& arch, const char* gitVersion)
    : m_Arch(arch)
    , m_DataType(nullptr)
    , m_Name()
    , m_GitVersion(gitVersion)
    , m_DoBatching(false)
    , m_Load(std::make_unique<Load>(*this))
    , m_Save(std::make_unique<Save>(*this))
{}


Network::~Network() = default;




//--------------------------------------------------------
wave::WaveOp*
Network::findWaveOp(const std::string& waveOpName)
{
    wave::WaveOp* waveOp = m_Name2WaveOp[waveOpName];
    Assert(waveOp, "Could not find WaveOp ", waveOpName);
    return waveOp;
}

//--------------------------------------------------------
void
Network::revertSavedWaveops()
{
    std::vector<wave::WaveOp*> empty;

    Assert(m_SaveWaveOps.size() > 0, "Saved waveops empty in revert");
    std::swap(m_SaveWaveOps, m_WaveOps);
    std::swap(m_SaveWaveOps, empty);
}

//--------------------------------------------------------
void
Network::replaceWaveops(std::vector<wave::WaveOp*>& newWaveops, bool save)
{
    const kcc_int32 numWaveops = newWaveops.size();
    for (kcc_int32 k = 0; k < numWaveops; ++k) {
        newWaveops[k]->rOrder(k);
    }
    if (save) {
        Assert(m_SaveWaveOps.size()==0, "Saved waveops not empty");
        std::swap(m_WaveOps, m_SaveWaveOps);
    }
    std::swap(newWaveops, m_WaveOps);
}

void
Network::ClearEvents()
{
    const kcc_int32 numWaveops = gNumberWaveops();
    for (kcc_int32 waveopIdx = 0; waveopIdx < numWaveops; ++waveopIdx) {
        const auto waveop = gWaveOp(waveopIdx);
        for (auto succEdge : waveop->gSuccWaveEdges()) {
            succEdge->clearEvent();
        }
    }
}

//--------------------------------------------------------
const std::string&
Network::gInTensorFormat() const
{
    static const std::string emptyStr;

    for (auto waveop : m_WaveOps) {
        const auto sbLoadWaveop =
                dynamic_cast<const wave::SbAtomLoadWaveOp*>(waveop);
        if (sbLoadWaveop && ! sbLoadWaveop->qContainWeights()) {
            return sbLoadWaveop->gRefFileFormat();
        }
    }
    Assert(false, "Network::gTensorFormat: did not find IFMAP Load waveop");
    return emptyStr;
}

//--------------------------------------------------------
const utils::TensorParams::ShapeType&
Network::gInTensorDimensions() const
{
    static const utils::TensorParams::ShapeType badDim = {{ -1, -1, -1, -1, -1 }};

    for (auto waveop : m_WaveOps) {
        const auto sbLoadWaveop =
                dynamic_cast<const wave::SbAtomLoadWaveOp*>(waveop);
        if (sbLoadWaveop && ! sbLoadWaveop->qContainWeights()) {
            return sbLoadWaveop->gRefFileShape();
        }
    }
    Assert(false, "Network::gTensorDimensions: did not find IFMAP Load waveop");
    return badDim;
}


//--------------------------------------------------------
kcc_int32
Network::gInDataSizeInBytes() const
{
    kcc_int64 inSizeInBytes = gDataType().gSizeInBytes();
    const auto& refShape(gInTensorDimensions());

    for (auto n : refShape) {
        inSizeInBytes *= n;
    }
    return inSizeInBytes;
}

//--------------------------------------------------------
kcc_int32
Network::gOutDataSizeInBytes() const
{
    for (auto waveop : m_WaveOps) {
        const auto sbSaveWaveop =
                dynamic_cast<const wave::SbAtomSaveWaveOp*>(waveop);
        if (sbSaveWaveop) {
            kcc_int64 outSizeInBytes = gDataType().gSizeInBytes();
            for (auto n : sbSaveWaveop->gRefFileShape()) {
                outSizeInBytes *= n;
            }
            return outSizeInBytes;
        }
    }
    Assert(false, "Network::gOutDataSizeInBytes: did not find IFMAP Load waveop");
    return -1;
}

//--------------------------------------------------------
void
Network::rUseSem(bool useSem)
{
    m_UseSem = useSem;
}

//--------------------------------------------------------
void
Network::SplitReplicatedLoads()
{
    LoadSplitter splitter(*this);
    splitter.SplitReplicatedLoads();
}

//--------------------------------------------------------
void
Network::MarkTmpLoads()
{
    // Collect all Ref files that are saved into by some tmp AtomSave
    std::set<std::string> savedTmpRefFiles;
    std::set<std::string> loadedTmpRefFiles;

    for (auto waveop : m_WaveOps) {
        auto saveWop = dynamic_cast<const wave::SbAtomSaveWaveOp*>(waveop);
        if (! saveWop) {
            continue;
        }
        savedTmpRefFiles.insert(saveWop->gRefFileName());
    }

    for (auto waveop : m_WaveOps) {
        auto loadWop = dynamic_cast<wave::SbAtomLoadWaveOp*>(waveop);
        if (! loadWop) {
            continue;
        }
        loadedTmpRefFiles.insert(loadWop->gRefFileName());
    }

    for (auto waveop : m_WaveOps) {
        auto saveWop = dynamic_cast<wave::SbAtomSaveWaveOp*>(waveop);
        if (! saveWop) {
            continue;
        }
        if (loadedTmpRefFiles.find(saveWop->gRefFileName()) != loadedTmpRefFiles.end()) {
            saveWop->rTmpBuffer(true);
            std::cout << "Save " << saveWop->gName() << " marked as tmp_buf\n";
        } else {
            std::cout << "Save " << saveWop->gName() << " NOT marked as tmp_buf (i.e. is input)\n";
        }
    }

    for (auto waveop : m_WaveOps) {
        auto loadWop = dynamic_cast<wave::SbAtomLoadWaveOp*>(waveop);
        if (! loadWop) {
            continue;
        }
        if (savedTmpRefFiles.find(loadWop->gRefFileName()) != savedTmpRefFiles.end()) {
            loadWop->rTmpBuffer(true);
            std::cout << "Load " << loadWop->gName() << " marked as tmp_buf\n";
        } else {
            std::cout << "Load " << loadWop->gName() << " NOT marked as tmp_buf (i.e. is input)\n";
        }
    }
}

}}


