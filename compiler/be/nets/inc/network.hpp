#pragma once

#ifndef KCC_NETS_NETWORK_H
#define KCC_NETS_NETWORK_H

#include <assert.h>

#include <string>
#include <vector>
#include <map>



#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

#include "wave/inc/waveop.hpp"


namespace kcc {

namespace arch {
    class Arch;
}


namespace wave {
    class WaveOp;
    class SbAtomWaveOp;
    class SbAtomLoadWaveOp;
    class SbAtomSaveWaveOp;
    class TpbCopyWaveOp;
    class MatMulWaveOp;
    class PoolWaveOp;
    class ReciprocalWaveOp;
    class RegLoadWaveOp;
    class RegStoreWaveOp;
    class ActivationWaveOp;
    class ClipByValueWaveOp;
    class TensorWaveOp;
    class TensorTensorWaveOp;
    class TensorScalarWaveOp;
    class NopWaveOp;
}

namespace serialize {
    class SerWaveOp;
}

namespace nets {

using namespace utils;



constexpr const char* const NetKey_WaveOps              = "waveops";
constexpr const char* const NetKey_NetName              = "net_name";
constexpr const char* const NetKey_DataType             = "data_type";
constexpr const char* const NetKey_GitVersion           = "git_version";




//--------------------------------------------------------
// The whole neural net
//--------------------------------------------------------
class Network {
public:
    template<typename Archive>
    void save(Archive & archive) const;

    template<typename Archive>
    void load(Archive & archive);

    void rUseSem(bool useSem);

private:
    enum : kcc_int32 { LevelDelta = 10 };
    class LoadSaveBase;
    class Load;
    class Save;


public:
    //----------------------------------------------------------------
    Network(const arch::Arch& arch, const char* gitVersion);

    ~Network();

    void SplitReplicatedLoads();

    const std::string& gGitVersion() const {
        return m_GitVersion;
    }

#if 0
    Network(const DataType* dataType, const std::string& netName);
#endif

    bool qDoBatching() const {
        return m_DoBatching;
    }
    void rDoBatching(bool doBatch) {
        m_DoBatching = doBatch;
    }


    std::vector<wave::WaveOp*>& gWaveOps() {
        return m_WaveOps;
    }

    const std::vector<wave::WaveOp*>& gWaveOps() const {
        return m_WaveOps;
    }

    wave::WaveOp* gWaveOp(kcc_int32 waveIdx) const {
        return m_WaveOps[waveIdx];
    }

    kcc_int32 gNumberWaveops() const {
        return m_WaveOps.size();
    }


    const DataType& gDataType() const {
        return *m_DataType;
    }
    const DataType& gInDataType() const {
        return gDataType();
    }
    const std::string& gInTensorFormat() const;
    const utils::TensorParams::ShapeType& gInTensorDimensions() const;

    kcc_int32 gInDataSizeInBytes() const;
    kcc_int32 gOutDataSizeInBytes() const;

    const std::string& gName() const {
        return m_Name;
    }

    void rUseWave (bool useWave) {
        m_UseWave = useWave;
    }


    void replaceWaveops(std::vector<wave::WaveOp*>& newWaveops, bool save);
    void revertSavedWaveops();
    void ClearEvents();

    void RewireMultiOutEdgesOfMatMults();

private:
    wave::WaveOp*  findWaveOp(const std::string& prevWaveOpName);
    void levelizeByLongestPath();
    wave::NopWaveOp* rewireMultiOutEdgesOfOneMatMul(wave::MatMulWaveOp* matmulWaveop);



private:
    Network() = delete;
    Network(const Network&) = delete;

private:
    const arch::Arch&                       m_Arch;
    std::unique_ptr<DataType>               m_DataType;
    std::string                             m_Name;
    std::string                             m_GitVersion;
    std::vector<wave::WaveOp*>              m_WaveOps;
    std::vector<wave::WaveOp*>              m_SaveWaveOps;
    bool                                    m_DoBatching;
    std::map<std::string, wave::WaveOp*>    m_Name2WaveOp;
    bool                                    m_UseWave = false;
    std::unique_ptr<Load>                   m_Load;
    std::unique_ptr<Save>                   m_Save;
    bool                                    m_UseSem;
}; // Network




} // namespace nets
} // namespace kcc

#endif // KCC_NETS_NETWORK_H

