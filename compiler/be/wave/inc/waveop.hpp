#pragma once

#ifndef KCC_WAVE_WAVEOP_H
#define KCC_WAVE_WAVEOP_H


#include <string>
#include <vector>
#include <assert.h>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"


namespace kcc {

namespace layers {
    class Layer;
}
namespace nets {
    class Network;
}

namespace wave {

using namespace utils;

//--------------------------------------------------------
// The base class of all wave.
//--------------------------------------------------------
class WaveOp { // abstract class
protected:

    //----------------------------------------------------
public:
    class Params;

protected:
    //----------------------------------------------------------------
    WaveOp(const WaveOp::Params& params, const std::vector<WaveOp*>& prevWaveOps);

    virtual ~WaveOp()
    {}

private:
    WaveOp() = delete;
    WaveOp(const WaveOp&) = delete;

    WaveOp& operator= (const WaveOp&) const = delete;

protected:

    virtual bool verify() const = 0;

public:

    //----------------------------------------------------------------
    virtual bool qMatMultWaveOp() const {
        return false;
    }
    virtual bool qSbAtomtWaveOp() const {
        return false;
    }

    //----------------------------------------------------------------
    const DataType& gDataType() const;

    //----------------------------------------------------------------
    std::string gName() const {
        return m_Name;
    }

    layers::Layer* gLayer() const {
        return m_Layer;
    }


protected:
    std::string             m_Name          = "";
    std::vector<WaveOp*>    m_PrevWaveOps;
    FmapDesc                m_OfmapDesc;
    layers::Layer*          m_Layer         = nullptr;
private:
}; // class WaveOp


class WaveOp::Params {
public:
    std::string             m_WaveOpName    = "";
    FmapDesc                m_OfmapDesc;
    layers::Layer*          m_Layer         = nullptr;
};


} // namespace wave
} // namespace kcc

#endif // KCC_WAVE_WAVEOP_H

