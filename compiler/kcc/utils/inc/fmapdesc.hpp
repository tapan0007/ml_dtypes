#pragma once

#ifndef KCC_UTILS_FMAPDESC_H
#define KCC_UTILS_FMAPDESC_H

#include <assert.h>
#include <string>
using std::string;

#include "types.hpp"
#include "datatype.hpp"

//########################################################

namespace kcc {
namespace utils {


class FmapDesc {
public:
    //----------------------------------------------------------------
    FmapDesc(int32 num_maps, int32 map_height, int32 map_width)
        : m_NumMaps(num_maps)
        , m_MapHeight(map_height)
        , m_MapWidth(map_width)
    {
        assert(num_maps > 0);
        assert(map_height > 0);
        assert(map_width > 0);
    }


    //----------------------------------------------------------------
    int32 gNumMaps() const
    {
        return m_NumMaps;
    }

    //----------------------------------------------------------------
    int32 gMapWidth() const
    {
        return m_MapWidth;
    }

    //----------------------------------------------------------------
    int32 gMapHeight() const
    {
        return m_MapHeight;
    }

    //----------------------------------------------------------------
    string gString() const;

    //----------------------------------------------------------------
    bool operator== (const FmapDesc& rhs) const
    {
        return gNumMaps() == rhs.gNumMaps()
            && gMapWidth() == rhs.gMapWidth()
            && gMapHeight() == rhs.gMapHeight();
    }

    //----------------------------------------------------------------
    bool operator!= (const FmapDesc& rhs) const
    {
        return ! (*this == rhs);
    }


private:
    FmapDesc() = delete;

private:
    int32 m_NumMaps;
    int32 m_MapHeight;
    int32 m_MapWidth;
};

} // namespace layers
} // namespace kcc

#endif

