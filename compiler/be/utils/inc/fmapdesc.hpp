#pragma once

#ifndef KCC_UTILS_FMAPDESC_H
#define KCC_UTILS_FMAPDESC_H

#include <assert.h>
#include <string>

#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"

//########################################################

namespace kcc {
namespace utils {


class FmapDesc {
public:
    //----------------------------------------------------------------
    FmapDesc(kcc_int32 num_maps, kcc_int32 map_height, kcc_int32 map_width)
        : m_NumMaps(num_maps)
        , m_MapHeight(map_height)
        , m_MapWidth(map_width)
    {
        assert(num_maps > 0 && "Number FMAPs must be positive");
        assert(map_height > 0 && "FMAP height must be positive");
        assert(map_width > 0 && "FMAP width must be positive");
    }

    FmapDesc()
        : m_NumMaps(-1)
        , m_MapHeight(-1)
        , m_MapWidth(-1)
    {
    }


    //----------------------------------------------------------------
    kcc_int32 gNumMaps() const
    {
        return m_NumMaps;
    }

    //----------------------------------------------------------------
    kcc_int32 gMapWidth() const
    {
        return m_MapWidth;
    }

    //----------------------------------------------------------------
    kcc_int32 gMapHeight() const
    {
        return m_MapHeight;
    }

    //----------------------------------------------------------------
    std::string gString() const;

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
    kcc_int32 m_NumMaps;
    kcc_int32 m_MapHeight;
    kcc_int32 m_MapWidth;
};

} // namespace layers
} // namespace kcc

#endif

