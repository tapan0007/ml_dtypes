#ifndef KCC_LAYERS_CONVLAYER_H
#define KCC_LAYERS_CONVLAYER_H

#include "types.hpp"
#include "datatype.hpp"
#include "fmapdesc.hpp"
#include "subsamplelayer.hpp"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

//--------------------------------------------------------
class ConvLayer : public SubSampleLayer {
public:
    static const char* const filter_file_key;
    static const char* const kernel_format_key;

    //----------------------------------------------------------------
    ConvLayer(const Params& params, Layer* prev_layer, int num_ofmaps,
        const std::tuple<int,int>& stride, const std::tuple<int,int>& kernel,
        const char* filterFileName, const char* filterTensorDimSemantics);

#if 0
    //----------------------------------------------------------------
    def gJson(self):

    @classmethod
    def constructFromJson(cls, layerDict, nn):
#endif

    //----------------------------------------------------------------
    string gString() const;

    //----------------------------------------------------------------
    bool verify() const;

    //----------------------------------------------------------------
    string gFilterFileName() const {
        return m_FilterFileName;
    }

    //----------------------------------------------------------------
    string gFilterTensorDimSemantics() const {
        return m_FilterTensorDimSemantics;
    }

    //----------------------------------------------------------------
    const char* gTypeStr() const {
        return "Conv";
    }

    //----------------------------------------------------------------
    bool qPassThrough() const {
        return false;
    }

    //----------------------------------------------------------------
    int gNumberWeights() const;

    //----------------------------------------------------------------
    long gNumberWeightsPerPartition() const;

    //----------------------------------------------------------------
    bool qConvLayer() const {
        return true;
    }

private:
    string m_FilterFileName;
    string m_FilterTensorDimSemantics;

};

} // namespace layers
} // namespace kcc


#endif // KCC_LAYERS_CONVLAYER_H

