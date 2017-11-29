#ifndef KCC_NETS_NETWORK_H
#define KCC_NETS_NETWORK_H


#include <string>
#include <vector>
#include <assert.h>

using std::string;
using std::vector;


#include "types.hpp"
#include "datatype.hpp"
#include "fmapdesc.hpp"


namespace kcc {

namespace layers {
    class Layer;
}

namespace nets {

using namespace utils;
using layers::Layer;

//--------------------------------------------------------
class Network {
public:
    //----------------------------------------------------------------
    Network(const DataType& dataType, const string& netName);

    const DataType& gDataType() const {
        return m_DataType;
    }

    void addLayer(Layer* layer);
    

private:
    const DataType& m_DataType;
    string m_Name;
    vector<Layer*> m_Layers;
}; // class Layer



} // namespace nets
} // namespace kcc

#endif // KCC_NETS_NETWORK_H

