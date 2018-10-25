#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/transitive_reduction.hpp>
#include <iostream>
#include "packages/nlohmann/json.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace boost;
using json = nlohmann::json;
using graph_t = adjacency_list<listS, vecS, bidirectionalS>;
using vertex_t = graph_traits<graph_t>::vertex_descriptor;
using edge_t = graph_traits<graph_t>::edge_descriptor;
using vertex_itr = graph_traits<graph_t>::vertex_iterator;
using edge_itr = graph_traits<graph_t>::edge_iterator;
using in_edge_itr = graph_traits<graph_t>::in_edge_iterator;

class Cleaner {
    public:
    Cleaner(json& wavegraph_json);
    ~Cleaner();
    json& RunTR();
    private:
    void TransformWavegraphJson2BoostGraph();
    void TransformBoostGraph2WavegraphJson(
        std::map<vertex_t, vertex_t>& g_to_tr
        );
    private:
    json wavegraph_json;
    graph_t wavegraph_boost;
    graph_t tr_wavegraph_boost;
    std::vector<json> mVtx2Wop;
    std::unordered_map<std::string, vertex_t> mWopName2Vtx;
};