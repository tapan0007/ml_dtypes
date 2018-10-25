#include "cleaner.hpp"
#include <utility>
#include <fstream>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/graph_utility.hpp>

Cleaner::Cleaner(json& wj)
{
    wavegraph_json = wj;
}
void Cleaner::TransformWavegraphJson2BoostGraph()
{
    int vtx_id = 0;
    mVtx2Wop.resize(wavegraph_json["waveops"].size());
    int num_edges = 0;
    for (auto op : wavegraph_json["waveops"])
    {
        mVtx2Wop[vtx_id] = op;
        json* j = &op;
        mWopName2Vtx.insert(
            std::make_pair(op["waveop_name"].get<std::string>(),vtx_id)
            );
        for (auto p : op["previous_waveops"])
        {
            add_edge(mWopName2Vtx[p], vtx_id, wavegraph_boost);
            num_edges++;
        }
        vtx_id++;
    }
    std::cout << "INFO:Num Edges = " << num_edges << std::endl;
}

Cleaner::~Cleaner()
{
}

void Cleaner::TransformBoostGraph2WavegraphJson(
    std::map<vertex_t, vertex_t>& g_to_tr
    )
{

    std::pair<graph_t::vertex_iterator, graph_t::vertex_iterator> vs =
        vertices(wavegraph_boost);
    graph_t::vertex_iterator v_itr;
    std::list<json> waveops;
    std::map<vertex_t, vertex_t> tr_to_g;
    for (auto e: g_to_tr)
    {
        tr_to_g[e.second] = e.first;
    }
    uint64_t num_erased_edges = 0;
    for (v_itr = vs.first;v_itr != vs.second;++v_itr)
    {
        std::set<std::string> preds;
        vertex_t tr_v = g_to_tr[*v_itr];
        json wop = mVtx2Wop[*v_itr];
        for (std::string p_name : wop["previous_waveops"])
        {
            preds.insert(p_name);
        }
        std::pair<in_edge_itr, in_edge_itr> in_es =
            in_edges(tr_v, tr_wavegraph_boost);
        in_edge_itr ie_itr;
        std::set<std::string> new_preds;
        for(ie_itr = in_es.first;ie_itr != in_es.second;++ie_itr)
        {
            vertex_t p = source(*ie_itr, tr_wavegraph_boost);
            json p_wop = mVtx2Wop[tr_to_g[p]];
            auto p_wop_name = p_wop["waveop_name"].get<std::string>();
            new_preds.insert(p_wop_name);
        }
        json new_prev = json::array();
        for (auto org_p : preds)
        {
            if (new_preds.find(org_p) != new_preds.end())
            {
                new_prev.push_back(org_p);
            } else {
                num_erased_edges++;
            }
        }
        wop["previous_waveops"] = new_prev;
        waveops.push_back(wop);
    }
    std::cout << "INFO: " << num_erased_edges << " edges are removed"
        << std::endl;
    wavegraph_json["waveops"] = waveops;
}

json& Cleaner::RunTR()
{
    TransformWavegraphJson2BoostGraph();
    std::map<vertex_t, vertex_t> g_to_tr;
    std::vector<size_t> id_map(num_vertices(wavegraph_boost));
    std::iota(id_map.begin(), id_map.end(), 0u);

    transitive_reduction(
        wavegraph_boost
        , tr_wavegraph_boost
        , make_assoc_property_map(g_to_tr)
        , id_map.data()
        );
    TransformBoostGraph2WavegraphJson(g_to_tr);
    return wavegraph_json;
}

int main(int argc, char *argv[])
{
    enum ERROR_TYPE
    {
        OK = 0,
        ERROR = 1,
        NOINFILE = 2,
        NOOUTFILE = 3
    };
    int err = OK;
    po::options_description desc{"Options"};
    desc.add_options()
        ("help,h", "Help")
        ("wavegraph, w", po::value<std::string>()
            , "Input wavegraph in json format")
        ("fileout", po::value<std::string>()
            , "Name and location of an output wavegraph json file");
    po::variables_map cli;
    po::store(po::parse_command_line(argc, argv, desc), cli);

    if (cli.count("help") || !cli.size() || !cli.count("wavegraph"))
    {
        std::cout << desc << std::endl;
    }
    else
    {
        std::ifstream in_wave(cli["wavegraph"].as<std::string>());
        if (in_wave.is_open())
        {
            json j;
            in_wave >> j;
            Cleaner c(j);
            json o_j = c.RunTR();
            std::ofstream o(cli["fileout"].as<std::string>());
            if (o.is_open())
            {
                o << std::setw(4) << o_j << std::endl;
            } else {
                err = NOOUTFILE;
            }
        } else {
            err = NOINFILE;
        }
    }
    return err;
}
