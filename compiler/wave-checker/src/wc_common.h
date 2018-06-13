#ifndef __WC_COMMON_H__
#define __WC_COMMON_H__

#include <boost/graph/adjacency_list.hpp>
#include <boost/program_options.hpp>
#include <boost/graph/breadth_first_search.hpp>

#include <sstream>
#include <iostream>

#include "packages/nlohmann/json.hpp"

#define NO_EVENT -1

class WaveOp;
struct EventEdge {
  int64_t id;
  uint8_t wait_mode;
  uint8_t set_mode;
}; // Event

struct CommandLineOptions {
  bool color;
  bool structure_check;
  bool data_race_check;
  bool event_conflict_check;
};

/// Based on
/// https://www.boost.org/doc/libs/1_64_0/libs/graph/example/dfs-example.cpp
//class dfs_target_visitor:public boost::default_dfs_visitor {
template<class T>
class bfs_target_visitor:public boost::default_bfs_visitor {
  public:
    bfs_target_visitor(T* pi) : m_pset(pi)
  {
  }
  template < typename Vertex, typename Graph >
    void discover_vertex(Vertex u, const Graph& g)
    {
      m_pset->insert(u);
    }
  T* m_pset;
};


namespace po = boost::program_options;
using namespace boost;
using json = nlohmann::json;

using graph_t  = adjacency_list<listS, vecS, bidirectionalS, WaveOp*
, EventEdge>;
using vertex_t = graph_traits<graph_t>::vertex_descriptor;
using edge_t   = graph_traits<graph_t>::edge_descriptor;

#endif //__WC_COMMON_H__
