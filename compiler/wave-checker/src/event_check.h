#ifndef __EVENT_CHECK_H__
#define __EVENT_CHECK_H__

#include <vector>
#include <unordered_map>
#include <iostream>
#include "wc_common.h"
using namespace boost;
using json = nlohmann::json;

class EventChecker {
  typedef int64_t evid_t;
  typedef std::vector<edge_t> EdgeContainer_t;

  static const uint64_t NUM_EVENTS = 256;
  using eog_t = adjacency_list<vecS, vecS, bidirectionalS
    , property<vertex_index_t, std::size_t> >;
  using eog_v_t = graph_traits<eog_t>::vertex_descriptor;
  using eog_e_t = graph_traits<eog_t>::edge_descriptor;

  public:
    EventChecker(graph_t& g) : wg(g)
    {
      mEventIDUsed.resize(NUM_EVENTS);
      for(uint64_t i = 0;i < NUM_EVENTS;++i)
      {
        EdgeContainer_t* e_list = new EdgeContainer_t;
        mEventID2Edgelist.insert(
            std::pair<evid_t, EdgeContainer_t*>(i,e_list));
        mEventIDUsed[i] = false;
      }
      PreprocessWaveGraph();
    }
    ~EventChecker()
    {
      for(auto a : mEventID2Edgelist)
      {
        delete a.second;
      }
    }
    bool Run();
    const std::stringstream& get_msg() const {return mMessages;}
    friend std::ostream& operator<<(std::ostream& os, const EventChecker& evc);
  private:
    void PreprocessWaveGraph();
    bool RunEventConflictCheck(evid_t evid);
    void PrintNonOrderedPairsOfSetWait(eog_t& eog, evid_t evid);
  private:
    graph_t& wg; // Wave graph
    std::unordered_map<evid_t, EdgeContainer_t*> mEventID2Edgelist;
    std::vector<bool> mEventIDUsed;
    std::vector<eog_v_t> mVertexID2Vertex;
    // map from a vertex of EOG to an edge of WG
    std::unordered_map<eog_v_t, edge_t> mEOG_V2WG_E;
    
    std::stringstream mMessages;
    //std::ostream mMessages;
}; // EventChecker


#endif //__EVENT_CHECK_H__
