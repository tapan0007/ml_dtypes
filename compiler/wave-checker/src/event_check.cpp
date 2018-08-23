#include "event_check.h"
#include "wave_graph.h"
#include <unordered_set>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <iterator>

#include <cassert>

std::ostream& operator<<(std::ostream& os, const EventChecker& evc)
{
  return os << evc.mMessages.str();
}

void EventChecker::PreprocessWaveGraph()
{
  graph_t::edge_iterator e_itr, e_end;
  for (boost::tie(e_itr, e_end) = boost::edges(wg); e_itr != e_end; ++e_itr)
  {
    // If event is not in set and wait mode 0, then the edge is not actually
    // assigned to the id
    if (wg[*e_itr].wait_mode && wg[*e_itr].set_mode)
    {
      mEventID2Edgelist[wg[*e_itr].id]->push_back(*e_itr);
      mEventIDUsed[wg[*e_itr].id] = true;
    }
  }
}

bool EventChecker::RunEventConflictCheck(evid_t evid)
{
  bool conflict = false;
  if (mEventID2Edgelist[evid]->size())
  {
    eog_t eog;
    for(int i = 0;i < mEventID2Edgelist[evid]->size();++i)
    {
      eog_v_t eog_v = boost::add_vertex(i, eog);
      //mEOG_V2WG_E.insert(std::pair<eog_v_t, edge_t>
          //(eog_v,(*mEventID2Edgelist[evid])[i]));
      mEOG_V2WG_E[eog_v] = (*mEventID2Edgelist[evid])[i];
    }
    // Copy an input wave graph into a local wave graph where
    // topological sort will be performed. Note that the local copy
    // will be modified, which is the reason that we copy original wave graph.
    graph_t wg_copy;
    boost::copy_graph<graph_t, graph_t>(wg, wg_copy);
    // Since local copy has its own instances of vertex descriptor and
    // edge descriptor, we construct a map to match one descriptor in an
    // original wave graph with another descriptor in the local copy.
    // For example, when we want to get a vertex with vertex descriptor instance
    // of an original wave graph, we use the map container to get an instance
    // of vertex descriptor of the local copy.
    std::unordered_map<WaveOp*, vertex_t> l_vid2v;
    graph_t::vertex_iterator v_itr, v_end;
    for (boost::tie(v_itr, v_end) = boost::vertices(wg_copy);
        v_itr != v_end; ++v_itr)
    {
      l_vid2v[wg_copy[*v_itr]] = *v_itr;
    }

    // We remove edges associated with evid in order to compute dependency
    // path (DP). If there is an edge with evid between two edges for which
    // we compute a DP, then we don't consider the path as a DP. Thus, rather
    // than computing every path and then examining if there is such an edge
    // between two target edges, we simply remove all the edges associated with
    // evid except the target edges. In that way, we don't redundantly enumerate
    // paths associated with evid except the target edges.
    for(auto e : *mEventID2Edgelist[evid])
    {
      //std::cout << "e = " << e << std::endl;
      WaveOp* h_wg = wg[boost::source(e, wg)];
      WaveOp* t_wg = wg[boost::target(e, wg)];
      // If two wave ops are executed on the same engine,
      // then we consider them ordered. Thus, even if
      // evid is assigned to e, we do not remove the edge.
      if ((h_wg->get_engine() != t_wg->get_engine()) ||
          (h_wg->get_engine() == WaveOp::DMA
           && t_wg->get_engine() == WaveOp::DMA))
      {
        vertex_t h = l_vid2v[wg[boost::source(e, wg)]];
        vertex_t t = l_vid2v[wg[boost::target(e, wg)]];
        edge_t e_;
        bool found;
        boost::tie(e_, found) = boost::edge(h, t, wg_copy);
        boost::remove_edge(e_, wg_copy);
      }
    }

    // Check if there exists a DP from Tail(e_i^evid) to
    // Head(e_j^evid) and Tail(e_j^evid) (i.e. Tail(e_i^evid) -> Head(e_j^evid),
    // Tail(e_i^evid) -> Tail(e_j^evid))
    typedef std::unordered_set<vertex_t> set_v;
    std::cout << "EVID = " << evid << std::endl;
    for(int i = 0;i < (int)(boost::num_vertices(eog));++i)
    {
      edge_t source_edge = (*mEventID2Edgelist[evid])[i];
      vertex_t start = boost::target(source_edge, wg); // Head(e_i^evid)

      set_v pi_v;
      WaveGraphChecker::b_search<set_v, vertex_t, graph_t>(&pi_v,start,wg_copy);
      for(int j = 0;j < boost::num_vertices(eog);++j)
      {
        if (i != j)
        {
          edge_t target_edge = (*mEventID2Edgelist[evid])[j];
          vertex_t tail = boost::source(target_edge, wg);
          tail = l_vid2v[wg[tail]];
          vertex_t head = boost::target(target_edge, wg);
          head = l_vid2v[wg[head]];
          // Tail(e_j^evid) and Head(e_j^evid)
          if (pi_v.find(tail) != pi_v.end() && pi_v.find(head) != pi_v.end())
          {
            std::cout << wg[start]->get_name()
              << " to T:" << wg[tail]->get_name() << std::endl;
            std::cout << wg[start]->get_name()
              << " to H:" << wg[head]->get_name() << std::endl;
            std::cout << "EVID "<< evid
              << ": Add edge from " << i << " to " << j << std::endl;
            boost::add_edge(boost::vertex(i, eog), boost::vertex(j, eog), eog);
          }
        }
      }
    }

    // Create source and sink vertices and add them to eog
    eog_v_t s =
      boost::add_vertex(boost::num_vertices(eog), eog); // Source vertex of eog
    eog_v_t t =
      boost::add_vertex(boost::num_vertices(eog), eog); // Sink vertex of eog
    eog_t::vertex_iterator eog_v_itr, eog_v_itr_end;
    for(boost::tie(eog_v_itr, eog_v_itr_end) = boost::vertices(eog);
        eog_v_itr != eog_v_itr_end;++eog_v_itr)
    {
      if (!boost::in_degree(*eog_v_itr, eog) && s != *eog_v_itr)
      {
        boost::add_edge(s, *eog_v_itr, eog);
      }
      if (!boost::out_degree(*eog_v_itr, eog) && t != *eog_v_itr)
      {
        boost::add_edge(*eog_v_itr, t, eog);
      }
    }

    boost::write_graphviz(std::cout, eog, [&] (std::ostream& out, eog_v_t v) {
        //out << "[label=\"" << (&eog[v]) << "\"]";
        out << "[label=\"" << (v) << "\"]";
        });
    std::cout << std::flush;

    std::vector<eog_v_t> c;
    boost::topological_sort(eog, std::back_inserter(c));

    // Check the uniqueness of topological sort of eog
    int loc_v = 0;
    eog_v_t prev_v;
    std::cout << "TOPO for evid " << evid << std::endl;
    for(auto ii : boost::adaptors::reverse(c))
    {
      std::cout << ii << " ";
    }
    std::cout << std::endl;
    for(auto ii : boost::adaptors::reverse(c))
    {
      if (loc_v)
      {
        eog_e_t e_;
        bool found;
        boost::tie(e_, found) = boost::edge(prev_v, ii, eog);
        if (!found)
        {
          conflict = true;
          break;
        }
      }
      prev_v = ii;
      loc_v++;
    }
    if (conflict)
    {
      mMessages << "ERROR: Event " << evid
        << " could have conflicts" << std::endl;
      boost::clear_vertex(t, eog);
      boost::remove_vertex(t, eog);
      boost::clear_vertex(s, eog);
      boost::remove_vertex(s, eog);
      PrintNonOrderedPairsOfSetWait(eog, evid);
    }
  }

  return conflict;
}

bool EventChecker::Run()
{
  bool result = false;
  mMessages << "INFO: Starting Event Conflict Check" << std::endl;
  for(int i = 0;i < NUM_EVENTS;++i)
  {
    bool res;
    res = RunEventConflictCheck(i);
    result |= res;
  }
  mMessages << "INFO: Finished Event Conflict Check" << std::endl;
  return result;
}

void EventChecker::PrintNonOrderedPairsOfSetWait(
    eog_t& eog
    , evid_t evid
    )
{
  std::unordered_map<eog_v_t, std::unordered_set<eog_v_t>* > v2reachable_vs;

  eog_t::vertex_iterator v_itr, v_end;
  for (boost::tie(v_itr, v_end) = boost::vertices(eog);
        v_itr != v_end; ++v_itr)
  {
    std::unordered_set<eog_v_t>* v_set = new std::unordered_set<eog_v_t>;
    //v2reachable_vs.insert(std::pair<eog_v_t, std::unordered_set<eog_v_t> >(
    //      *v_itr, v_set));
    v2reachable_vs.insert({*v_itr, v_set});
    //std::cout << (&v_set) << std::endl;
    WaveGraphChecker::b_search<std::unordered_set<eog_v_t>, eog_v_t, eog_t>
      (v_set, *v_itr, eog);
    if (*v_itr == 0 && evid == 7)
    {
      auto a = v2reachable_vs[*v_itr];
      std::cout << "Vertices reachable from " << *v_itr << std::endl;
      for(auto i : *a)
      {
        std::cout << i << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "size of v_set = " << v_set->size() << std::endl;
  }
  std::cout << "size of v2reachable_vs = " << v2reachable_vs.size() <<std::endl;
  eog_t::vertex_iterator v_itr2;
  for (boost::tie(v_itr, v_end) = boost::vertices(eog);
        v_itr != v_end; ++v_itr)
  {
    std::unordered_set<eog_v_t>* left2right; // u -> v
    left2right = v2reachable_vs[*v_itr];
    v_itr2 = v_itr;
    ++v_itr2;
    std::cout << "-"<<left2right << std::endl;
    std::cout << "-size of left2right = " << left2right->size() << std::endl;
    if (*v_itr == 0 && evid == 7)
    {
      std::cout << "Vertices reachable from " << *v_itr << std::endl;
      for(auto i : *left2right)
      {
        std::cout << i << " ";
      }
      std::cout << std::endl;
    }
    for (;v_itr2 != v_end; ++v_itr2)
    {
      if (*v_itr != *v_itr2)
      {
        std::unordered_set<eog_v_t>* right2left; // u <- v
        right2left = v2reachable_vs[*v_itr2];
        if (left2right->find(*v_itr2) == left2right->end() &&
            right2left->find(*v_itr) == right2left->end())
        {
          edge_t e1 = mEOG_V2WG_E[*v_itr];
          edge_t e2 = mEOG_V2WG_E[*v_itr2];
          //std::cout << "ERROR : (Event " << evid
          std::cout << "evid " << evid << " : "
            << *v_itr <<", "<<*v_itr2 << std::endl;
          mMessages << "\t"
            << "(" << wg[boost::source(e1, wg)]->get_name()
            << "->" << wg[boost::target(e1, wg)]->get_name() << ")"
            << " : "
            << "(" << wg[boost::source(e2, wg)]->get_name()
            << "->" << wg[boost::target(e2, wg)]->get_name() << ")"
            << std::endl;
        }
      }
    }
  }
}
