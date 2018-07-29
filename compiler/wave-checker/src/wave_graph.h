#ifndef __WAVE_GRAPH_H__
#define __WAVE_GRAPH_H__

#include <vector>
#include <map>
#include <set>
#include <list>
#include <string>
#include <boost/graph/adjacency_list.hpp>
//#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/breadth_first_search.hpp>
//#include "packages/nlohmann/json.hpp"
#include "common/aws_tonga_isa_common.h"
#include "meminfo.h"
#include "wc_common.h"
#include "event_check.h"
#include <unordered_map>

//extern po::variables_map g_cli;

using namespace boost;
using json = nlohmann::json;


class WaveOp {
  public:
  enum WaveOpType {MatMul, SBAtomLoad, SBAtomSave, Pool, Activation, ResAdd
    , Nop};
  WaveOp(std::string n, std::string wot) : name(n)
  {
    std::tie(waveop_type, engine) = ExtractWaveOpTypeEngine(wot);
  }
  enum Engine {PE, ACT, POOL, DMA, NOP};
  virtual ~WaveOp()
  {
  }
  std::string get_name() {return name;}
  WaveOpType get_waveop_type() {return waveop_type;}
  Engine get_engine() {return engine;}
  bool IsDRAMOp();
  bool IsMatMul(){ return (waveop_type == MatMul); }
  virtual size_t get_sb_in_footprint_size() {return 0;}
  virtual size_t get_sb_out_footprint_size(){return 0;}
  virtual size_t get_psum_in_footprint_size() {return 0;}
  virtual size_t get_psum_out_footprint_size() {return 0;}
  virtual std::list<AddrRange>& get_sb_in_footprint() = 0;
  virtual std::list<AddrRange>& get_sb_out_footprint() = 0;
  virtual std::list<AddrRange>& get_psum_in_footprint() = 0;
  virtual std::list<AddrRange>& get_psum_out_footprint() = 0;
  private:
  std::pair<WaveOpType, Engine> ExtractWaveOpTypeEngine (std::string& wot);
  private:
  std::string name;
  WaveOpType waveop_type;
  Engine engine;
}; // WaveOp

class NopOp : public WaveOp {
  public:
    NopOp(json& op) : WaveOp (op["waveop_name"].get<std::string>()
        , op["waveop_type"].get<std::string>())
    {}
    ~NopOp() {}
    std::list<AddrRange>& get_sb_in_footprint()
    {
      return empty_mem_footprint;
    }
    std::list<AddrRange>& get_sb_out_footprint()
    {
      return empty_mem_footprint;
    }
    std::list<AddrRange>& get_psum_in_footprint()
    {
      return empty_mem_footprint;
    }
    std::list<AddrRange>& get_psum_out_footprint()
    {
      return empty_mem_footprint;
    }
  private:
    std::list<AddrRange> empty_mem_footprint;
}; // NopOp

class MMOp : public WaveOp {
  public:
    MMMemInfo m_mi;
    MMOp (json& op) : WaveOp (op["waveop_name"].get<std::string>()
        , op["waveop_type"].get<std::string>())
                      , m_mi(extract_sb_params(op), extract_psum_params(op)
                          , op["src_sb_address"].get<tonga_addr>()
                          , op["weights_sb_address"].get<tonga_addr>()
                          , op["num_column_partitions"].get<length_t>()
                          )
    {
    }
    ~MMOp()
    {
    }
    size_t get_sb_in_footprint_size()
    {return m_mi.get_sb_in_footprint().size();}
    size_t get_sb_out_footprint_size(){return 0;}
    size_t get_psum_in_footprint_size() {return 0;}
    size_t get_psum_out_footprint_size()
    {return m_mi.get_psum_out_footprint().size();}
    std::list<AddrRange>& get_sb_in_footprint()
    {return m_mi.get_sb_in_footprint();}
    std::list<AddrRange>& get_sb_out_footprint() // Don't call this
    {
      assert(0);
      return m_mi.get_sb_in_footprint();
    }
    std::list<AddrRange>& get_psum_in_footprint() // Don't call this
    {
      assert(0);
      return m_mi.get_psum_out_footprint();
    }
    std::list<AddrRange>& get_psum_out_footprint()
    {return m_mi.get_psum_out_footprint();}
  private:
    MemInfo_Params extract_sb_params(json& op);
    MemInfo_PSUM_Params extract_psum_params(json& op);
}; // MMOp

class SBAtomOp : public WaveOp {
  public:
    //enum atom_type {SBAtomLoad, SBAtomSave};
    SBAtomMemInfo::atom_type m_atom_type;
    SBAtomMemInfo m_mi;
    SBAtomOp (json& op) : WaveOp (op["waveop_name"].get<std::string>()
        , op["waveop_type"].get<std::string>())
                               //, m_mi(op["sb_address"].get<tonga_addr>()
                                   //, op["length"].get<length_t>()
                                   //, op["start_at_mid_part"].get<bool>()
                          , m_mi(op , extract_atom_type(op)
                              )
    {
      m_atom_type = m_mi.get_atom_type();
    }
    ~SBAtomOp()
    {
    }
    size_t get_sb_in_footprint_size()
    {
      if (m_atom_type == SBAtomMemInfo::SBAtomLoad) return 0;
      else return m_mi.get_footprint().size();
    }
    size_t get_sb_out_footprint_size()
    {
      if (m_atom_type == SBAtomMemInfo::SBAtomSave) return 0;
      else return m_mi.get_footprint().size();
    }
    size_t get_psum_in_footprint_size() {return 0;}
    size_t get_psum_out_footprint_size() {return 0;}
    std::list<AddrRange>& get_sb_in_footprint()
    {
      if (m_atom_type == SBAtomMemInfo::SBAtomLoad) assert(0);
      else return m_mi.get_footprint();
    }
    std::list<AddrRange>& get_sb_out_footprint()
    {
      if (m_atom_type == SBAtomMemInfo::SBAtomSave) assert(0);
      else return m_mi.get_footprint();
    }
    std::list<AddrRange>& get_psum_in_footprint() // Don't call this
    {
      assert(0);
    }
    std::list<AddrRange>& get_psum_out_footprint() // Don't call this
    {
      assert(0);
    }
  private:
    SBAtomMemInfo::atom_type extract_atom_type(json& op);
}; // SBAtomLoadOp

class PoolActOp : public WaveOp {
  public:
    WaveOpMemInfo m_mi;
    PoolActOp (json& op) : WaveOp (op["waveop_name"].get<std::string>()
        , op["waveop_type"].get<std::string>())
                        , m_mi(extract_sb_in_params(op)
                            , extract_sb_out_params(op)
                            , extract_src_sb_addr(op)
                            , extract_dst_sb_addr(op)
                            , extract_bias_sb_addr(op)
                            , extract_bias_dtype(op)
                            , extract_psum_in_params(op)
                            , extract_psum_out_params(op)
                            , extract_bias_add_en(op))
    {
    }
    ~PoolActOp()
    {
    }
    size_t get_sb_in_footprint_size()
    {return m_mi.get_sb_in_footprint().size();}
    size_t get_sb_out_footprint_size()
    {return m_mi.get_sb_out_footprint().size();}
    size_t get_psum_in_footprint_size()
    {return m_mi.get_psum_in_footprint().size();}
    size_t get_psum_out_footprint_size()
    {return m_mi.get_psum_out_footprint().size();}
    std::list<AddrRange>& get_sb_in_footprint()
    {
      return m_mi.get_sb_in_footprint();
    }
    std::list<AddrRange>& get_sb_out_footprint()
    {
      return m_mi.get_sb_out_footprint();
    }
    std::list<AddrRange>& get_psum_in_footprint()
    {
      return m_mi.get_psum_in_footprint();
    }
    std::list<AddrRange>& get_psum_out_footprint()
    {
      return m_mi.get_psum_out_footprint();
    }
  private:
    MemInfo_Params extract_sb_in_params(json& op);
    MemInfo_Params extract_sb_out_params(json& op);
    MemInfo_PSUM_Params extract_psum_in_params(json& op);
    MemInfo_PSUM_Params extract_psum_out_params(json& op);
    tonga_addr extract_bias_sb_addr(json& op);
    tonga_addr extract_src_sb_addr(json& op);
    tonga_addr extract_dst_sb_addr(json& op);
    bool extract_bias_add_en(json& op);
    std::string extract_bias_dtype(json& op);
}; // PoolActOp

class ResAddOp : public WaveOp {
  public:
    WaveOpGenericMemInfo m_mi;
    ResAddOp(json& op) : WaveOp(op["waveop_name"].get<std::string>()
        , op["waveop_type"].get<std::string>())
                         , m_mi(
                             extract_in_params(op)
                             , extract_out_params(op)
                             , extract_in_addrs(op)
                             , extract_out_addrs(op))
    {
    }
    ~ResAddOp()
    {
    }
    size_t get_sb_in_footprint_size()
    {
      if (m_mi.available_sb_in_mi())
        return m_mi.get_sb_in_footprint().size();
      else return 0;
    }
    size_t get_sb_out_footprint_size()
    {
      if (m_mi.available_sb_out_mi())
        return m_mi.get_sb_out_footprint().size();
      else return 0;
    }
    size_t get_psum_in_footprint_size()
    {
      if (m_mi.available_psum_in_mi())
        return m_mi.get_psum_in_footprint().size();
      else return 0;
    }
    size_t get_psum_out_footprint_size()
    {
      if (m_mi.available_psum_out_mi())
        return m_mi.get_psum_out_footprint().size();
      else return 0;
    }
    std::list<AddrRange>& get_sb_in_footprint()
    {
      return m_mi.get_sb_in_footprint();
    }
    std::list<AddrRange>& get_sb_out_footprint()
    {
      return m_mi.get_sb_out_footprint();
    }
    std::list<AddrRange>& get_psum_in_footprint()
    {
      return m_mi.get_psum_in_footprint();
    }
    std::list<AddrRange>& get_psum_out_footprint()
    {
      return m_mi.get_psum_out_footprint();
    }
  private:
    std::vector<MemInfo_Params> extract_in_params(json& op);
    std::vector<MemInfo_Params> extract_out_params(json& op);
    std::vector<tonga_addr> extract_in_addrs(json& op);
    std::vector<tonga_addr> extract_out_addrs(json& op);
}; // ResAddOp

//class ActOp : public WaveOp {
  //WaveOpMemInfo m_mi;
//}; // ActOp
//using graph_t  = adjacency_list<listS, vecS, bidirectionalS, WaveOp*
//, EventEdge>;
//using vertex_t = graph_traits<graph_t>::vertex_descriptor;
//using edge_t   = graph_traits<graph_t>::edge_descriptor;

class EventChecker;

class WaveGraphChecker {
  public:
  typedef boost::graph_traits<graph_t>::in_edge_iterator ie_itr;
  typedef boost::graph_traits<graph_t>::out_edge_iterator oe_itr;
  enum OPS {LD, ST, ACT, POOL, MM, RESADD};
  enum RaceKind {WAW_SB, WAW_PSUM, RAW_SB, RAW_PSUM, WAR_SB, WAR_PSUM};

  /*
  /// Based on
  /// https://www.boost.org/doc/libs/1_64_0/libs/graph/example/dfs-example.cpp
  //class dfs_target_visitor:public boost::default_dfs_visitor {
  template<class T>
  class bfs_target_visitor:public boost::default_bfs_visitor {
    //typedef std::set<vertex_t> PathSet;
    public:
    //bfs_target_visitor(std::set<vertex_t>* pi) : m_pset(pi)
    bfs_target_visitor(T* pi) : m_pset(pi)
    {
    }
    template < typename Vertex, typename Graph >
      //void discover_vertex(vertex_t u, const graph_t& g)
      void discover_vertex(Vertex u, const Graph& g)
      {
        m_pset->insert(u);
      }
    //std::set<vertex_t>* m_pset;
    T* m_pset;
  };
  */
  public:
  WaveGraphChecker(json& j, CommandLineOptions cli);
  ~WaveGraphChecker();

  template<typename Set, typename VertexT, typename GraphT>
    static void b_search(Set* pi, VertexT v, GraphT& g)
    {
      bfs_target_visitor<Set> vis(pi);
      auto indexmap = boost::get(boost::vertex_index, g);
      auto colormap = boost::make_vector_property_map<boost::default_color_type>
        (indexmap);
      boost::queue<VertexT> buffer;

      boost::breadth_first_search(g, v, buffer, vis, colormap);
    }

  void write_graph_viz();
  bool structure_check();
  /// CheckImmNeighbors_NonDRAMOp
  /// This perfoms simple rules that a Non-DRAM waveop has to satisfy
  /// as described below.
  /// 1. Each non-dram waveop should have at least one predecessor
  /// 2. Non-dram waveop predecessors should not include Saves
  ///    2.1 This is not really an error but more like warning that does not
  //         affect functionality but do affect performance
  /// 3. Each non-dram waveop should have at least one successor
  bool CheckImmNeighbors_NonDRAMOp(vertex_t v);
  bool InputOperandCheck(vertex_t v);
  bool OutputOperandCheck(vertex_t v);
  bool CheckDuplicatedEdges(vertex_t v);
  bool RunDataRaceChecker();
  bool RunEventConflictChecker();
  const std::ostringstream& get_msg() {
    if (mCLI.event_conflict_check)
    {
      messages << mEventChecker->get_msg().str();
    }
    return messages;
  }
// __TEST_TURNON__ should be turned off when code is compiled in release mode
// It allows to get access to private memers of the class.
#ifdef __TEST_TURNON__
  std::map<std::string, WaveOp*>& test_get_mName2WaveOp()
  {return mName2WaveOp;}
  std::map<WaveOp*, vertex_t> test_get_mWaveOp2V() {return mWaveOp2V;}
  graph_t& test_get_wg() {return wg;}
  void test_call_MakeImplicitEdgesExplicit() {MakeImplicitEdgesExplicit();}
  std::list<vertex_t>& test_get_mMMops() {return mMMops;}
  std::list<vertex_t>& test_get_mLDops() {return mLDops;}
  std::list<vertex_t>& test_get_mSTops() {return mSTops;}
  std::list<vertex_t>& test_get_mACTops() {return mACTops;}
  std::list<vertex_t>& test_get_mPOOLops() {return mPOOLops;}
  json& test_get_mJson() {return mJson;}
#endif //__TEST_TURNON__

  private:
  void ConstructPathInfo(
      vertex_t u
      , std::set<vertex_t>& pathinfo
      );
  bool DataRaceChecker (
      std::list<vertex_t>& u
      , std::list<vertex_t>& v
      );
  bool DataRace(WaveOp* u, WaveOp* v);
  bool AddrSOverlap(std::list<AddrRange>& a, std::list<AddrRange>& b);
  bool AddrOverlap(AddrRange a, AddrRange b);
  std::string WaveOpType (int i)
  {
    std::string o;
    if ((OPS)i == LD) o = "SBAtomLoad";
    if ((OPS)i == ST) o = "SBAtomSave";
    if ((OPS)i == ACT) o = "Activation";
    if ((OPS)i == POOL) o = "Pool";
    if ((OPS)i == MM) o = "MatMul";
    if ((OPS)i == RESADD) o = "ResAdd";
    return o;
  }
  WaveOp* ConstructWaveOp(json& op);
  void DataRacePrint(WaveOp*u, WaveOp*v, RaceKind rk);
  void InfoPrefix() {
    if (mCLI.color) {
      messages << "\033[1;34m";
    }
    messages << "INFO: ";
    if (mCLI.color) {
      messages << "\033[0m";
    }
  }
  void WarningPrefix() {
    if (mCLI.color) {
      messages << "\033[1;34m";
    }
    messages << "WARNING: ";
    if (mCLI.color) {
      messages << "\033[0m";
    }
  }
  void ErrorPrefix() {
    if (mCLI.color) {
      messages << "\033[1;31m";
    }
    messages << "ERROR: ";
    if (mCLI.color) {
      messages << "\033[0m";
    }
  }
  void MakeImplicitEdgesExplicit();
  private:
  json& mJson;
  CommandLineOptions mCLI;
  graph_t wg;
  EventChecker* mEventChecker;

  std::map<std::string, WaveOp*> mName2WaveOp;
  std::map<WaveOp*, vertex_t> mWaveOp2V;
  std::list<vertex_t> mLDops;
  std::list<vertex_t> mSTops;
  std::list<vertex_t> mACTops;
  std::list<vertex_t> mPOOLops;
  std::list<vertex_t> mMMops;
  std::list<vertex_t> mResAddops;
  std::ostringstream messages;
}; // WaveGraphChecker
#endif //__WAVE_GRAPH_H__
