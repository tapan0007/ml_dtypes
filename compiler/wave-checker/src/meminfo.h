/// WGC : Wave Graph Checker
#ifndef __WGC_MEMINFO_H__
#define __WGC_MEMINFO_H__
#include "packages/nlohmann/json.hpp"
#include "common/aws_tonga_isa_common.h"
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <stdlib.h>

using json = nlohmann::json;

typedef uint32_t length_t;
typedef int num_t;
typedef int step_t;
//typedef uint32_t pattern_t;

struct AddrRange {
  tonga_addr begin;
  tonga_addr end;
  AddrRange(tonga_addr b, tonga_addr e) : begin(b), end(e)
  {
  }
  void print_ar(bool endl)
  {
    for(int i = 0;i < begin;++i)
    {
      std::cout << " ";
    }
    for(int i = 0;i <= (end - begin);++i)
    {
      std::cout << "-";
    }
    if (endl)
    {
      std::cout << std::endl;
    }
  }
  void print_text_ar()
  {
    std::cout << "[" << begin << ", " << end << "]" << std::endl;
  }
  static void print (uint32_t skip_cnt, uint32_t cnt)
  {
    for(int i = 0;i < skip_cnt;++i)
    {
      std::cout << " ";
    }
    for(int i = 0;i < cnt;++i)
    {
      std::cout << "-";
    }
  }
  // T should be STL container
  template<class T>
  static void print_ars(T& ars)
  {
    tonga_addr prev_end = 0;
    for(auto ar : ars)
    {
      if (ar.begin)
        print(ar.begin - prev_end - 1, ar.end - ar.begin + 1);
      else
        print(0, ar.end - ar.begin + 1);
      prev_end = ar.end;
    }
    std::cout << std::endl;
  }
  template<class T>
  static void print_text_ars(T& ars)
  {
    for(auto a : ars)
    {
      std::cout << "[" << a.begin << ", " << a.end << "]" << " ";
    }
    std::cout << std::endl;
  }
};//AddrRange

struct MemInfo_Params {
  num_t nx, ny, nz, nw;
  step_t sx, sy, sz, sw;
  std::string dtype;
  bool enable;
  bool psum;
  MemInfo_Params(int x, int y, int z, int w, int _sx, int _sy, int _sz, int _sw
      , std::string dt, bool en) : nx(x), ny(y), nz(z), nw(w), sx(_sx), sy(_sy)
                                   , sz(_sz), sw(_sw), dtype(dt)
  {}
  MemInfo_Params() {}
}; // MemInfo_Params
struct MemInfo_PSUM_Params : public MemInfo_Params {
  int pbid;
}; // MemInfo_PSUM_Params

struct MemInfo {
  static const tonga_addr SB_END_ADDR = 96 * 1024 - 1;
  static const tonga_addr PSUM_BANK_SIZE = 2048; // in Bytes
  uint32_t x_num;
  uint32_t x_step;
  uint32_t y_num;
  uint32_t y_step;
  uint32_t z_num;
  uint32_t z_step;
  uint32_t w_num;
  uint32_t w_step;
  tonga_addr start_addr;
  uint32_t elem_size;
  bool enable;
  std::list<AddrRange> mem_footprints;

  MemInfo(MemInfo_Params mi_params, int st_addr)
    : x_num(mi_params.nx), x_step(mi_params.sx), y_num(mi_params.ny)
      , y_step(mi_params.sy), z_num(mi_params.nz), z_step(mi_params.sz)
      , w_num(mi_params.nw), w_step(mi_params.sw)
      , start_addr(st_addr), enable(mi_params.enable)
  {
    elem_size = compute_elem_size(mi_params.dtype);
  }

  void merge_addr_range(AddrRange);

  static uint32_t compute_elem_size(std::string dtype)
  {
    uint32_t es = 1;
    if (!dtype.compare("float16")) es = 2;
    else if (!dtype.compare("float32")) es = 4;
    else if (!dtype.compare("int16")) es = 2;
    else assert(0);
    return es;
  }

  tonga_addr last_begin(int64_t start_addr, int step, int elem_size, int num_elem)
  {return (tonga_addr)(start_addr + elem_size * step * (num_elem - 1)); }
  tonga_addr last_end(int64_t start_addr, int step, int elem_size, int num_elem)
  {
    return (last_begin(start_addr,step,elem_size,num_elem)
      + (tonga_addr)elem_size - 1);
  }
  inline tonga_addr last_w_begin()
  { return last_begin(start_addr, w_step, elem_size, w_num); }
  inline tonga_addr last_w_end()
  { return last_end(start_addr, w_step, elem_size, w_num); }
  //inline tonga_addr last_z_begin()
  //{ return last_begin(start_addr, z_step, elem_size, z_num); }
  //inline tonga_addr last_z_end()
  //{ return last_end(start_addr, z_step, elem_size, z_num); }
  inline tonga_addr last_z_begin()
  { return (last_w_begin() + elem_size * z_step * (z_num - 1)); }
  inline tonga_addr last_z_end()
  { return (last_z_begin() + elem_size - 1); }
  inline tonga_addr last_y_begin()
  { return (last_z_begin() + elem_size * y_step * (y_num - 1)); }
  inline tonga_addr last_y_end()
  { return (last_y_begin() + elem_size - 1); }
  inline tonga_addr last_x_begin()
  { return (last_y_begin() + elem_size * x_step * (x_num - 1)); }
  inline tonga_addr last_x_end()
  { return (last_x_begin() + elem_size - 1); }
  void compute_footprint()
  {
    if (enable)
    {
      //tonga_addr end = 
        //(std::max(last_z_end(), std::max(last_y_end(), last_x_end())));
      tonga_addr end = 
        (std::max(std::max(last_w_end(), last_z_end())
                  , std::max(last_y_end(), last_x_end())));
      AddrRange addr_range(start_addr, end);
      //addr_range = AddrRange(start_addr, end);
      mem_footprints.push_back(addr_range);
    }
  }
}; // MemInfo

// MatMul memory information
class MMMemInfo {
  public:
    MMMemInfo(MemInfo_Params mi_sb_params
        , MemInfo_PSUM_Params mi_psum_params
        , tonga_addr if_start_addr
        , tonga_addr w_start_addr
        , length_t ofmap_cnt
        ) : ifmap_mi(MemInfo(mi_sb_params, if_start_addr))
            , wmap_mi(MemInfo(mi_sb_params, w_start_addr))
            , psum_mi(MemInfo(mi_psum_params
                  , mi_psum_params.pbid*MemInfo::PSUM_BANK_SIZE))
  {
    ifmap_mi.compute_footprint();
    //wmap_mi.compute_footprint();

    AddrRange weight_footprint(
        w_start_addr
        , w_start_addr + ofmap_cnt * wmap_mi.elem_size
        );
    psum_mi.compute_footprint();
    ifmap_mi.merge_addr_range(weight_footprint);
    //for(auto a : wmap_mi.mem_footprints)
    //{
      //ifmap_mi.merge_addr_range(a);
    //}
  }
  
  //std::list<AddrRange>& get_ifmap_footprint() {return ifmap_mi.mem_footprints;}
  //std::list<AddrRange>& get_wmap_footprint() {return wmap_mi.mem_footprints;}
  std::list<AddrRange>& get_psum_out_footprint()
  {return psum_mi.mem_footprints;}
  std::list<AddrRange>& get_sb_in_footprint() {return ifmap_mi.mem_footprints;}
  private:
    MemInfo ifmap_mi;
    MemInfo wmap_mi;
    MemInfo psum_mi;
}; // MMMemInfo

/// Memory information for ACT and POOL
/// Note that MatMul memory information does not fall into this category.
/// Thus, we have a separate class for that.
class WaveOpMemInfo {
  public:
    WaveOpMemInfo(MemInfo_Params mip_sb_in, MemInfo_Params mip_sb_out
        , tonga_addr src_sb_addr, tonga_addr dst_sb_addr , tonga_addr bias_sb_addr
        , std::string bias_dtype
        , MemInfo_PSUM_Params mip_psum_in, MemInfo_PSUM_Params mip_psum_out
        , bool compute_bias_mi
        ) : sb_in_mi(MemInfo(mip_sb_in, src_sb_addr))
            , sb_out_mi(MemInfo(mip_sb_out, dst_sb_addr))
            , psum_in_mi(MemInfo(mip_psum_in
                  , mip_psum_in.pbid*MemInfo::PSUM_BANK_SIZE))
            , psum_out_mi(MemInfo(mip_psum_out
                  , mip_psum_out.pbid*MemInfo::PSUM_BANK_SIZE))
            , m_bias_dtype(bias_dtype), m_bias_sb_addr(bias_sb_addr)
  {
    sb_in_mi.compute_footprint();
    sb_out_mi.compute_footprint();
    psum_in_mi.compute_footprint();
    psum_out_mi.compute_footprint();
    if (compute_bias_mi)
      compute_bias_footprint();
  }
  std::list<AddrRange>& get_sb_in_footprint()
  {return sb_in_mi.mem_footprints;}
  std::list<AddrRange>& get_sb_out_footprint()
  {return sb_out_mi.mem_footprints;}
  std::list<AddrRange>& get_psum_in_footprint()
  {return psum_in_mi.mem_footprints;}
  std::list<AddrRange>& get_psum_out_footprint()
  {return psum_out_mi.mem_footprints;}
  private:
    void compute_bias_footprint();
  private:
    MemInfo sb_in_mi;
    MemInfo sb_out_mi;
    MemInfo psum_in_mi;
    MemInfo psum_out_mi;
    //std::vector<AddrRange> m_sb_footprints;
    //std::vector<AddrRange> m_psum_footprints;
    bool m_dst_is_psum;
    bool m_src_is_psum;
    bool m_biase_add_en;
    tonga_addr m_bias_sb_addr;
    bool m_bias_start_at_mid_part;
    std::string m_bias_dtype;
}; // WaveOpMemInfo

class WaveOpGenericMemInfo {
  typedef std::vector<MemInfo_Params> mip_cont_t;
  typedef std::vector<tonga_addr> mi_addr_cont_t;
  public:
  WaveOpGenericMemInfo(mip_cont_t mip_in, mip_cont_t mip_out
      , mi_addr_cont_t src_addr, mi_addr_cont_t dst_addr
      )
  {
    assert(mip_in.size() == src_addr.size());
    assert(mip_out.size() == dst_addr.size());
    auto mi_gen = [](mip_cont_t& mip, mi_addr_cont_t& addr
        , std::vector<MemInfo*>& psum_mi
        , std::vector<MemInfo*>& sb_mi)
    {
      for(int i = 0;i < mip.size();++i)
      {
        MemInfo* mi = new MemInfo(mip[i], addr[i]);
        mi->compute_footprint();
        if (mip[i].psum) psum_mi.push_back(mi);
        else sb_mi.push_back(mi);
      }
    };
    mi_gen(mip_in, src_addr, psum_in_mi, sb_in_mi);
    mi_gen(mip_out, dst_addr, psum_out_mi, sb_out_mi);
    auto merge_mi = [](std::vector<MemInfo*>& mis)
    {
      if (mis.size() > 1)
      {
        for (int i = 1;i < mis.size();++i)
        {
          for(auto a : mis[i]->mem_footprints)
            mis[0]->merge_addr_range(a);
        }
      }
    };
    merge_mi(sb_in_mi);
    merge_mi(sb_out_mi);
    merge_mi(psum_in_mi);
    merge_mi(psum_out_mi);
  }
  ~WaveOpGenericMemInfo()
  {
    for(int i = 0;i < sb_in_mi.size();++i)
      delete sb_in_mi[i];
    for(int i = 0;i < sb_out_mi.size();++i)
      delete sb_out_mi[i];
    for(int i = 0;i < psum_in_mi.size();++i)
      delete psum_in_mi[i];
    for(int i = 0;i < psum_out_mi.size();++i)
      delete psum_out_mi[i];
  }

  std::list<AddrRange>& get_sb_in_footprint()
  {return sb_in_mi[0]->mem_footprints;}
  std::list<AddrRange>& get_sb_out_footprint()
  {
    std::list<AddrRange> a;
    if (sb_out_mi.size())
      return sb_out_mi[0]->mem_footprints;
    else assert(0);
  }
  std::list<AddrRange>& get_psum_in_footprint()
  {return psum_in_mi[0]->mem_footprints;}
  std::list<AddrRange>& get_psum_out_footprint()
  {return psum_out_mi[0]->mem_footprints;}
  bool available_sb_in_mi() {return (sb_in_mi.size() > 0);}
  bool available_sb_out_mi() {return (sb_out_mi.size() > 0);}
  bool available_psum_in_mi() {return (psum_in_mi.size() > 0);}
  bool available_psum_out_mi() {return (psum_out_mi.size() > 0);}
  private:
    std::vector<MemInfo*> sb_in_mi;
    std::vector<MemInfo*> sb_out_mi;
    std::vector<MemInfo*> psum_in_mi;
    std::vector<MemInfo*> psum_out_mi;
}; // WaveOpGenericMemInfo

class SBAtomMemInfo {
  public:
    enum atom_type {SBAtomFile, SBAtomSave};
    SBAtomMemInfo(tonga_addr start_addr, length_t len, bool mid_part
        , atom_type at)
      : m_sb_start_addr(start_addr), m_length(len)
        , m_start_at_mid_part(mid_part), m_atom_type(at)
  {
    AddrRange ar(start_addr, start_addr + len - 1);
    mem_footprints.push_back(ar);
  }
    std::list<AddrRange>& get_footprint() {return mem_footprints;}
    atom_type get_atom_type() {return m_atom_type;}
  private:
    tonga_addr m_sb_start_addr;
    length_t m_length;
    bool m_start_at_mid_part;
    atom_type m_atom_type;
    std::list<AddrRange> mem_footprints;
}; // SBAtomMemInfo

#endif //__WGC_MEMINFO_H__
