//#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "meminfo.h"

//using json = nlohmann::json;
/*
struct mi_params {
  int nx, ny, nz, nw;
  int sx, sy, sz, sw;
  int pnx, pny, pnz, pnw;
  int psx, psy, psz, psw;
  std::string in_dtype;
  std::string out_dtype;
}; // mi_params
*/
struct mi_params {
  MemInfo_Params sb;
  MemInfo_PSUM_Params psum;
}; // mi_params

mi_params parse_mm (json& op)
{
  //std::cout << waveop_type << std::endl;
  mi_params mp;
  mp.sb.enable = true;
  mp.sb.nx = op["src_x_num"];
  mp.sb.sx = op["src_x_step"];
  mp.sb.ny = 0;mp.sb.nz = 0;mp.sb.nw = 0;mp.sb.sy = 0;mp.sb.sz = 0;mp.sb.sw = 0;
  mp.psum.enable = true;
  mp.psum.nx = op["dst_x_num"];
  mp.psum.sx = op["dst_x_step"];
  mp.psum.ny = 0;mp.psum.nz = 0;mp.psum.nw = 0;
  mp.psum.sy = 0;mp.psum.sz = 0;mp.psum.sw = 0;
  mp.psum.pbid = op["dst_psum_bank_id"];
  if (op["src_z_num"] != nullptr) {
    mp.sb.nz = op["src_z_num"];
    mp.sb.sz = op["src_z_step"];
    mp.sb.ny = op["src_y_num"];
    mp.sb.sy = op["src_y_step"];
  } else if (op["src_y_num"] != nullptr) {
    mp.sb.ny = op["src_y_num"];
    mp.sb.sy = op["src_y_step"];
  } 
  if (op["dst_z_num"] != nullptr) {
    mp.psum.nz = op["dst_z_num"];
    mp.psum.sz = op["dst_z_step"];
    mp.psum.ny = op["dst_y_num"];
    mp.psum.sy = op["dst_y_step"];
  } else if (op["dst_y_num"] != nullptr) {
    mp.psum.ny = op["dst_y_num"];
    mp.psum.sy = op["dst_y_step"];
  } 
  mp.sb.dtype = op["in_dtype"];
  mp.psum.dtype = op["out_dtype"];

  return mp;
}

std::vector<mi_params> parse_act(json& op)
{
  mi_params mp_in;
  mi_params mp_out;
  bool src_is_psum;
  if (op["src_is_psum"] == nullptr) src_is_psum = false;
  else src_is_psum = op["src_is_psum"];
  mp_in.sb.enable = !src_is_psum;
  //std::cout << "src_is_psum = " << src_is_psum << " "
    //<< "mp_in.sb.enable = " << mp_in.sb.enable << std::endl;
  mp_in.sb.nx = op["src_x_num"];
  mp_in.sb.sx = op["src_x_step"];
  mp_in.sb.ny = 0;mp_in.sb.nz = 0;mp_in.sb.nw = 0;
  mp_in.sb.sy = 0;mp_in.sb.sz = 0;mp_in.sb.sw = 0;
  mp_in.psum.enable = src_is_psum;
  mp_in.psum.nx = op["src_x_num"];
  mp_in.psum.sx = op["src_x_step"];
  mp_in.psum.ny = 0;mp_in.psum.nz = 0;mp_in.psum.nw = 0;
  mp_in.psum.sy = 0;mp_in.psum.sz = 0;mp_in.psum.sw = 0;
  mp_in.psum.pbid = op["src_psum_bank_id"];
  if (op["src_z_num"] != nullptr) {
    mp_in.sb.nz = op["src_z_num"];
    mp_in.sb.sz = op["src_z_step"];
    mp_in.sb.ny = op["src_y_num"];
    mp_in.sb.sy = op["src_y_step"];
  } else if (op["src_y_num"] != nullptr) {
    mp_in.sb.ny = op["src_y_num"];
    mp_in.sb.sy = op["src_y_step"];
  } 
  if (op["src_z_num"] != nullptr) {
    mp_in.psum.nz = op["src_z_num"];
    mp_in.psum.sz = op["src_z_step"];
    mp_in.psum.ny = op["src_y_num"];
    mp_in.psum.sy = op["src_y_step"];
  } else if (op["src_y_num"] != nullptr) {
    mp_in.psum.ny = op["src_y_num"];
    mp_in.psum.sy = op["src_y_step"];
  } 
  mp_in.sb.dtype = op["in_dtype"];
  mp_in.psum.dtype = op["in_dtype"];

  bool dst_is_psum;
  if (op["dst_is_psum"] == nullptr) dst_is_psum = false;
  else dst_is_psum = op["dst_is_psum"];
  mp_out.sb.enable = !dst_is_psum;
  mp_out.sb.nx = op["dst_x_num"];
  mp_out.sb.sx = op["dst_x_step"];
  mp_out.sb.ny = 0;mp_out.sb.nz = 0;mp_out.sb.nw = 0;
  mp_out.sb.sy = 0;mp_out.sb.sz = 0;mp_out.sb.sw = 0;
  mp_out.psum.enable = dst_is_psum;
  mp_out.psum.nx = op["dst_x_num"];
  mp_out.psum.sx = op["dst_x_step"];
  mp_out.psum.ny = 0;mp_out.psum.nz = 0;mp_out.psum.nw = 0;
  mp_out.psum.sy = 0;mp_out.psum.sz = 0;mp_out.psum.sw = 0;
  if (op["dst_z_num"] != nullptr) {
    mp_out.sb.nz = op["dst_z_num"];
    mp_out.sb.sz = op["dst_z_step"];
    mp_out.sb.ny = op["dst_y_num"];
    mp_out.sb.sy = op["dst_y_step"];
  } else if (op["dst_y_num"] != nullptr) {
    mp_out.sb.ny = op["dst_y_num"];
    mp_out.sb.sy = op["dst_y_step"];
  } 
  if (op["dst_z_num"] != nullptr) {
    mp_out.psum.nz = op["dst_z_num"];
    mp_out.psum.sz = op["dst_z_step"];
    mp_out.psum.ny = op["dst_y_num"];
    mp_out.psum.sy = op["dst_y_step"];
  } else if (op["dst_y_num"] != nullptr) {
    mp_out.psum.ny = op["dst_y_num"];
    mp_out.psum.sy = op["dst_y_step"];
  } 
  mp_out.sb.dtype = op["in_dtype"];
  mp_out.psum.dtype = op["in_dtype"];
  if (dst_is_psum) mp_out.psum.pbid = op["dst_psum_bank_id"];

  std::vector<mi_params> res;
  res.push_back(mp_in);
  res.push_back(mp_out);

  return res;
}

// print intervals of sb read by matmul
void print_footprints_waveops (json& j)
{
  for(auto op : j["waveops"])
  {
    std::string waveop_type = op["waveop_type"];
    if (!waveop_type.compare("MatMul"))
    {
      mi_params mp = parse_mm(op);
      MMMemInfo m(
          mp.sb
          , mp.psum
          , op["src_sb_address"]
          , op["weights_sb_address"]
          , op["num_column_partitions"]
	  );
      std::list<AddrRange> ar = m.get_sb_in_footprint();
      std::cout << "MM sb_in footprint: ";
      AddrRange::print_text_ars<std::list<AddrRange> >(ar);

      //ar = m.get_wmap_footprint();
      //std::cout << "weight interval: ";
      //AddrRange::print_text_ars<std::list<AddrRange> >(ar);

      ar = m.get_psum_out_footprint();
      std::cout << "MM psum_out footprint: ";
      AddrRange::print_text_ars<std::list<AddrRange> >(ar);
    }
    else if (!waveop_type.compare("Activation") ||
        !waveop_type.compare("Pool"))
    {
      std::string op_t;
      if (!waveop_type.compare("Activation")) op_t = "ACT";
      if (!waveop_type.compare("Pool")) op_t = "POOL";
      std::vector<mi_params> mp = parse_act(op);
      std::string bias_dtype;
      tonga_addr bias_sb_addr;
      tonga_addr src_sb_addr;
      tonga_addr dst_sb_addr;
      bool bias_add_en = false;
      if (op["bias_sb_address"] != nullptr)
        bias_sb_addr = op["bias_sb_address"];
      if (op["src_sb_address"] != nullptr)
        src_sb_addr = op["src_sb_address"];
      if (op["sb_address"] != nullptr) // For SBAtomSave
        src_sb_addr = op["src_sb_address"];
      if (op["dst_sb_address"] != nullptr) // For Act and Pool
        dst_sb_addr = op["dst_sb_address"];
      if (op["sb_address"] != nullptr) // For SBAtomLoad
        dst_sb_addr = op["dst_sb_address"];
      if (op["bias_add_en"] != nullptr)
        bias_add_en = op["bias_add_en"];
      if (op["bias_dtype"] != nullptr) bias_dtype = op["bias_dtype"];
      WaveOpMemInfo m(mp[0].sb, mp[1].sb, src_sb_addr
          , dst_sb_addr, bias_sb_addr, bias_dtype
          , mp[0].psum, mp[1].psum, bias_add_en
          );
      std::list<AddrRange> ar;
      bool src_is_psum;
      if (op["src_is_psum"] == nullptr) src_is_psum = false;
      else src_is_psum = op["src_is_psum"];
      bool dst_is_psum;
      if (op["dst_is_psum"] == nullptr) dst_is_psum = false;
      else dst_is_psum = op["dst_is_psum"];
      if (!src_is_psum)
      {
        ar = m.get_sb_in_footprint();
        std::cout << op_t<< " sb_in footprint: ";
      } else {
        ar = m.get_psum_in_footprint();
        std::cout << op_t << " psum_in footprint: ";
      }
      AddrRange::print_text_ars<std::list<AddrRange> >(ar);
      if (bias_add_en) {
        ar = m.get_sb_in_footprint();
        std::cout << op_t << " sb_in footprint: ";
        AddrRange::print_text_ars<std::list<AddrRange> >(ar);
      }
      if (!dst_is_psum)
      {
        ar = m.get_sb_out_footprint();
        std::cout << op_t << " sb_out footprint: ";
      } else
      {
        ar = m.get_psum_out_footprint();
        std::cout << op_t << " psum_out footprint: ";
      }
      AddrRange::print_text_ars<std::list<AddrRange> >(ar);
    }
    else if (!waveop_type.compare("SBAtomLoad") ||
        !waveop_type.compare("SBAtomSave"))
    {
      SBAtomMemInfo::atom_type at =
        (!waveop_type.compare("SBAtomLoad")) ? SBAtomMemInfo::SBAtomLoad :
        SBAtomMemInfo::SBAtomSave;
      //tonga_addr start_addr = op["sb_address"];
      //length_t len = op["length"];
      //bool mid = op["start_at_mid_part"];
      //SBAtomMemInfo m(start_addr, len, mid, at);
      SBAtomMemInfo m(op, at);
      std::list<AddrRange> ar = m.get_footprint();
      if (at == SBAtomMemInfo::SBAtomSave) {
        std::cout << "SBAtomSave sb_in footprint: ";
      } else {
        std::cout << "SBAtomLoad sb_out footprint: ";
      }
      AddrRange::print_text_ars<std::list<AddrRange> >(ar);
    }
  }
}

int main(int argc, char* argv[])
{
  std::ifstream in_wave(argv[1]);
  json j;
  in_wave >> j;

  print_footprints_waveops(j);
}
