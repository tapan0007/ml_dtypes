#include <boost/graph/graphviz.hpp>
#include <iostream>
#include "wave_graph.h"
#include <unordered_set>
#include <unordered_map>

WaveOp::WaveOpType WaveOp::ExtractWaveOpTypeEngine(
    std::string& wot)
{
  WaveOpType t;
  if (!wot.compare("SBAtomLoad")) t = SBAtomLoad;
  else if (!wot.compare("SBAtomSave")) t = SBAtomSave;
  else if (!wot.compare("MatMul")) t = MatMul;
  else if (!wot.compare("Pool")) t = Pool;
  else if (!wot.compare("Reciprocal")) t = Pool;
  else if (!wot.compare("RegLoad")) t = Pool;
  else if (!wot.compare("RegStore")) t = Pool;
  else if (!wot.compare("TpbCopy")) t = Pool;
  else if (!wot.compare("ResAdd")) t = ResAdd;
  else if (!wot.compare("Multiply")) t = ResAdd;
  else if (!wot.compare("Sub")) t = ResAdd;
  else if (!wot.compare("Add")) t = ResAdd;
  else if (!wot.compare("TensorTensor")) t = ResAdd;
  else if (!wot.compare("TensorScalar")) t = Pool;
  else if (!wot.compare("TensorScalarPtr")) t = Pool;
  else if (!wot.compare("ScaleAdd")) t = Pool;
  else if (!wot.compare("Maximum")) t = Pool;
  else if (!wot.compare("Minimum")) t = Pool;
  else if (!wot.compare("Activation")) t = Activation;
  else if (!wot.compare("ClipByValue")) t = Pool;
  else if (!wot.compare("Nop")) t = Nop;
  else assert(0);
  return t;
}

inline bool WaveOp::IsDRAMOp()
{
  return ((waveop_type == SBAtomLoad) || (waveop_type == SBAtomSave));
}

MemInfo_Params MMOp::extract_sb_params(json& op)
{
  MemInfo_Params mp;
  mp.enable = true;
  mp.nx = op["src_x_num"];
  mp.sx = op["src_x_step"];
  mp.ny = 1;mp.nz = 1;mp.nw = 1;mp.sy = 0;mp.sz = 0;mp.sw = 0;
  if (op["src_z_num"] != nullptr) {
    mp.nz = op["src_z_num"];
    mp.sz = op["src_z_step"];
    mp.ny = op["src_y_num"];
    mp.sy = op["src_y_step"];
  } else if (op["src_y_num"] != nullptr) {
    mp.ny = op["src_y_num"];
    mp.sy = op["src_y_step"];
  } 
  mp.dtype = op["in_dtype"];

  return mp;
}

MemInfo_PSUM_Params MMOp::extract_psum_params(json& op)
{
  MemInfo_PSUM_Params mp;
  mp.enable = true;
  mp.nx = op["dst_x_num"];
  mp.sx = op["dst_x_step"];
  mp.ny = 1;mp.nz = 1;mp.nw = 1;
  mp.sy = 0;mp.sz = 0;mp.sw = 0;
  mp.pbid = op["dst_psum_bank_id"];
  if (op["dst_w_num"] != nullptr) {
    mp.nw = op["dst_w_num"];
    mp.sw = op["dst_w_step"];
    mp.nz = op["dst_z_num"];
    mp.sz = op["dst_z_step"];
    mp.ny = op["dst_y_num"];
    mp.sy = op["dst_y_step"];
  } else if (op["dst_z_num"] != nullptr) {
    mp.nz = op["dst_z_num"];
    mp.sz = op["dst_z_step"];
    mp.ny = op["dst_y_num"];
    mp.sy = op["dst_y_step"];
  } else if (op["dst_y_num"] != nullptr) {
    mp.ny = op["dst_y_num"];
    mp.sy = op["dst_y_step"];
  } 
  mp.dtype = op["out_dtype"];

  return mp;
}

MemInfo_Params PoolActOp::extract_sb_in_params(json& op)
{
  MemInfo_Params mp_in;
  const std::string wave_op_type = op["waveop_type"].get<std::string>();
  if (wave_op_type == "RegStore") {
    // RegStore moves data from internal Reg to SBUF, so no input from SBUF
    mp_in.enable = false;
    mp_in.SetZeroSize();
    mp_in.dtype = op["out_dtype"];
    return mp_in;
  } else if (wave_op_type == "TpbCopy") {
    const uint32_t len = op["size_in_bytes"];
    mp_in.SetDmaEquiv(len);
    return mp_in;
  }
  bool src_is_psum;
  if (op["src_is_psum"] == nullptr) src_is_psum = false;
  else src_is_psum = op["src_is_psum"];
  mp_in.enable = !src_is_psum;
  mp_in.nx = op["src_x_num"];
  mp_in.sx = op["src_x_step"];
  mp_in.ny = 1;mp_in.nz = 1;mp_in.nw = 1;
  mp_in.sy = 0;mp_in.sz = 0;mp_in.sw = 0;
  if (op["src_w_num"] != nullptr) {
    mp_in.nw = op["src_w_num"];
    mp_in.sw = op["src_w_step"];
    mp_in.nz = op["src_z_num"];
    mp_in.sz = op["src_z_step"];
    mp_in.ny = op["src_y_num"];
    mp_in.sy = op["src_y_step"];
  } else if (op["src_z_num"] != nullptr) {
    mp_in.nz = op["src_z_num"];
    mp_in.sz = op["src_z_step"];
    mp_in.ny = op["src_y_num"];
    mp_in.sy = op["src_y_step"];
  } else if (op["src_y_num"] != nullptr) {
    mp_in.ny = op["src_y_num"];
    mp_in.sy = op["src_y_step"];
  } 
  mp_in.dtype = op["in_dtype"];

  return mp_in;
}

MemInfo_Params PoolActOp::extract_sb_out_params(json& op)
{
  MemInfo_Params mp_out;
  const std::string wave_op_type = op["waveop_type"].get<std::string>();
  if (wave_op_type == "RegLoad") {
    // RegLoad  moves data from SBUF to internal Reg, so no output to SBUF
    mp_out.enable = false;
    mp_out.SetZeroSize();
    mp_out.dtype = op["in_dtype"];
    return mp_out;
  } else if (wave_op_type == "TpbCopy") {
    const uint32_t len = op["size_in_bytes"];
    mp_out.SetDmaEquiv(len);
    return mp_out;
  }
  bool dst_is_psum;
  if (op["dst_is_psum"] == nullptr) dst_is_psum = false;
  else dst_is_psum = op["dst_is_psum"];
  mp_out.enable = !dst_is_psum;
  mp_out.nx = op["dst_x_num"];
  mp_out.sx = op["dst_x_step"];
  mp_out.ny = 1;mp_out.nz = 1;mp_out.nw = 1;
  mp_out.sy = 0;mp_out.sz = 0;mp_out.sw = 0;
  if (op["dst_w_num"] != nullptr) {
    mp_out.nw = op["dst_w_num"];
    mp_out.sw = op["dst_w_step"];
    mp_out.nz = op["dst_z_num"];
    mp_out.sz = op["dst_z_step"];
    mp_out.ny = op["dst_y_num"];
    mp_out.sy = op["dst_y_step"];
  } else if (op["dst_z_num"] != nullptr) {
    mp_out.nz = op["dst_z_num"];
    mp_out.sz = op["dst_z_step"];
    mp_out.ny = op["dst_y_num"];
    mp_out.sy = op["dst_y_step"];
  } else if (op["dst_y_num"] != nullptr) {
    mp_out.ny = op["dst_y_num"];
    mp_out.sy = op["dst_y_step"];
  } 
  mp_out.dtype = op["out_dtype"];

  return mp_out;
}

MemInfo_PSUM_Params PoolActOp::extract_psum_in_params(json& op)
{
  MemInfo_PSUM_Params mp_in;
  const std::string wave_op_type = op["waveop_type"].get<std::string>();
  if (wave_op_type == "RegLoad") {
    // RegLoad/Store  move data from SBUF/internal Reg to internal Reg/SBUF, so no interaction with PSUM
    mp_in.enable = false;
    mp_in.SetZeroSize();
    mp_in.dtype = op["in_dtype"];
    return mp_in;
  } else if (op["waveop_type"] == "RegStore") {
    mp_in.enable = false;
    mp_in.SetZeroSize();
    mp_in.dtype = op["out_dtype"];
    return mp_in;
  } else if (wave_op_type == "TpbCopy") {
    // TpbCopy does not access PSUM, make empty pattern (all Ns == 0)
    mp_in.enable = false;
    mp_in.SetZeroSize();
    mp_in.dtype = "uint8";
    return mp_in;
  }

  bool src_is_psum;
  if (op["src_is_psum"] == nullptr) src_is_psum = false;
  else src_is_psum = op["src_is_psum"];
  mp_in.enable = src_is_psum;
  mp_in.nx = op["src_x_num"];
  mp_in.sx = op["src_x_step"];
  mp_in.ny = 1;mp_in.nz = 1;mp_in.nw = 1;
  mp_in.sy = 0;mp_in.sz = 0;mp_in.sw = 0;
  if (op["src_psum_bank_id"] != nullptr)
  {
    mp_in.pbid = op["src_psum_bank_id"];
  } else 
  {
    mp_in.pbid = 0;
  }
  if (op["src_w_num"] != nullptr) {
    mp_in.nw = op["src_w_num"];
    mp_in.sw = op["src_w_step"];
    mp_in.nz = op["src_z_num"];
    mp_in.sz = op["src_z_step"];
    mp_in.ny = op["src_y_num"];
    mp_in.sy = op["src_y_step"];
  } else if (op["src_z_num"] != nullptr) {
    mp_in.nz = op["src_z_num"];
    mp_in.sz = op["src_z_step"];
    mp_in.ny = op["src_y_num"];
    mp_in.sy = op["src_y_step"];
  } else if (op["src_y_num"] != nullptr) {
    mp_in.ny = op["src_y_num"];
    mp_in.sy = op["src_y_step"];
  } 
  mp_in.dtype = op["in_dtype"];

  return mp_in;
}

MemInfo_PSUM_Params PoolActOp::extract_psum_out_params(json& op)
{
  MemInfo_PSUM_Params mp_out;
  const std::string wave_op_type = op["waveop_type"].get<std::string>();
  if (wave_op_type == "RegLoad") {
    // RegLoad/Store  move data from SBUF/internal Reg to internal Reg/SBUF, so no interaction with PSUM
    mp_out.enable = false;
    mp_out.SetZeroSize();
    mp_out.dtype = op["in_dtype"];
    return mp_out;
  } else if (wave_op_type == "RegStore") {
    mp_out.enable = false;
    mp_out.SetZeroSize();
    mp_out.dtype = op["out_dtype"];
    return mp_out;
  } else if (wave_op_type == "TpbCopy") {
    // TpbCopy does not access PSUM, make empty pattern (all Ns == 0)
    mp_out.enable = false;
    mp_out.SetZeroSize();
    mp_out.dtype = "uint8";
    return mp_out;
  }
  bool dst_is_psum;
  if (op["dst_is_psum"] == nullptr) dst_is_psum = false;
  else dst_is_psum = op["dst_is_psum"];
  mp_out.enable = dst_is_psum;
  mp_out.nx = op["dst_x_num"];
  mp_out.sx = op["dst_x_step"];
  mp_out.ny = 1;mp_out.nz = 1;mp_out.nw = 1;
  mp_out.sy = 0;mp_out.sz = 0;mp_out.sw = 0;
  if (op["dst_w_num"] != nullptr) {
    mp_out.nw = op["dst_w_num"];
    mp_out.sw = op["dst_w_step"];
    mp_out.nz = op["dst_z_num"];
    mp_out.sz = op["dst_z_step"];
    mp_out.ny = op["dst_y_num"];
    mp_out.sy = op["dst_y_step"];
  } else if (op["dst_z_num"] != nullptr) {
    mp_out.nz = op["dst_z_num"];
    mp_out.sz = op["dst_z_step"];
    mp_out.ny = op["dst_y_num"];
    mp_out.sy = op["dst_y_step"];
  } else if (op["dst_y_num"] != nullptr) {
    mp_out.ny = op["dst_y_num"];
    mp_out.sy = op["dst_y_step"];
  } 
  mp_out.dtype = op["out_dtype"];
  if (dst_is_psum) mp_out.pbid = op["dst_psum_bank_id"];

  return mp_out;
}

tonga_addr PoolActOp::extract_bias_sb_addr(json& op)
{
  tonga_addr a = 0;
  if (op["bias_sb_address"] != nullptr)
    a = op["bias_sb_address"].get<tonga_addr>();
  return a;
}

tonga_addr PoolActOp::extract_src_sb_addr(json& op)
{
  tonga_addr a = 0;
  const std::string wave_op_type = op["waveop_type"].get<std::string>();
  if (wave_op_type == "RegStore") {
    // RegStore moves data from internal Reg to SBUF, so no input from SBUF
    return a;
  } else if (wave_op_type == "TpbCopy") {
    a = op["src_sb_address"].get<tonga_addr>();
    return a;
  }
  if (op["src_sb_address"] != nullptr)
    a = op["src_sb_address"].get<tonga_addr>();
  return a;
}

tonga_addr PoolActOp::extract_dst_sb_addr(json& op)
{
  tonga_addr a = 0;
  const std::string wave_op_type = op["waveop_type"].get<std::string>();
  if (wave_op_type == "RegLoad") {
    // RegLoad  moves data from SBUF to internal Reg, so no output to SBUF
    return a;
  } else if (wave_op_type == "TpbCopy") {
    a = op["dst_sb_address"].get<tonga_addr>();
    return a;
  }
  if (op["dst_sb_address"] != nullptr)
    a = op["dst_sb_address"].get<tonga_addr>();
  return a;
}

bool PoolActOp::extract_bias_add_en(json& op)
{
  bool a = false;
  if (op["bias_add_en"] != nullptr)
    a = op["bias_add_en"].get<bool>();
  return a;
}

std::string PoolActOp::extract_bias_dtype(json& op)
{
  std::string a = "float16";
  if (op["bias_dtype"] != nullptr)
    a = op["bias_dtype"].get<std::string>();
  return a;
}

SBAtomMemInfo::atom_type SBAtomOp::extract_atom_type(json& op)
{
  std::string waveop_type = op["waveop_type"].get<std::string>();
  SBAtomMemInfo::atom_type at =
    (!waveop_type.compare("SBAtomLoad")) ? SBAtomMemInfo::SBAtomLoad :
    SBAtomMemInfo::SBAtomSave;
  return at;
}

std::vector<MemInfo_Params> ResAddOp::extract_in_params(json& op)
{
  MemInfo_Params mip_ina;
  MemInfo_Params mip_inb;

  mip_ina.psum = op["src_a_is_psum"].get<bool>();
  mip_ina.nx =
    (op["src_a_x_num"] != nullptr) ? op["src_a_x_num"].get<num_t>() : 0;
  mip_ina.ny =
    (op["src_a_y_num"] != nullptr) ? op["src_a_y_num"].get<num_t>() : 0;
  mip_ina.nz =
    (op["src_a_z_num"] != nullptr) ? op["src_a_z_num"].get<num_t>() : 0;
  mip_ina.nw =
    (op["src_a_w_num"] != nullptr) ? op["src_a_w_num"].get<num_t>() : 0;
  mip_ina.sx =
    (op["src_a_x_step"] != nullptr) ? op["src_a_x_step"].get<step_t>() : 0;
  mip_ina.sy =
    (op["src_a_y_step"] != nullptr) ? op["src_a_y_step"].get<step_t>() : 0;
  mip_ina.sz =
    (op["src_a_z_step"] != nullptr) ? op["src_a_z_step"].get<step_t>() : 0;
  mip_ina.sw =
    (op["src_a_w_step"] != nullptr) ? op["src_a_w_step"].get<step_t>() : 0;
  mip_ina.dtype =
    (op["in_a_dtype"] != nullptr) ?
    op["in_a_dtype"].get<std::string>() : "float16";

  mip_inb.psum = op["src_b_is_psum"].get<bool>();
  mip_inb.nx =
    (op["src_b_x_num"] != nullptr) ? op["src_b_x_num"].get<num_t>() : 0;
  mip_inb.ny =
    (op["src_b_y_num"] != nullptr) ? op["src_b_y_num"].get<num_t>() : 0;
  mip_inb.nz =
    (op["src_b_z_num"] != nullptr) ? op["src_b_z_num"].get<num_t>() : 0;
  mip_inb.nw =
    (op["src_b_w_num"] != nullptr) ? op["src_b_w_num"].get<num_t>() : 0;
  mip_inb.sx =
    (op["src_b_x_step"] != nullptr) ? op["src_b_x_step"].get<step_t>() : 0;
  mip_inb.sy =
    (op["src_b_y_step"] != nullptr) ? op["src_b_y_step"].get<step_t>() : 0;
  mip_inb.sz =
    (op["src_b_z_step"] != nullptr) ? op["src_b_z_step"].get<step_t>() : 0;
  mip_inb.sw =
    (op["src_b_w_step"] != nullptr) ? op["src_b_w_step"].get<step_t>() : 0;
  mip_inb.dtype =
    (op["in_b_dtype"] != nullptr) ?
    op["in_b_dtype"].get<std::string>() : "float16";
  std::vector<MemInfo_Params> mip_ins;
  mip_ins.resize(2);
  mip_ins[0] = mip_ina;
  mip_ins[1] = mip_inb;

  return mip_ins;
}

std::vector<MemInfo_Params> ResAddOp::extract_out_params(json& op)
{
  MemInfo_Params mip_out;

  mip_out.psum = op["dst_is_psum"].get<bool>();
  mip_out.nx =
    (op["dst_x_num"] != nullptr) ? op["dst_x_num"].get<num_t>() : 0;
  mip_out.ny =
    (op["dst_y_num"] != nullptr) ? op["dst_y_num"].get<num_t>() : 0;
  mip_out.nz =
    (op["dst_z_num"] != nullptr) ? op["dst_z_num"].get<num_t>() : 0;
  mip_out.nw =
    (op["dst_w_num"] != nullptr) ? op["dst_w_num"].get<num_t>() : 0;
  mip_out.sx =
    (op["dst_x_step"] != nullptr) ? op["dst_x_step"].get<step_t>() : 0;
  mip_out.sy =
    (op["dst_y_step"] != nullptr) ? op["dst_y_step"].get<step_t>() : 0;
  mip_out.sz =
    (op["dst_z_step"] != nullptr) ? op["dst_z_step"].get<step_t>() : 0;
  mip_out.sw =
    (op["dst_w_step"] != nullptr) ? op["dst_w_step"].get<step_t>() : 0;
  mip_out.dtype =
    (op["out_dtype"] != nullptr) ?
    op["out_dtype"].get<std::string>() : "float16";
  std::vector<MemInfo_Params> mip_outs;
  mip_outs.resize(1);
  mip_outs[0] = mip_out;

  return mip_outs;
}

std::vector<tonga_addr> ResAddOp::extract_in_addrs(json& op)
{
  tonga_addr a1;
  if (op["src_a_is_psum"].get<bool>())
    a1 = op["src_a_psum_bank_id"].get<tonga_addr>() * MemInfo::PSUM_BANK_SIZE;
  else
    a1 = op["src_a_sb_address"].get<tonga_addr>();
  tonga_addr a2;
  if (op["src_b_is_psum"].get<bool>())
    a2 = op["src_b_psum_bank_id"].get<tonga_addr>() * MemInfo::PSUM_BANK_SIZE;
  else
    a2 = op["src_b_sb_address"].get<tonga_addr>();
  std::vector<tonga_addr> addrs;
  addrs.resize(2);
  addrs[0] = a1;
  addrs[1] = a2;

  return addrs;
}

std::vector<tonga_addr> ResAddOp::extract_out_addrs(json& op)
{
  tonga_addr a;
  if (op["dst_is_psum"].get<bool>())
    a = op["dst_psum_bank_id"].get<tonga_addr>() * MemInfo::PSUM_BANK_SIZE;
  else a = op["dst_sb_address"].get<tonga_addr>();

  std::vector<tonga_addr> as;
  as.resize(1);
  as[0] = a;

  return as;
}

WaveGraphChecker::WaveGraphChecker(json& j, CommandLineOptions cli)
  : mJson(j), mCLI(cli)
{
  uint32_t num_neighs = 0;
  uint32_t num_ops = 0;
  int64_t eid = 0;
  uint8_t event_wait_mode = 0;
  uint8_t event_set_mode = 0;
  for(auto op : j["waveops"])
  {
    std::string n = op["waveop_name"];
    std::string wave_op_type_json = op["waveop_type"];
    WaveOp* wo = ConstructWaveOp(op);
    assert(wo != nullptr);
    vertex_t v = boost::add_vertex(wo, wg);
    if (wo->get_waveop_type() == WaveOp::MatMul) mMMops.push_back(v);
    if (wo->get_waveop_type() == WaveOp::Activation) mACTops.push_back(v);
    if (wo->get_waveop_type() == WaveOp::Pool) mPOOLops.push_back(v);
    if (wo->get_waveop_type() == WaveOp::SBAtomLoad) mLDops.push_back(v);
    if (wo->get_waveop_type() == WaveOp::SBAtomSave) mSTops.push_back(v);
    if (wo->get_waveop_type() == WaveOp::ResAdd) mResAddops.push_back(v);
    mName2WaveOp.insert(std::pair<std::string, WaveOp*>(n, wo));
    mWaveOp2V.insert(std::pair<WaveOp*, vertex_t>(wo, v));
    if (op["previous_waveops"] != nullptr) {
      int prev_idx = 0;
      for(std::string p : op["previous_waveops"])
      {
        if (mName2WaveOp.find(p) != mName2WaveOp.end())
        {
          if (op["previous_event_ids"] != nullptr)
          {
            eid = op["previous_event_ids"][prev_idx].get<int64_t>();
          }
          if (op["previous_event_wait_modes"] != nullptr)
          {
            event_wait_mode =
              op["previous_event_wait_modes"][prev_idx].get<uint8_t>();
          }
          if (op["previous_event_set_modes"] != nullptr)
          {
            event_set_mode =
              op["previous_event_set_modes"][prev_idx].get<uint8_t>();
          }
          EventEdge ev = {eid, event_wait_mode, event_set_mode};
          vertex_t u = mWaveOp2V[mName2WaveOp[p]];
          edge_t e = boost::add_edge(u, v, ev, wg).first;
        }
        prev_idx++;
      }
      num_neighs += op["previous_waveops"].size();
    }
  }
  MakeImplicitEdgesExplicit();
  if (mCLI.event_conflict_check)
  {
    mEventChecker = new EventChecker(wg);
  }
  InfoPrefix();
  messages << "Average degree = "
    << ((float)num_neighs / (float)j["waveops"].size())
    << std::endl;
}

WaveOp* WaveGraphChecker::ConstructWaveOp(json& op)
{
  std::string wave_op_type = op["waveop_type"].get<std::string>();
  WaveOp* wo = nullptr;
  if (!wave_op_type.compare("MatMul")) wo = new MMOp(op);
  else if (!wave_op_type.compare("Activation") ||
      !wave_op_type.compare("Pool") ||
      !wave_op_type.compare("Reciprocal") ||
      !wave_op_type.compare("RegLoad") ||
      !wave_op_type.compare("RegStore") ||
      !wave_op_type.compare("TpbCopy") ||
      !wave_op_type.compare("ClipByValue") ||
      !wave_op_type.compare("ScaleAdd")) wo = new PoolActOp(op);
  else if (!wave_op_type.compare("SBAtomLoad") ||
      !wave_op_type.compare("SBAtomSave")) wo = new SBAtomOp(op);
  else if (!wave_op_type.compare("ResAdd")) wo = new ResAddOp(op);
  else if (!wave_op_type.compare("Sub")) wo = new ResAddOp(op);
  else if (!wave_op_type.compare("Add")) wo = new ResAddOp(op);
  else if (!wave_op_type.compare("TensorTensor")) wo = new ResAddOp(op);
  else if (!wave_op_type.compare("TensorScalar")) wo = new PoolActOp(op);
  else if (!wave_op_type.compare("TensorScalarPtr")) wo = new PoolActOp(op);
  else if (!wave_op_type.compare("Multiply") ||
      !wave_op_type.compare("Maximum") ||
      !wave_op_type.compare("Minimum"))
  {
    if (op["is_scalar_op"] != nullptr && op["is_scalar_op"].get<bool>()==false)
    {
      wo = new ResAddOp(op);
    } else {
      wo = new PoolActOp(op);
    }
  } else if (!wave_op_type.compare("Nop")) wo = new NopOp(op);
  else 
  {
    std::cerr << "ASSERT:: " << wave_op_type
      << " is not currently supported" << std::endl;
    assert(0);
  }

  return wo;
}

WaveGraphChecker::~WaveGraphChecker()
{
  for(auto v : mName2WaveOp)
  {
    delete v.second;
  }
  if (mCLI.event_conflict_check)
  {
    delete mEventChecker;
  }
}

void WaveGraphChecker::write_graph_viz()
{
  boost::write_graphviz(std::cout, wg, [&] (std::ostream& out, vertex_t v) {
      out << "[label=\"" << wg[v]->get_name() << "\"]";
      });
  std::cout << std::flush;
}

bool WaveGraphChecker::structure_check()
{
  bool err = false;
  typedef boost::graph_traits<graph_t>::vertex_iterator v_itr;
  std::pair<v_itr, v_itr> vp = boost::vertices(wg);
  InfoPrefix();
  messages << "Starting Structure Inspection" << std::endl;
  for(;vp.first != vp.second;++vp.first) {
    err |= CheckImmNeighbors_NonDRAMOp(*vp.first);
    err |= CheckDuplicatedEdges(*vp.first);
  }
  InfoPrefix();
  messages << "Finished Structure Inspection" << std::endl;
  return err;
}

bool WaveGraphChecker::CheckDuplicatedEdges(vertex_t v)
{
  bool err = false;
  struct comp {
    bool operator() (const std::string a, const std::string b) const
    {
      return (a.compare(b) < 0);
    }
  };
  typename boost::graph_traits<graph_t>::in_edge_iterator ei, ei_end;
  std::set<std::string, comp> pred_n;
  for (boost::tie(ei, ei_end) = boost::in_edges(v, wg); ei != ei_end; ++ei) {
    auto source = boost::source (*ei, wg);
    std::string n = wg[source]->get_name();
    if (pred_n.find(n) == pred_n.end())
      pred_n.insert(n);
    else
    {
      err = true;
      ErrorPrefix();
      messages << wg[v]->get_name()
        << " has duplicated incoming edges" << std::endl;
      break;
    }
  }
  return err;
}

inline bool WaveGraphChecker::InputOperandCheck(vertex_t v)
{
  bool err = false;
  std::pair<ie_itr, ie_itr> iep;
  iep = boost::in_edges(v, wg);
  if (iep.first == iep.second && wg[v]->get_waveop_type() != WaveOp::Nop) {
    err = true;
    ErrorPrefix();
    messages << wg[v]->get_name()
      << " does not have an input operand" << std::endl;
  }
  if (wg[v]->IsMatMul()) {
    for(;iep.first != iep.second;++iep.first)
    {
      vertex_t s = boost::source(*iep.first, wg);
      if (wg[s]->get_waveop_type() == WaveOp::SBAtomSave)
      {
        WarningPrefix();
        messages << "Input operand of " << wg[v]->get_name()
          << " is SBAtomSave" << std::endl;
        messages << "\t" << wg[s]->get_name() << std::endl;
      }
    }
  }
  return err;
}

inline bool WaveGraphChecker::OutputOperandCheck(vertex_t v)
{
  bool err = false;
  std::pair<oe_itr, oe_itr> oep;
  oep = boost::out_edges(v, wg);
  if (oep.first == oep.second && wg[v]->get_waveop_type() != WaveOp::Nop)
  {
    err = true;
    ErrorPrefix();
    messages << wg[v]->get_name()
      << " does not have an output operand" << std::endl;
  }
  return err;
}

bool WaveGraphChecker::CheckImmNeighbors_NonDRAMOp(vertex_t v)
{
  bool err = false;
  if (!wg[v]->IsDRAMOp())
  {
    err |= InputOperandCheck(v);
    err |= OutputOperandCheck(v);
  }
  return err;
}

/// DataRaceChecker
/// Checks if there are issues of potential data race between ops in u and
/// those in v
bool WaveGraphChecker::DataRaceChecker (
    std::list<vertex_t>& u
    , map_v2reach_v* vtx_u_to_v
    , std::list<vertex_t>& v
    , map_v2reach_v* vtx_v_to_u
    )
{
  bool race = false;
  for(auto u_ : u)
  {
    set_v* set_u_to_v = (*vtx_u_to_v)[u_];
    for(auto v_ : v)
    {
      set_v* set_v_to_u = (*vtx_v_to_u)[v_];
      if (set_u_to_v->find(v_) != set_u_to_v->end()
          && set_v_to_u->find(u_) != set_v_to_u->end())
      {
        race |= DataRace(wg[u_], wg[v_], true);
      }
    }
  }

  return race;
}

bool WaveGraphChecker::DataRace(WaveOp* u, WaveOp* v, bool print)
{
  bool err = false;
  if (u->get_sb_in_footprint_size() && v->get_sb_out_footprint_size())
  {
    if (AddrSOverlap(u->get_sb_in_footprint(),v->get_sb_out_footprint()))
    {
      err = true;
      if (print)
      {
        DataRacePrint(u, v, RAW_SB);
        messages << "\tSB Read range : "
        << AddrRange::print_text_ars<std::list<AddrRange> >(
            u->get_sb_in_footprint());
        messages << "\tSB Write range : "
        << AddrRange::print_text_ars<std::list<AddrRange> >(
            v->get_sb_out_footprint());
      }
    }
  }
  if (u->get_sb_out_footprint_size() && v->get_sb_in_footprint_size())
  {
    if (AddrSOverlap(u->get_sb_out_footprint(),v->get_sb_in_footprint()))
    {
      err = true;
      if (print)
      {
        DataRacePrint(v, u, RAW_SB);
        messages << "\tSB Write range : "
        << AddrRange::print_text_ars<std::list<AddrRange> >(
            u->get_sb_out_footprint());
        messages << "\tSB Read range : "
        << AddrRange::print_text_ars<std::list<AddrRange> >(
            v->get_sb_in_footprint());
      }
    }
  }
  if (u->get_sb_out_footprint_size() && v->get_sb_out_footprint_size())
  {
    if (AddrSOverlap(u->get_sb_out_footprint(),v->get_sb_out_footprint()))
    {
      err = true;
      if (print)
      {
        DataRacePrint(u, v, WAW_SB);
        messages << "\tSB Write range : "
        << AddrRange::print_text_ars<std::list<AddrRange> >(
            u->get_sb_out_footprint());
        messages << "\tSB Write range : "
        << AddrRange::print_text_ars<std::list<AddrRange> >(
            v->get_sb_out_footprint());
      }
    }
  }
  if (u->get_psum_in_footprint_size() && v->get_psum_out_footprint_size())
  {
    if (AddrSOverlap(u->get_psum_in_footprint(),v->get_psum_out_footprint()))
    {
      err = true;
      if (print)
      {
        DataRacePrint(u, v, RAW_PSUM);
        messages << "\tPSUM Read range : "
        << AddrRange::print_text_ars<std::list<AddrRange> >(
            u->get_psum_in_footprint());
        messages << "\tPSUM Write range : "
        << AddrRange::print_text_ars<std::list<AddrRange> >(
            v->get_psum_out_footprint());
      }
    }
  }
  if (u->get_psum_out_footprint_size() && v->get_psum_in_footprint_size())
  {
    if (AddrSOverlap(u->get_psum_out_footprint(),v->get_psum_in_footprint()))
    {
      err = true;
      if (print)
      {
        DataRacePrint(v, u, RAW_PSUM);
        messages << "\tPSUM Write range : "
        << AddrRange::print_text_ars<std::list<AddrRange> >(
            u->get_psum_out_footprint());
        messages << "\tPSUM Read range : "
        << AddrRange::print_text_ars<std::list<AddrRange> >(
            v->get_psum_in_footprint());
      }
    }
  }
  if (u->get_psum_out_footprint_size() && v->get_psum_out_footprint_size())
  {
    if (AddrSOverlap(u->get_psum_out_footprint(),v->get_psum_out_footprint()))
    {
      err = true;
      if (print)
      {
        DataRacePrint(u, v, WAW_PSUM);
        messages << "\tPSUM Write range : "
        << AddrRange::print_text_ars<std::list<AddrRange> >(
            u->get_psum_out_footprint());
        messages << "\tPSUM Write range : "
        << AddrRange::print_text_ars<std::list<AddrRange> >(
            v->get_psum_out_footprint());
      }
    }
  }
  return err;
}

void WaveGraphChecker::DataRacePrint(WaveOp* u, WaveOp*v, RaceKind rk)
{
  ErrorPrefix();
  messages << "Potential ";
  
  switch (rk)
  {
    case WAW_SB:
      messages << "WAW hazard in SB between W:";break;
    case WAW_PSUM:
      messages << "WAW hazard in PSUM between W:";break;
    case RAW_PSUM:WAR_PSUM:
      messages << "RAW or WAR hazard in PSUM between R:";break;
    case RAW_SB:WAR_SB:
      messages << "RAW or WAR hazard in SB between R:";break;
    default:break;
  }
  messages << u->get_name() << " and "
    << "W:" << v->get_name() << std::endl;
}

/// This is almost brute-force
/// FIXME : If the number of address ranges is significantly large,
///         then we need to implement more efficient overlap check.
///         At the moment, brute-force should be fine due to small number of
///         address ranges.
bool WaveGraphChecker::AddrSOverlap (
    std::list<AddrRange>& a
    , std::list<AddrRange>& b
    )
{
  bool overlap = false;
  if (a.size() && b.size())
  {
    for(auto ar : a)
    {
      for(auto br : b)
      {
        if ((overlap = AddrOverlap(ar, br))) break;
      }
    }
  }
  return overlap;
}

inline bool WaveGraphChecker::AddrOverlap (AddrRange a, AddrRange b)
{
  return (!(a.end < b.begin || a.begin > b.end));
}

bool WaveGraphChecker::RunDataRaceChecker()
{
  bool err = false;
  std::vector<std::list<vertex_t>*> v_list;
  std::vector<std::vector<map_v2reach_v*>* > vec_v2reach_v;
  auto b_search = [](set_v* pi, vertex_t v, graph_t& g)
  { 
    bfs_target_visitor<set_v> vis(pi);
    auto indexmap = boost::get(boost::vertex_index, g);
    auto colormap = boost::make_vector_property_map<boost::default_color_type>
      (indexmap);
    boost::queue<vertex_t> buffer;

    boost::breadth_first_search(g, v, buffer, vis, colormap);
  };

  v_list.push_back(&mLDops);
  v_list.push_back(&mSTops);
  v_list.push_back(&mACTops);
  v_list.push_back(&mPOOLops);
  v_list.push_back(&mMMops);
  for(int i = 0;i < v_list.size();++i)
  {
    std::vector<map_v2reach_v*>* vec_reachable_vtxs =
      new std::vector<map_v2reach_v*>;
    vec_v2reach_v.push_back(vec_reachable_vtxs);
    for(auto v_ : *v_list[i])
    {
      set_v* pi_v = new set_v;
      b_search(pi_v, v_, wg);
      for(int j = 0;j < v_list.size();++j)
      {
        if (vec_v2reach_v[i]->size() != v_list.size())
        {
          vec_v2reach_v[i]->push_back(new map_v2reach_v);
        }
        if (i == 0 || i != j)
        {
          set_v* pruned_pi_v = new set_v;
          for(auto u_ : *v_list[j])
          {
            if (pi_v->find(u_) == pi_v->end())
            {
              if (DataRace(wg[v_], wg[u_], false) ||
                  DataRace(wg[u_], wg[v_], false))
              {
                pruned_pi_v->insert(u_);
              }
            }
          }
          std::vector<map_v2reach_v*>* t = vec_v2reach_v[i];
          (*t)[j]->insert(
              std::pair<vertex_t, set_v* >(v_, pruned_pi_v)
              );
        }
      }
      delete pi_v;
    }
  }
  std::cout << "INFO:Starting Data Race Check "
    << "after Reachability Data Construction!" << std::endl;
  for(int i = LD;i < v_list.size();++i)
  {
    std::vector<map_v2reach_v*>* s = vec_v2reach_v[i];
    for(int j = i + 1;j < v_list.size();++j)
    {
      std::vector<map_v2reach_v*>* t;
      t = vec_v2reach_v[j];
      if (i == ST && j == MM)
      {
      } else {
        InfoPrefix();
        messages << "Checking data race between "
          << WaveOpType(i) << " and " << WaveOpType(j) << std::endl;;
        if (v_list[i]->size() && v_list[j]->size())
        {
          err |= DataRaceChecker(
              *v_list[i]
              , (*s)[j]
              , *v_list[j]
              , (*t)[i]
              );
        }
      }
    }
  }
  InfoPrefix();
  messages << "Checking data race between "
    << WaveOpType(0) << " and " << WaveOpType(0) << std::endl;;
  if (v_list[0]->size())
  {
    err |= DataRaceChecker(
        *v_list[0]
        , (*vec_v2reach_v[0])[0]
        ,*v_list[0]
        , (*vec_v2reach_v[0])[0]
        );
  }
  return err;
}

// WaveGraphChecker::MakeImplicitEdgesExplicit()
// This method creates an edge between wave ops that are excuted
// on the same engine sequentially. For example, even if one of MatMuls
// is not dependent on the other, since they are executed on PE,
// they are supposed to start and finish sequentially, which makes the oder
// from event-checker perspective. Thus, we create an edge between them. 
// Note that this is only for event-checker.
void WaveGraphChecker::MakeImplicitEdgesExplicit()
{
  struct prev_engine_op {
    bool first;
    vertex_t prev;
  }; // prev_engine_op
  prev_engine_op prev_pe; prev_pe.first = true;
  prev_engine_op prev_dma; prev_dma.first = true;
  prev_engine_op prev_act; prev_act.first = true;
  prev_engine_op prev_pool; prev_pool.first = true;
  vertex_t cur_v, prev_v;
  bool create_edge = false;

  auto prev_exam = [](prev_engine_op& p)
  {
    bool c_edge = false;
    if (p.first == false)
    {
      c_edge = true;
    } else
    {
      p.first = false;
    }
    return c_edge;
  };

  auto make_edge = [](bool make, vertex_t current_v, prev_engine_op& p
      , graph_t& g)
  {
    if (make)
    {
      if (!boost::edge(p.prev, current_v, g).second)
      {
        EventEdge ev = {255, 0, 0};
        boost::add_edge(p.prev, current_v, ev, g);
      }
    }
    p.prev = current_v;
  };
  for(auto op : mJson["waveops"])
  {
    create_edge = false;
    std::string wop_type = op["waveop_type"].get<std::string>();
    cur_v = 
      mWaveOp2V[mName2WaveOp[op["waveop_name"].get<std::string>()]];
    if (!wop_type.compare("Activation"))
    {
      make_edge(prev_exam(prev_act), cur_v, prev_act, wg);
    }
    else if (!wop_type.compare("MatMul"))
    {
      make_edge(prev_exam(prev_pe), cur_v, prev_pe, wg);
    }
    else if (!wop_type.compare("Pool") || !wop_type.compare("ResAdd")
        || !wop_type.compare("Reciprocal")
        || !wop_type.compare("RegLoad")
        || !wop_type.compare("RegStore")
        || !wop_type.compare("TpbCopy")
        || !wop_type.compare("TensorTensor")
        || !wop_type.compare("TensorScalar")
        || !wop_type.compare("TensorScalarPtr")
        || !wop_type.compare("Multiply")
        || !wop_type.compare("ScaleAdd")
        || !wop_type.compare("Add")
        || !wop_type.compare("Sub")
        || !wop_type.compare("ClipByValue")
        || !wop_type.compare("Maximum")
        || !wop_type.compare("Minimum"))
    {
      make_edge(prev_exam(prev_pool), cur_v, prev_pool, wg);
    }
    else if (!wop_type.compare("SBAtomSave") ||
        !wop_type.compare("SBAtomLoad") ||
        !wop_type.compare("Nop"))
    {
      if (op["engine"] != nullptr)
      {
        std::string eng_name = op["engine"].get<std::string>();
        if (!eng_name.compare("ActivationEng"))
        {
          make_edge(prev_exam(prev_act), cur_v, prev_act, wg);
        }
        else if (!eng_name.compare("PeArrayEng"))
        {
          make_edge(prev_exam(prev_pe), cur_v, prev_pe, wg);
        }
        else if (!eng_name.compare("PoolEng"))
        {
          make_edge(prev_exam(prev_pool), cur_v, prev_pool, wg);
        }
        else if (!eng_name.compare("DmaEng") || !eng_name.compare("AngelEng"))
        {
          make_edge(prev_exam(prev_dma), cur_v, prev_dma, wg);
        }
        else
        {
          assert(0);
        }
      } else
      {
        make_edge(prev_exam(prev_dma), cur_v, prev_dma, wg);
      }
    }
    else
    {
      std::cerr << wop_type << " is not supported" << std::endl;
      assert(0);
    }
  }
}

bool WaveGraphChecker::RunEventConflictChecker()
{
  //MakeImplicitEdgesExplicit();
  return mEventChecker->Run();
}

