#include "meminfo.h"
#include <cassert>

void MemInfo::merge_addr_range(AddrRange ar)
{
  typedef std::list<AddrRange>::iterator itr_f; // Footprint iterator
  itr_f itr, itr_f_begin, itr_f_end;
  bool start_merge = false;
  itr_f_begin = mem_footprints.begin();
  itr_f_end = mem_footprints.end();
  for(itr = mem_footprints.begin();itr != mem_footprints.end();++itr)
  {
    AddrRange a = *itr;
    if (!start_merge) {
      if ((ar.begin >= a.begin && ar.begin <= a.end) ||
          (ar.end >= a.begin && ar.end <= a.end) ||
          (ar.begin <= a.begin && ar.end >= a.end)
          ) {
        itr_f_begin = itr;
        itr_f_end = itr;
        start_merge = true;
      }
      if (!start_merge) {
        if (ar.end < a.begin) {
          itr_f_end = itr;
          break;
        }
      }
    } else {
      if (ar.end < a.begin) {
        itr_f_end = --itr;
        break;
      }
      itr_f_end = itr;
    }
  }
  if (start_merge) {
    // Rip off those overlapping with ar and add merged one
    AddrRange a(0,0);
    if (ar.begin < itr_f_begin->begin) a.begin = ar.begin;
    else a.begin = itr_f_begin->begin;
    if (ar.end > itr_f_end->end) a.end = ar.end;
    else a.end = itr_f_end->end;
    mem_footprints.erase(itr_f_begin, ++itr_f_end);
    mem_footprints.insert(itr_f_end, a);
  } else {
    // since there's no overlap, simply put ar without breaking sorted order
    mem_footprints.insert(itr_f_end, ar);
  }
}

void WaveOpMemInfo::compute_bias_footprint()
{
  tonga_addr e = m_bias_sb_addr + MemInfo::compute_elem_size(m_bias_dtype) - 1;
  AddrRange ar(m_bias_sb_addr, e);
  // Since bias is read-only, it is merged into sb_in_mi.footprints
  sb_in_mi.merge_addr_range(ar);
}
