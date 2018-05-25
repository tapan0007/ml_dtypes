#include "meminfo.h"

// Merge an AR into a container with only one item
void test_one_elem_mem_footprints(AddrRange a1, AddrRange a2)
{
  std::string dt("float16");
  MemInfo mi(MemInfo_Params(0,0,0,0,0,0,0,0,dt,true), 0);
  a1.print_ar(true);
  mi.mem_footprints.push_back(a1);
  a2.print_ar(true);
  mi.merge_addr_range(a2);
  //std::cout << mi.mem_footprints.size() << std::endl;
  for(auto a : mi.mem_footprints)
  {
    std::cout << "[" << a.begin << ", " << a.end << "]" << std::endl;
  }
  AddrRange::print_ars<std::list<AddrRange> >(mi.mem_footprints);
}

std::list<AddrRange> create_ars()
{
  std::list<AddrRange> ars;
  ars.push_back(AddrRange(3, 7));
  ars.push_back(AddrRange(10, 13));
  return ars;
}

void test_multi_elems_mem_gootprints(std::list<AddrRange> a1, AddrRange a2)
{
  std::string dt("float16");
  MemInfo mi(MemInfo_Params(0,0,0,0,0,0,0,0,dt,true), 0);
  mi.mem_footprints = a1;
  std::cout << "Merging ";a2.print_text_ar();
  //AddrRange::print_ars<std::list<AddrRange> >(mi.mem_footprints);
  AddrRange::print_text_ars<std::list<AddrRange> >(mi.mem_footprints);
  mi.merge_addr_range(a2);
  AddrRange::print_text_ars<std::list<AddrRange> >(mi.mem_footprints);
  //AddrRange::print_ars<std::list<AddrRange> >(mi.mem_footprints);
}

int main()
{
  //test_one_elem_mem_footprints(AddrRange(13, 15), AddrRange(3, 5));
  //test_one_elem_mem_footprints(AddrRange(3, 5), AddrRange(13, 15));
  //test_one_elem_mem_footprints(AddrRange(3, 9), AddrRange(5, 13));
  //test_one_elem_mem_footprints(AddrRange(5, 13), AddrRange(3, 9));
  test_multi_elems_mem_gootprints(create_ars(), AddrRange(0, 1));
  test_multi_elems_mem_gootprints(create_ars(), AddrRange(7, 8));
  test_multi_elems_mem_gootprints(create_ars(), AddrRange(15, 17));
  test_multi_elems_mem_gootprints(create_ars(), AddrRange(1, 4));
  test_multi_elems_mem_gootprints(create_ars(), AddrRange(1, 5));
  test_multi_elems_mem_gootprints(create_ars(), AddrRange(1, 8));
  test_multi_elems_mem_gootprints(create_ars(), AddrRange(4, 8));
  test_multi_elems_mem_gootprints(create_ars(), AddrRange(4, 6));
  test_multi_elems_mem_gootprints(create_ars(), AddrRange(4, 11));
  test_multi_elems_mem_gootprints(create_ars(), AddrRange(4, 14));
  test_multi_elems_mem_gootprints(create_ars(), AddrRange(1, 11));
  test_multi_elems_mem_gootprints(create_ars(), AddrRange(1, 15));
  return 0;
}
