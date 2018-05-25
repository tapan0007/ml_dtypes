#include "wave_graph.h"
#include <fstream>

int main(int argc, char* argv[])
{
  std::ifstream in_wave(argv[1]);
  json j;
  in_wave >> j;

  WaveGraphChecker wg(j);
  //wg.write_graph_viz();
  wg.structure_check();
  wg.RunDataRaceChecker();
  return 0;
}
