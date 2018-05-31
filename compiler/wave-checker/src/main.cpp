#include "wave_graph.h"
#include <boost/program_options.hpp>
#include <fstream>

namespace po = boost::program_options;

int main(int argc, char* argv[])
{

  po::options_description desc{"Options"};
  desc.add_options()
    ("help,h", "Help")
    ("wave-graph-file,w ", po::value<std::string>()
     , "Name and location of an input wave graph")
    ("structure-check", po::bool_switch()->default_value(false)
     , "Enable structure checker")
    ("data-race-check", po::bool_switch()->default_value(false)
     , "Enable data race checker");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  //std::ifstream in_wave(argv[1]);
  if (vm.count("help") || !vm.size() || !vm.count("wave-graph-file")) {
    std::cout << desc << std::endl;
  } else {
    std::ifstream in_wave(vm["wave-graph-file"].as<std::string>().c_str());
    if (in_wave.is_open())
    {
      json j;
      in_wave >> j;
      WaveGraphChecker wg(j);
      //wg.write_graph_viz();
      if (vm["structure-check"].as<bool>()) {
        wg.structure_check();
      }
      if (vm["data-race-check"].as<bool>()) {
        wg.RunDataRaceChecker();
      }
    }
    else {
      std::cout << "Error opening file "
        << vm["wave-graph-file"].as<std::string>() << std::endl;
    }
  }
  return 0;
}
