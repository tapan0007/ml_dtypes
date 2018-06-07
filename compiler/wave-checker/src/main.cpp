#include "wave_graph.h"
#include "wc_common.h"
#include <boost/program_options.hpp>
#include <fstream>

po::variables_map g_cli; // Global command line options

int main(int argc, char* argv[])
{

  enum ERROR_TYPE {OK, ERROR};
  int err = OK;
  po::options_description desc{"Options"};
  desc.add_options()
    ("help,h", "Help")
    ("stdout", po::bool_switch()->default_value(false)
     , "Redirects output streams to standard output")
    ("fileout", po::value<std::string>()
     , "Name and location of an output file")
    ("color", po::bool_switch()->default_value(false)
     , "Enables colored messages. e.g. ERROR: red and INFO : blue")
    ("wave-graph-file,w ", po::value<std::string>()
     , "Name and location of an input wave graph")
    ("structure-check", po::bool_switch()->default_value(false)
     , "Enable structure checker")
    ("data-race-check", po::bool_switch()->default_value(false)
     , "Enable data race checker");
  //po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), g_cli);

  //std::ifstream in_wave(argv[1]);
  if (g_cli.count("help") || !g_cli.size() || !g_cli.count("wave-graph-file")) {
    std::cout << desc << std::endl;
  } else {
    std::ifstream in_wave(g_cli["wave-graph-file"].as<std::string>().c_str());
    if (in_wave.is_open())
    {
      json j;
      in_wave >> j;
      WaveGraphChecker wg(j);
      //wg.write_graph_viz();
      if (g_cli["structure-check"].as<bool>()) {
        if (wg.structure_check()) {
          err = ERROR;
        }
      }
      if (g_cli["data-race-check"].as<bool>()) {
        if (wg.RunDataRaceChecker()) {
          err = ERROR;
        }
      }
      if (g_cli.count("fileout"))
      {
        std::ofstream o_file(g_cli["fileout"].as<std::string>());
        o_file << wg.get_msg().str();
        o_file.close();
      } else
      {
        if (g_cli["stdout"].as<bool>()) {
          std::cout << wg.get_msg().str();
        }
      }
    }
    else {
      std::cout << "Error opening file "
        << g_cli["wave-graph-file"].as<std::string>() << std::endl;
    }
  }
  return err;
}
