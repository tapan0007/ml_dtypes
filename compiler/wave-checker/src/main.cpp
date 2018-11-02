#include "wave_graph.h"
#include "wc_common.h"
#include <boost/program_options.hpp>
#include <fstream>

po::variables_map g_cli; // Global command line options

CommandLineOptions parse_cli()
{
  CommandLineOptions cli;
  cli.structure_check = g_cli["structure-check"].as<bool>();
  cli.data_race_check = g_cli["data-race-check"].as<bool>();
  cli.event_conflict_check = g_cli["event-conflict-check"].as<bool>();
  cli.color = g_cli["color"].as<bool>();
  return cli;
}

int main(int argc, char* argv[])
{

  enum ERROR_TYPE {OK, ERROR, NOFILE};
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
     , "Enable data race checker")
    ("event-conflict-check", po::bool_switch()->default_value(false)
     , "Enable event conflict checker");
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
      WaveGraphChecker wg(j, parse_cli());
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
      if (g_cli["event-conflict-check"].as<bool>())
      {
        if (wg.RunEventConflictChecker())
        {
          err = ERROR;
        }
      }
      std::string final_msg;
      if (err != OK)
      {
        final_msg = "\nWaveChecker : FAILED\n";
      } else
      {
        final_msg = "\nWaveChecker : PASSED\n";
      }
      if (g_cli.count("fileout"))
      {
        std::ofstream o_file(g_cli["fileout"].as<std::string>());
        o_file << wg.get_msg().str();
        o_file << final_msg;
        o_file.close();
      } else
      {
        if (g_cli["stdout"].as<bool>()) {
          std::cout << wg.get_msg().str();
          std::cout << final_msg;
        }
      }
      //wg.write_graph_viz();
    }
    else {
      std::cout << "Error opening file "
        << g_cli["wave-graph-file"].as<std::string>() << std::endl;
      err = NOFILE;
    }
  }
  return err;
}
