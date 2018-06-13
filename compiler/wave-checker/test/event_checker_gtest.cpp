#include <fstream>
#include <string>
#include <cassert>
#include <vector>
#include "gtest/gtest.h"
#include "packages/nlohmann/json.hpp"
#include "event_check.h"
#include "wave_graph.h"
#include <boost/program_options.hpp>

po::variables_map g_cli;
std::string wave_graph_filename;

namespace {
  class EventCheckerTest: public ::testing::Test {
    public:
      EventCheckerTest() {}
    protected:
      virtual void SetUp() {
        CommandLineOptions cli = parse_cli();
        std::ifstream in_wave;
        in_wave.open(wave_graph_filename.c_str());
        if (in_wave.is_open())
        {
          in_wave >> j;
          CommandLineOptions cli;
          wgc = new WaveGraphChecker(j, cli);
          in_wave.close();
        }
        else
        {
          std::cerr << "ASSERT::Wave graph is not read!" << std::endl;
          assert(0);
        }
      }

      virtual void TearDown() {
        delete wgc;
      }

      CommandLineOptions parse_cli()
      {
        CommandLineOptions cli;
        cli.structure_check = g_cli["structure-check"].as<bool>();
        cli.data_race_check = g_cli["data-race-check"].as<bool>();
        cli.event_conflict_check = g_cli["event-conflict-check"].as<bool>();
        cli.color = g_cli["color"].as<bool>();
        return cli;
      }

      std::string GetEngine(json& op)
      {
        std::string eng;
        std::string waveop_type = op["waveop_type"].get<std::string>();
        eng = waveop_type;
        if (!waveop_type.compare("Nop"))
        {
          std::string engine_name = op["engine_name"].get<std::string>();
          if (!engine_name.compare("PoolEng"))
          {
            eng = "Pool";
          }
          if (!engine_name.compare("ActivationEng"))
          {
            eng = "Activation";
          }
          if (!engine_name.compare("PeArrayEng"))
          {
            eng = "MatMul";
          }
          if (!engine_name.compare("DmaEng"))
          {
            eng = "DMA";
          }
        }
        else if (!waveop_type.compare("Pool") || !waveop_type.compare("ResAdd"))
        {
          eng = "Pool";
        }
        else if (!waveop_type.compare("SBAtomFile")
            || !waveop_type.compare("SBAtomSave"))
        {
          eng = "DMA";
        }
        return eng;
      }
      bool CheckImplicitEdges(std::string engine)
      {
        WaveOp* prev = nullptr;
        for(auto op : j["waveops"])
        {
          if (!GetEngine(op).compare(engine))
          {
            vertex_t prev_v, cur_v;
            WaveOp* cur = (wgc->test_get_mName2WaveOp())[op["waveop_name"]]; 
            if (prev != nullptr)
            {
              prev_v = wgc->test_get_mWaveOp2V()[prev];
              cur_v = wgc->test_get_mWaveOp2V()[cur];
              edge_t e;
              bool found;
              std::tie(e, found) = boost::edge(prev_v,cur_v,wgc->test_get_wg());
              if (!found) 
              {
                return false;
              }
            }
            prev = cur;
          }
        }
        return true;
      }

      bool CheckEdgesbetweenWaveOpsOnTheSameEngine()
      {
        if (!CheckImplicitEdges("MatMul"))
        {
          return false;
        }
        if (!CheckImplicitEdges("Pool"))
        {
          return false;
        }
        if (!CheckImplicitEdges("Activation"))
        {
          return false;
        }
        if (!CheckImplicitEdges("DMA"))
        {
          return false;
        }
        return true;
      }

      WaveGraphChecker* wgc;
      nlohmann::json j;
  };

  TEST_F(EventCheckerTest, ImplicitEdges2ExplicitOnes_Test)
  {
    wgc->test_call_MakeImplicitEdgesExplicit();
    ASSERT_EQ(CheckEdgesbetweenWaveOpsOnTheSameEngine(), true);
  }
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  po::options_description desc{"Options"};
  desc.add_options()
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
  po::store(po::command_line_parser(argc, argv).
         options(desc).allow_unregistered().run(), g_cli);
  wave_graph_filename = g_cli["wave-graph-file"].as<std::string>();

  return RUN_ALL_TESTS();
}
