//#include "wavegraph.h"
#include <fstream>
#include <string>
#include <cassert>
#include <vector>
#include "gtest/gtest.h"

void read_file(std::vector<std::string>& in, const std::string file)
{
  std::ifstream infile(file);
  assert(infile.is_open());
  std::string line;
  while(std::getline(infile, line))
  {
    in.push_back(line);
  }
}

bool compare_files(std::vector<std::string>& g, std::vector<std::string>& t)
{
  if (g.size() != t.size()) return false;
  for(int i = 0;i < g.size();++i)
  {
    if (g[i].compare(t[i])) return false;
  }
  return true;
}

TEST(WCTest,CleanWaveTest)
{
  const std::string golden_log = "./sample_clean1.golden.log";
  const std::string test_log = "./sample_clean1.test.log";
  std::vector<std::string> golden;
  std::vector<std::string> test;
  read_file(golden, golden_log);
  read_file(test, test_log);
  EXPECT_EQ(true, compare_files(golden, test));
}

TEST(WCTest,StructureTest)
{
  const std::string golden_log = "./sample_act.golden.log";
  const std::string test_log = "./sample_act.test.log";
  std::vector<std::string> golden;
  std::vector<std::string> test;
  read_file(golden, golden_log);
  read_file(test, test_log);
  EXPECT_EQ(true, compare_files(golden, test));
}

TEST(WCTest,RAWTest)
{
  const std::string golden_log = "./sample_raw.golden.log";
  const std::string test_log = "./sample_raw.test.log";
  std::vector<std::string> golden;
  std::vector<std::string> test;
  read_file(golden, golden_log);
  read_file(test, test_log);
  EXPECT_EQ(true, compare_files(golden, test));
}

TEST(WCTest,WAWTest)
{
  const std::string golden_log = "./sample_waw.golden.log";
  const std::string test_log = "./sample_waw.test.log";
  std::vector<std::string> golden;
  std::vector<std::string> test;
  read_file(golden, golden_log);
  read_file(test, test_log);
  EXPECT_EQ(true, compare_files(golden, test));
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
