#pragma once

#ifndef KCC_COMPILER_STARGAZER_H
#define KCC_COMPILER_STARGAZER_H

#include <assert.h>

#include <string>
#include <vector>
#include <map>



#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

#include "wave/inc/waveop.hpp"


namespace kcc {

namespace arch {
    class Arch;
}
namespace wavecode {
    class WaveCode;
}
namespace events {
    class EventMgr;
}


namespace compiler {

using namespace utils;



//--------------------------------------------------------
// The whole neural net
//--------------------------------------------------------
class Stargazer {
public:
    int Main(int argc, char* argv[]);

private:
    FILE* openObjectFile(const std::string& objFileName, const char* engineName);
    void writeOutJson(nets::Network& ntwk, const char* jsonInFileName, const char* ext);
    void generateSequentialStream(nets::Network& ntwk, const char* JsonInFileName,
            wavecode::WaveCode& waveCode);
    void generateAngelStreams(nets::Network& ntwk, const char* JsonInFileName,
            wavecode::WaveCode& waveCode, events::EventMgr& eventMgr, bool kelf);
    void generateKelfStreams(nets::Network& ntwk, const char* JsonInFileName,
            wavecode::WaveCode& waveCode, events::EventMgr& eventMgr,
            bool kelf, bool realDma, bool useSem);

private:
    bool m_SbufCopy;

}; // Stargazer




} // namespace compiler
} // namespace kcc

#endif // KCC_COMPILER_STARGAZER_H

