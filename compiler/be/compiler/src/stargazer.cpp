#include <fstream>
#include <iostream>

#include <stdio.h>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>

#include "utils/inc/version.hpp"
#include "utils/inc/asserter.hpp"
#include "utils/inc/misc.hpp"
#include "utils/inc/debug.hpp"
#include "utils/inc/datatype.hpp"

#include "nets/inc/network.hpp"
#include "events/inc/events.hpp"
#include "events/inc/eventmgr.hpp"

#include "arch/inc/arch.hpp"
#include "wavecode/inc/wavecode.hpp"
#include "nets/inc/network.hpp"

#include "compiler/inc/stargazer.hpp"


namespace kcc {
namespace arch {
    class Arch;
    class PeArray;
    class PsumBuffer;
    class PoolingEng;
    class ActivationEng;
    class StateBuffer;
}

namespace compiler {

enum {
    EnginePrintFormatSize = 20
};

//--------------------------------------------------------
FILE*
Stargazer::openObjectFile(const std::string& objFileName, const char* engineName)
{
    std::cout << "    " << std::setw(EnginePrintFormatSize) << std::left << engineName << objFileName << "\n";
    FILE* file = fopen(objFileName.c_str(), "wb");
    Assert(file, "Cannot open %s object file: %s", engineName, objFileName.c_str());
    return file;
}

//--------------------------------------------------------
void
Stargazer::writeOutJson(nets::Network& ntwk, const char* jsonInFileName, const char* ext)
{
    char JsonOutFileName[256];
    strncpy(JsonOutFileName, jsonInFileName,
        ArraySizeof(JsonOutFileName)-1);
    char* q = JsonOutFileName + (strlen(JsonOutFileName) - 5);
    Assert(0 == strcmp(q, ".json"),
        "Input Json file name does not end in '.json'");
    sprintf(q, "-%s.json", ext);

    std::ofstream os(JsonOutFileName);

    std::cout << "Writing NN JSON to file '" << JsonOutFileName << "'\n";

    try {
        cereal::JSONOutputArchive ar(os);
        ntwk.save(ar);
    } catch (const cereal::Exception& except) {
        std::cerr << "Error <"  << except.what() << "> when writing JSON file '" << JsonOutFileName << "'\n";
        exit(1);
    } catch (...) {
        std::cerr << "Error writing JSON file '" << JsonOutFileName << "'\n";
        exit(1);
    }
}


//------------------------------------------------
void
Stargazer::generateSequentialStream(
        nets::Network& ntwk,
        const char* JsonInFileName,
        wavecode::WaveCode& waveCode)
{
    std::cout << "Wavegraph code generation: ";
    std::string objFileName(ntwk.gName() + ".tpb");
    FILE* file = openObjectFile(objFileName, "all");

    wavecode::WaveCode::InstrStreams instrStreams;
    instrStreams.m_StreamProc.m_InstrStream    = file;
    instrStreams.m_PeArray.m_InstrStream       = file;
    instrStreams.m_PoolEng.m_InstrStream       = file;
    instrStreams.m_ActEng.m_InstrStream        = file;
    instrStreams.m_Angel.m_InstrStream         = file;

    writeOutJson(ntwk, JsonInFileName, "seqtpb");
    waveCode.generate(instrStreams, false);

    fclose(file);

    instrStreams.m_StreamProc.m_InstrStream    = nullptr;
    instrStreams.m_PeArray.m_InstrStream       = nullptr;
    instrStreams.m_PoolEng.m_InstrStream       = nullptr;
    instrStreams.m_ActEng.m_InstrStream        = nullptr;
    instrStreams.m_Angel.m_InstrStream         = nullptr;
}

//------------------------------------------------
void
Stargazer::generateAngelStreams(
        nets::Network& ntwk,
        const char* JsonInFileName,
        wavecode::WaveCode& waveCode,
        events::EventMgr& eventMgr,
        bool kelf)
{
    eventMgr.rKelf(kelf);
    std::cout << "  Angel non-reserved events:\n"
              << "    First non-reserved = " << eventMgr.EventId_FirstNonReserved() << "\n"
              << "    Last non-reserved = "  << events::EventId_LastNonReserved()   << "\n";
    ntwk.rUseSem(false);
    std::cout << "Wavegraph code generation for SIM/Angel:\n";
    std::cout << "    " << std::setw(EnginePrintFormatSize) << std::left << "Engine" << "File\n";
    std::cout << "    " << std::setw(EnginePrintFormatSize) << std::left << "------" << "----\n";

    wavecode::WaveCode::InstrStreams instrStreams;
    std::string objFileName;
    instrStreams.m_PeArray.m_BinFile = objFileName = ntwk.gName() + "-pe.tpb";
    instrStreams.m_PeArray.m_InstrStream       = openObjectFile(objFileName, "PE-Array");

    instrStreams.m_StreamProc.m_BinFile = objFileName = ntwk.gName() + "-sp.tpb";
    instrStreams.m_StreamProc.m_InstrStream    = openObjectFile(objFileName, "Stream-Processor");

    instrStreams.m_Angel.m_BinFile = objFileName = ntwk.gName() + "-dma.tpb";
    instrStreams.m_Angel.m_InstrStream    = openObjectFile(objFileName, "Angel-Eng");

    instrStreams.m_PoolEng.m_BinFile = objFileName = ntwk.gName() + "-pool.tpb";
    instrStreams.m_PoolEng.m_InstrStream       = openObjectFile(objFileName, "Pool-Eng");

    instrStreams.m_ActEng.m_BinFile = objFileName = ntwk.gName() + "-act.tpb";
    instrStreams.m_ActEng.m_InstrStream        = openObjectFile(objFileName, "Act-Eng");
    std::cout << "\n";

    waveCode.rBinFileType(BinFileType::SimAngel);
    waveCode.DetermineEngines();
    eventMgr.processWaveops(false);
    waveCode.generate(instrStreams, true);

    writeOutJson(ntwk, JsonInFileName, "tpb");
    instrStreams.closeAll();
}

//------------------------------------------------
void
Stargazer::generateKelfStreams(
        nets::Network& ntwk,
        const char* JsonInFileName,
        wavecode::WaveCode& waveCode,
        events::EventMgr& eventMgr,
        bool kelf,
        bool realDma,
        bool useSem)
{
    eventMgr.rKelf(kelf);
    std::cout << "  Kelf non-reserved events:\n"
              << "    First non-reserved = " << eventMgr.EventId_FirstNonReserved() << "\n"
              << "    Last non-reserved = "  << events::EventId_LastNonReserved()   << "\n";
    ntwk.rUseSem(useSem);
    std::cout << "Wavegraph code generation for Qemu/Emu:\n";
    std::cout << "    " << std::setw(EnginePrintFormatSize) << std::left << "Engine" << "File\n";
    std::cout << "    " << std::setw(EnginePrintFormatSize) << std::left << "------" << "----\n";
    wavecode::WaveCode::InstrStreams instrStreams;
    std::string objFileName;
    instrStreams.m_PeArray.m_BinFile = objFileName = ntwk.gName() + "-pe.bin";
    instrStreams.m_PeArray.m_InstrStream       = openObjectFile(objFileName, "PE-Array");

    instrStreams.m_StreamProc.m_BinFile = objFileName = ntwk.gName() + "-sp.bin";
    instrStreams.m_StreamProc.m_InstrStream    = openObjectFile(objFileName, "Stream-Processor");

    instrStreams.m_Angel.m_InstrStream    = nullptr;

    instrStreams.m_PoolEng.m_BinFile = objFileName = ntwk.gName() + "-pool.bin";
    instrStreams.m_PoolEng.m_InstrStream       = openObjectFile(objFileName, "Pool-Eng");

    instrStreams.m_ActEng.m_BinFile = objFileName = ntwk.gName() + "-act.bin";
    instrStreams.m_ActEng.m_InstrStream        = openObjectFile(objFileName, "Act-Eng");
    std::cout << "\n";

    if (!realDma) {
        ntwk.revertSavedWaveops();
        ntwk.ClearEvents();
    }

    ntwk.MarkTmpLoads();
    if (m_SbufCopy) {
        ntwk.SplitReplicatedLoads();
    }

    waveCode.rBinFileType(BinFileType::RuntimeKelf);
    waveCode.DetermineEngines();
    eventMgr.processWaveops(useSem);
    //writeOutJson(ntwk, JsonInFileName, "bin2");
    waveCode.generate(instrStreams, true);

    writeOutJson(ntwk, JsonInFileName, "bin");
    instrStreams.closeAll();
}

//------------------------------------------------
int
Stargazer::Main(int argc, char* argv[])
{
    bool DoBatching    = false;
    bool ParallelStreams = true;
    const char* JsonInFileName = nullptr;
    bool realDma = false;
    bool useSem = true;
    m_SbufCopy = false;


    kcc_int32 numTpbEvents = -1;
    {
        std::cout << "Command line arguments:\n";
        for (int i = 0; i < argc; ++i) {
            std::cout << "    [" << i << "] = '" << argv[i] << "'\n";
        }
    }

    int i = 1;
    while (i < argc) {
        const std::string arg(argv[i]);
        if (arg == "--batch" or arg == "--batching") {
            DoBatching = true;
        } else if (arg == "--parallel_streams" || arg == "--parallel-streams") {
            ParallelStreams = true;
        } else if (arg == "--sequential_stream" || arg == "--sequential-stream") {
            ParallelStreams = false;
        } else if (arg == "--real-dma") {
            realDma = true;
        } else if (arg == "--number-tpb-events") {
            Assert(i+1 < argc, "Option '--number-tpb-events' has no arguments");
            numTpbEvents = atoi(argv[i+1]);
            ++i;
        } else if (arg == "--sync-with-semaphores" || arg == "-s") {
            useSem = true;
        } else if (arg == "--sync-with-events" || arg == "-e") {
            useSem = false;
        } else if (arg == "--sbuf-replay" || arg == "--sbuf-copy") {
            m_SbufCopy = true;
        } else if (arg == "--no-sbuf-replay" || arg == "--no-sbuf-copy") {
            m_SbufCopy = false;
        } else if (arg == "--wavegraph" || arg == "-w") {
            if (JsonInFileName) {
                std::cerr << "NN file name already specified" << "\n";
                exit(1);
            }
            Assert(i+1 < argc, "Option '--wavegraph' has no arguments");
            JsonInFileName = argv[i+1];
            i += 1;
        } else {
            std::cerr << "Wrong argument: " << arg << "\n";
            std::cerr << "Usage: " << argv[0] << " [options] --wavegraph wavegrephjson\n"
                      << "options:\n"
                      <<      "--parallel-stream\n"
                      <<      "--sequential-stream\n"
                      <<      "--sync-with-events\n"
                      <<      "--sync-with-semaphores\n"
                      <<      "--real-dma\n";
            exit(1);
        }

        ++i;
    }

    if (JsonInFileName == nullptr) {
        std::cerr << "Must specify NN file name" << "\n";
        exit(1);
    }


    //------------------------------------------------
    arch::Arch::init(numTpbEvents);
    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer psumBuf(arch.gPsumBuffer());
    std::cout << "Generating Arch '" << arch.gArchVersion() << "'\n";
    std::cout << "Git commit " << utils::Git::gShaLong() << "\n";

    // Does not matter which DataType because entry index is 0.
    const utils::DataTypeFloat32 dtypeFloat32;
    if (false) {
        std::cout << "PSUM buffer, bank 0, entry 0: TPB address =  "
                << psumBuf.gEntryTpbAddress(0, 0, dtypeFloat32) << "'\n";
        std::cout << "PSUM buffer, bank 1, entry 0: TPB address =  "
                << psumBuf.gEntryTpbAddress(1, 0, dtypeFloat32) << "'\n";
    }



    //------------------------------------------------
    nets::Network ntwk(arch, utils::Git::gShaLong());
    //nets::Network* network = &ntwk;

    {
        std::cout << "Reading NN from JSON file '" << JsonInFileName << "'\n";
        std::ifstream is(JsonInFileName);
        const bool isOpen = is.is_open();
        Assert(isOpen, "JSON input file '", JsonInFileName, "' is not open\n");

        try {
            cereal::JSONInputArchive ar(is);
            ntwk.rUseWave(true);
            ntwk.load(ar);
        } catch (const cereal::Exception& except) {
            std::cerr << "Error <"  << except.what() << "> when reading JSON file '" << JsonInFileName << "'\n";
            exit(1);
        } catch (...) {
            std::cerr << "Error reading JSON file '" << JsonInFileName << "'\n";
            exit(1);
        }
    }

    ntwk.rDoBatching(DoBatching);

    if (ParallelStreams) {
        //==========================================================
        std::string objFileName;
        events::EventMgr eventMgr(ntwk);
        wavecode::WaveCode waveCode(ntwk, arch, useSem);
        std::cout << "Events:\n"
            << "    Arch: number of all TPB events = " << arch::Arch::gArch().gNumberAllTpbEvents() << "\n"
            << "    Invalid event: " << events::EventId_Invalid() << "\n"
            << "    First runtime reserved events = " << eventMgr.EventId_RunTimeFirst() << "\n"
            << "    Last runtime reserved events = "  << eventMgr.EventId_RunTimeLast() << "\n"
            << "    MatMult multi fanout = "          << eventMgr.EventId_MMStartMultiSet() << "\n";

        if (!realDma) { // with Angel
            generateAngelStreams(ntwk, JsonInFileName, waveCode, eventMgr, false);
        } else { // Runtime Kelf
            generateKelfStreams(ntwk, JsonInFileName, waveCode, eventMgr, true, realDma, useSem);
        }

    } else {
        wavecode::WaveCode waveCode(ntwk, arch, false);
        generateSequentialStream(ntwk, JsonInFileName, waveCode);
    }

    std::cout << "Compiler BE: PASSED\n";

    return 0;
}

}} // kcc/compiler





