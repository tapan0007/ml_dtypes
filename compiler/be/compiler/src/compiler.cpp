#include <fstream>
#include <iostream>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>

#include "utils/inc/asserter.hpp"
#include "utils/inc/debug.hpp"
#include "utils/inc/printers.hpp"
#include "utils/inc/datatype.hpp"

#include "nets/inc/network.hpp"
#include "events/inc/events.hpp"
#include "events/inc/eventmgr.hpp"

#include "arch/inc/arch.hpp"
#include "memmgr/inc/statebuffermgr.hpp"
#include "codegen/inc/codegen.hpp"
#include "wavecode/inc/wavecode.hpp"
#include "schedule/inc/scheduler.hpp"
#include "layers/inc/layer.hpp"
#include "nets/inc/network.hpp"

//#include "printer.hpp"


namespace kcc {
namespace arch {
    class Arch;
    class PeArray;
    class PsumBuffer;
    class PoolingEng;
    class ActivationEng;
    class StateBuffer;
}


namespace layers {
    class Layer;
}
namespace schedule {
    class Scheduler;
}
namespace memmgr {
    class StateBufferMgr;
}
namespace codegen {
    class CodeGen;
}


static
FILE* openObjectFile(const std::string& objFileName, const char* engineName)
{
    std::cout << "Wavegraph code generation: Generating " << engineName
              << " instructions to file '" << objFileName << "'\n";
    FILE* file = fopen(objFileName.c_str(), "wb");
    Assert(file, "Cannot open PE array object file ", objFileName.c_str());
    return file;
}

static
void writeOutJson(nets::Network* ntwk, const char* jsonInFileName)
{
    char JsonOutFileName[256];
    const char* p = jsonInFileName;
    char* q = JsonOutFileName;
    while (*p) {
        if (*p == '.') {
            *q++ = '-'; *q++ = 'o'; *q++ = 'u'; *q++ = 't';
        }
        *q++ = *p++;
    }
    *q = '\0';

    std::ofstream os(JsonOutFileName);

    std::cout << "Writing NN JSON to file '" << JsonOutFileName << "'\n";
    cereal::JSONOutputArchive ar(os);
    ntwk->save(ar);
    std::cout << "Finished writing NN JSON to file '" << JsonOutFileName << "'\n";
}

//------------------------------------------------

int
Main(int argc, char* argv[])
{
#if 1
    bool PrintLevels   = false;
    bool PrintSchedule = false;
    bool PrintDot      = false;
    bool PrintLayers   = false;
#endif
    bool DoBatching    = false;
    bool ParallelStreams = false;
    const char* JsonInFileName = nullptr;
    bool useWave = false;

    int i = 1;
    while (i < argc) {
        std::string arg(argv[i]);
#if 1
        if (arg == "--print-layers") {
            PrintLayers = true;
        } else if (arg == "--print-levels") {
            PrintLevels = true;
        } else if (arg == "--print-sched") {
            PrintSchedule = true;
        } else if (arg == "--no-print-sched") {
            PrintSchedule = false;
        } else if (arg == "--print-dot") {
            PrintDot = true;
        } else if (arg == "--batch" or arg == "--batching") {
            DoBatching = true;
        } else if (arg == "--parallel_streams") {
            ParallelStreams = true;
        } else
#endif
        if (arg == "--json") {
            if (JsonInFileName) {
                std::cerr << "Must specify net" << "\n";
                exit(1);
            }
            JsonInFileName = argv[i+1];
            i += 1;
        } else if (arg == "--wavegraph") {
            if (JsonInFileName) {
                std::cerr << "Must specify net" << "\n";
                exit(1);
            }
            useWave = true;
            JsonInFileName = argv[i+1];
            i += 1;
        } else {
            std::cerr << "Wrong argument: " << arg << "\n";
            std::cerr << "Legal argumens:\n";
            std::cerr << "  --parallel_streams\n";
            std::cerr << "  --wavegraph\n";
            std::cerr << "  --json\n";
            exit(1);
        }

        ++i;
    }

    if (JsonInFileName == nullptr) {
        std::cerr << "Must specify net" << "\n";
        exit(1);
    }


    //------------------------------------------------
    arch::Arch::init();
    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer psumBuf(arch.gPsumBuffer());
    std::cout << "Generating Arch '" << arch.gArchVersion() << "'\n";

    // Does not matter which DataType because entry index is 0.
    const utils::DataTypeFloat32 dtypeFloat32;
    std::cout << "PSUM buffer, bank 0, entry 0: TPB address =  " << psumBuf.gEntryTpbAddress(0, 0, dtypeFloat32) << "'\n";
    std::cout << "PSUM buffer, bank 1, entry 0: TPB address =  " << psumBuf.gEntryTpbAddress(1, 0, dtypeFloat32) << "'\n";
    std::cout << "Arch: number TPB events = " << arch::Arch::gNumberTpbEvents() << "\n";
    std::cout << "Events: invalid EventId = " << events::EventId_Invalid() << "\n";

#if 0
    const arch::StateBuffer stateBuf(arch.gStateBuffer());
    std::cout << "State buffer, partition size =  " << stateBuf.gPartitionSizeInBytes() << "'\n";
    std::cout << "State buffer, partition 0, entry 0: TPB address =  " << stateBuf.gEntryTpbAddress(0, 0) << "'\n";
    std::cout << "State buffer, partition 1, entry 0: TPB address =  " << stateBuf.gEntryTpbAddress(1, 0) << "'\n";
    std::cout << "State buffer, All zero TPB address =  " << stateBuf.gAllZeroOffsetTpbAddress() << "'\n";
    std::cout << "State buffer, delta TPB address between part 0 and 1=  "
              << (stateBuf.gEntryTpbAddress(1, 0) - stateBuf.gEntryTpbAddress(0, 0)) << "'\n";
#endif


    //------------------------------------------------
    nets::Network network(arch);
    nets::Network* ntwk = &network;

    {
        std::cout << "Reading NN from JSON file '" << JsonInFileName << "'\n";
        std::ifstream is(JsonInFileName);
        const bool isOpen = is.is_open();
        Assert(isOpen, "JSON input file '", JsonInFileName, "' is not open\n");
        cereal::JSONInputArchive ar(is);
        ntwk->rUseWave(useWave);
        ntwk->load(ar);
    }

    ntwk->rDoBatching(DoBatching);

    if (! useWave) {
        std::string objFileName(ntwk->gName());
        objFileName += ".tpb";
        //--------------------------------------------------------
        schedule::Scheduler* scheduler = new schedule::Scheduler();
        std::cout << "Scheduling NN '" << ntwk->gName() << "'\n";
        scheduler->Schedule(ntwk);
        //ntwk->rLevels(scheduler->gLevels());

        //--------------------------------------------------------
        memmgr::StateBufferMgr* sbmgr = new memmgr::StateBufferMgr(arch, ntwk);
        std::cout << "Calculating FMAP and weight state buffer addresses\n";
        sbmgr->calcLayerFmapAddresses();

        //--------------------------------------------------------
        codegen::CodeGen codegen(ntwk, arch);

        std::cout << "Codegen: Generating instructions to file '" << objFileName << "'\n";
        codegen.generate(objFileName.c_str());

        if (false) {
            //--------------------------------------------------------
            const auto printer = new kcc::utils::Printer(ntwk);

            if (PrintLayers) {
                printer->printNetwork();
                std::cout << "\n";
            }

            if (PrintLevels) {
                std::cout << "By level\n";
                printer->printLevels(scheduler);
                std::cout << "\n";
            }

            if (PrintSchedule) {
                std::cout << "By scheduling\n";
                printer->printSched();
                std::cout << "\n";
            }

            if (PrintDot) {
                std::cout << "Dot\n";
                printer->printDot();
                std::cout << "\n";
            }
        }
        writeOutJson(ntwk, JsonInFileName);
    } else {
        wavecode::WaveCode::InstrStreams instrStreams;
        if (ParallelStreams) {
            std::string objFileName;

            objFileName = ntwk->gName() + "-pe.tpb";
            instrStreams.m_PeArrayInstrStream       = openObjectFile(objFileName, "PE array");

            objFileName = ntwk->gName() + "-sp.tpb";
            instrStreams.m_StreamProcInstrStream    = openObjectFile(objFileName, "stream processor");

            objFileName = ntwk->gName() + "-dma.tpb";
            instrStreams.m_DmaInstrStream    = openObjectFile(objFileName, "DMA");

            objFileName = ntwk->gName() + "-pool.tpb";
            instrStreams.m_PoolEngInstrStream       = openObjectFile(objFileName, "pooling engine");

            objFileName = ntwk->gName() + "-act.tpb";
            instrStreams.m_ActEngInstrStream        = openObjectFile(objFileName, "activation engine");

            if (ParallelStreams) {
                events::EventMgr eventMgr(*ntwk);
                eventMgr.processWaveops();
            }
        } else {
            std::string objFileName(ntwk->gName() + ".tpb");
            FILE* file = openObjectFile(objFileName, "all");

            instrStreams.m_StreamProcInstrStream    = file;
            instrStreams.m_PeArrayInstrStream       = file;
            instrStreams.m_PoolEngInstrStream       = file;
            instrStreams.m_ActEngInstrStream        = file;
            instrStreams.m_DmaInstrStream           = file;
        }

        writeOutJson(ntwk, JsonInFileName);

        wavecode::WaveCode waveCode(ntwk, arch);
        waveCode.generate(instrStreams, ParallelStreams);
    }

    return 0;
}

} // namespace kcc

int
main(int argc, char* argv[])
{
    return kcc::Main(argc, argv);
}

