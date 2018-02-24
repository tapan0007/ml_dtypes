#include <fstream>
#include <iostream>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>

#include "utils/inc/debug.hpp"
#include "utils/inc/printers.hpp"

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



//------------------------------------------------

int
Main(int argc, char* argv[])
{
    kcc::utils::breakFunc(44);
#if 1
    bool PrintLevels   = false;
    bool PrintSchedule = false;
    bool PrintDot      = false;
    bool PrintLayers   = false;
#endif
    bool DoBatching    = false;
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
    const arch::StateBuffer stateBuf(arch.gStateBuffer());
    std::cout << "Generating Arch '" << arch.gArchVersion() << "'\n";

    std::cout << "PSUM buffer, bank 0, entry 0: TPB address =  " << psumBuf.gEntryTpbAddress(0, 0) << "'\n";
    std::cout << "PSUM buffer, bank 1, entry 0: TPB address =  " << psumBuf.gEntryTpbAddress(1, 0) << "'\n";

    std::cout << "State buffer, partition size =  " << stateBuf.gPartitionSizeInBytes() << "'\n";
    std::cout << "State buffer, partition 0, entry 0: TPB address =  " << stateBuf.gEntryTpbAddress(0, 0) << "'\n";
    std::cout << "State buffer, partition 1, entry 0: TPB address =  " << stateBuf.gEntryTpbAddress(1, 0) << "'\n";
    std::cout << "State buffer, All zero TPB address =  " << stateBuf.gAllZeroOffsetTpbAddress() << "'\n";
    std::cout << "State buffer, delta TPB address between part 0 and 1=  "
              << (stateBuf.gEntryTpbAddress(1, 0) - stateBuf.gEntryTpbAddress(0, 0)) << "'\n";


    //------------------------------------------------
    nets::Network network(arch);
    nets::Network* ntwk = &network;
    kcc::utils::breakFunc(44);
    {
        std::cout << "Reading NN from JSON file '" << JsonInFileName << "'\n";
        std::ifstream is(JsonInFileName);
        cereal::JSONInputArchive ar(is);
        ntwk->rUseWave(useWave);
        ntwk->load(ar);
    }

    ntwk->rDoBatching(DoBatching);

    //--------------------------------------------------------
    {
        char JsonOutFileName[256];
        const char* p = JsonInFileName;
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
    }

    std::string objFileName(ntwk->gName());
    objFileName += ".tpb";
    if (! useWave) {
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
    } else {
        wavecode::WaveCode waveCode(ntwk, arch);
        std::cout << "WaveCodegen: Generating instructions to file '" << objFileName << "'\n";
        FILE* file = fopen(objFileName.c_str(), "wb");
        assert(file && "Cannot open object file");
        wavecode::WaveCode::InstrStreams instrStreams;
        instrStreams.m_StreamProcInstrStream    = file;
        instrStreams.m_PeArrayInstrStream       = file;
        instrStreams.m_PoolEngInstrStream       = file;
        instrStreams.m_ActEngInstrStream        = file;
        waveCode.generate(instrStreams);
    }

    return 0;
}

}

int
main(int argc, char* argv[])
{
    return kcc::Main(argc, argv);
}

