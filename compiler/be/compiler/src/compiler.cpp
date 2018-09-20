#include <fstream>
#include <iostream>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>

#include "utils/inc/version.hpp"
#include "utils/inc/asserter.hpp"
#include "utils/inc/misc.hpp"
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
    Assert(file, "Cannot open %s object file: %s", engineName, objFileName.c_str());
    return file;
}

static
void writeOutJson(nets::Network* ntwk, const char* jsonInFileName, const char* ext)
{
    char JsonOutFileName[256];
    strncpy(JsonOutFileName, jsonInFileName,
        kcc::ArraySizeof(JsonOutFileName)-1);
    char* q = JsonOutFileName + (strlen(JsonOutFileName) - 5);
    Assert(0 == strcmp(q, ".json"),
        "Input Json file name does not end in '.json'");
    sprintf(q, "-%s.json", ext);

    std::ofstream os(JsonOutFileName);

    std::cout << "Writing NN JSON to file '" << JsonOutFileName << "'\n";

    try {
        cereal::JSONOutputArchive ar(os);
        ntwk->save(ar);
    } catch (const cereal::Exception& except) {
        std::cerr << "Error <"  << except.what() << "> when writing JSON file '" << JsonOutFileName << "'\n";
        exit(1);
    } catch (...) {
        std::cerr << "Error writing JSON file '" << JsonOutFileName << "'\n";
        exit(1);
    }

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

    kcc_int32 numTpbEvents = -1;

    int i = 1;
    while (i < argc) {
        const std::string arg(argv[i]);
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
        } else if (arg == "--parallel_streams" || arg == "--parallel-streams") {
            ParallelStreams = true;
        } else if (arg == "--sequential_stream" || arg == "--sequential-stream") {
            ParallelStreams = false;
        } else if (arg == "--number-tpb-events") {
            numTpbEvents = atoi(argv[i+1]);
            ++i;
        } else
#endif
        if (arg == "--json") {
            if (JsonInFileName) {
                std::cerr << "Must specify net" << "\n";
                exit(1);
            }
            JsonInFileName = argv[i+1];
            i += 1;
        } else if (arg == "--wavegraph" || arg == "-w") {
            if (JsonInFileName) {
                std::cerr << "Must specify net" << "\n";
                exit(1);
            }
            useWave = true;
            JsonInFileName = argv[i+1];
            i += 1;
        } else {
            std::cerr << "Wrong argument: " << arg << "\n";
            std::cerr << "Usage: " << argv[0]
                      << "[[--parallel_streams] --wavegraph WAVEGRAPH.JSON|--json LAYER.JSON]\n";
            exit(1);
        }

        ++i;
    }

    if (JsonInFileName == nullptr) {
        std::cerr << "Must specify net" << "\n";
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
    if (true) {
        std::cout << "PSUM buffer, bank 0, entry 0: TPB address =  "
                << psumBuf.gEntryTpbAddress(0, 0, dtypeFloat32) << "'\n";
        std::cout << "PSUM buffer, bank 1, entry 0: TPB address =  "
                << psumBuf.gEntryTpbAddress(1, 0, dtypeFloat32) << "'\n";

        std::cout << "Events:\n"
                << "Arch: number all TPB events = "
                << arch::Arch::gArch().gNumberAllTpbEvents() << "\n"

                << "Invalid: " << events::EventId_Invalid() << "\n"
                << "Start inference: " << events::EventId_StartInference() << "\n"
                << "Before PE ifmap read: " << events::EventId_BeforeInputRead_PeArray() << "\n"
                << "Before Act ifmap read: " << events::EventId_BeforeInputRead_ActEng() << "\n"
                << "MatMult multi fanout: " << events::EventId_MMStartMultiSet() << "\n"
                << "Last non-reserved: " << events::EventId_LastNonReserved() << "\n"

                << "EventMgr: Number reserved TPB events (i.e., first non reserved) = "
                << events::EventMgr::gNumberReservedTpbEvents() << "\n"
                ;
        std::cout << "Events: start inference EventId = " << events::EventId_StartInference() << "\n";
    }

#if 0
    const arch::StateBuffer stateBuf(arch.gStateBuffer());
    std::cout << "State buffer, partition size =  " << stateBuf.gPartitionSizeInBytes() << "'\n";
    std::cout << "State buffer, partition 0, entry 0: TPB address =  " << stateBuf.gEntryTpbAddress(0, 0) << "'\n";
    std::cout << "State buffer, partition 1, entry 0: TPB address =  " << stateBuf.gEntryTpbAddress(1, 0) << "'\n";
    //std::cout << "State buffer, All zero TPB address =  " << stateBuf.gAllZeroOffsetTpbAddress() << "'\n";
    std::cout << "State buffer, delta TPB address between part 0 and 1=  "
              << (stateBuf.gEntryTpbAddress(1, 0) - stateBuf.gEntryTpbAddress(0, 0)) << "'\n";
#endif


    //------------------------------------------------
    nets::Network network(arch, utils::Git::gShaLong());
    nets::Network* ntwk = &network;

    {
        std::cout << "Reading NN from JSON file '" << JsonInFileName << "'\n";
        std::ifstream is(JsonInFileName);
        const bool isOpen = is.is_open();
        Assert(isOpen, "JSON input file '", JsonInFileName, "' is not open\n");

        try {
            cereal::JSONInputArchive ar(is);
            ntwk->rUseWave(useWave);
            ntwk->load(ar);
        } catch (const cereal::Exception& except) {
            std::cerr << "Error <"  << except.what() << "> when reading JSON file '" << JsonInFileName << "'\n";
            exit(1);
        } catch (...) {
            std::cerr << "Error reading JSON file '" << JsonInFileName << "'\n";
            exit(1);
        }
    }

    ntwk->rDoBatching(DoBatching);

    assert(useWave);
    if (false && ! useWave) {
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
        writeOutJson(ntwk, JsonInFileName, "lay");
    } else {
        wavecode::WaveCode::InstrStreams instrStreams;
        if (ParallelStreams) {
            //==========================================================
            bool kelf = false;
            std::string objFileName;
            events::EventMgr eventMgr(*ntwk);
            wavecode::WaveCode waveCode(*ntwk, arch);

            // with Angel/Dma
            if (true) {
                instrStreams.m_PeArrayBinFile = objFileName = ntwk->gName() + "-pe.tpb";
                instrStreams.m_PeArrayInstrStream       = openObjectFile(objFileName, "PE array");

                instrStreams.m_StreamProcBinFile = objFileName = ntwk->gName() + "-sp.tpb";
                instrStreams.m_StreamProcInstrStream    = openObjectFile(objFileName, "stream processor");

                instrStreams.m_DmaBinFile = objFileName = ntwk->gName() + "-dma.tpb";
                instrStreams.m_DmaInstrStream    = openObjectFile(objFileName, "DMA");

                instrStreams.m_PoolEngBinFile = objFileName = ntwk->gName() + "-pool.tpb";
                instrStreams.m_PoolEngInstrStream       = openObjectFile(objFileName, "pooling engine");

                instrStreams.m_ActEngBinFile = objFileName = ntwk->gName() + "-act.tpb";
                instrStreams.m_ActEngInstrStream        = openObjectFile(objFileName, "activation engine");

                waveCode.rBinFileType(BinFileType::SimAngel);
                waveCode.DetermineEngines();
                eventMgr.processWaveops(kelf);
                waveCode.generate(instrStreams, ParallelStreams);

                writeOutJson(ntwk, JsonInFileName, "tpb");
                instrStreams.closeAll();
            }

            //==========================================================
            // Runtime Kelf
            if (true) {
                kelf = true;
                instrStreams.m_PeArrayBinFile = objFileName = ntwk->gName() + "-pe.bin";
                instrStreams.m_PeArrayInstrStream       = openObjectFile(objFileName, "PE array KELF");

                instrStreams.m_StreamProcBinFile = objFileName = ntwk->gName() + "-sp.bin";
                instrStreams.m_StreamProcInstrStream    = openObjectFile(objFileName, "stream processor KELF");

                instrStreams.m_DmaInstrStream    = nullptr;

                instrStreams.m_PoolEngBinFile = objFileName = ntwk->gName() + "-pool.bin";
                instrStreams.m_PoolEngInstrStream       = openObjectFile(objFileName, "pooling engine KELF");

                instrStreams.m_ActEngBinFile = objFileName = ntwk->gName() + "-act.bin";
                instrStreams.m_ActEngInstrStream        = openObjectFile(objFileName, "activation engine KELF");

                ntwk->revertSavedWaveops();
                ntwk->ClearEvents();

                waveCode.rBinFileType(BinFileType::RuntimeKelf);
                waveCode.DetermineEngines();
                eventMgr.processWaveops(kelf);
                waveCode.generate(instrStreams, ParallelStreams);

                writeOutJson(ntwk, JsonInFileName, "bin");
                instrStreams.closeAll();
            }

            //==========================================================
            // Sim Kelf
            if (false) {
                kelf = true;

                instrStreams.m_PeArrayBinFile = objFileName = ntwk->gName() + "-pe.kbin";
                instrStreams.m_PeArrayInstrStream       = openObjectFile(objFileName, "PE array KELF");

                instrStreams.m_StreamProcBinFile = objFileName = ntwk->gName() + "-sp.kbin";
                instrStreams.m_StreamProcInstrStream    = openObjectFile(objFileName, "stream processor KELF");

                instrStreams.m_DmaInstrStream    = nullptr;

                instrStreams.m_PoolEngBinFile = objFileName = ntwk->gName() + "-pool.kbin";
                instrStreams.m_PoolEngInstrStream       = openObjectFile(objFileName, "pooling engine KELF");

                instrStreams.m_ActEngBinFile = objFileName = ntwk->gName() + "-act.kbin";
                instrStreams.m_ActEngInstrStream        = openObjectFile(objFileName, "activation engine KELF");

                //

                waveCode.rBinFileType(BinFileType::SimKelf);
                waveCode.generate(instrStreams, ParallelStreams);

                instrStreams.closeAll();
            }

        } else {
            std::string objFileName(ntwk->gName() + ".tpb");
            FILE* file = openObjectFile(objFileName, "all");

            instrStreams.m_StreamProcInstrStream    = file;
            instrStreams.m_PeArrayInstrStream       = file;
            instrStreams.m_PoolEngInstrStream       = file;
            instrStreams.m_ActEngInstrStream        = file;
            instrStreams.m_DmaInstrStream           = file;

            writeOutJson(ntwk, JsonInFileName, "seqtpb");
            wavecode::WaveCode waveCode(*ntwk, arch);
            waveCode.generate(instrStreams, ParallelStreams);

            fclose(file);

            instrStreams.m_StreamProcInstrStream    = nullptr;
            instrStreams.m_PeArrayInstrStream       = nullptr;
            instrStreams.m_PoolEngInstrStream       = nullptr;
            instrStreams.m_ActEngInstrStream        = nullptr;
            instrStreams.m_DmaInstrStream           = nullptr;
        }

    }
    std::cout << "Compiler BE: PASSED\n";

    return 0;
}

} // namespace kcc

int
main(int argc, char* argv[])
{
    return kcc::Main(argc, argv);
}

