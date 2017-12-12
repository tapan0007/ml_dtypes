#include <fstream>
#include <iostream>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>

#include "debug.hpp"
#include "arch.hpp"
#include "statebufmgr.hpp"
#include "codegen.hpp"
#include "scheduler.hpp"
#include "layer.hpp"
#include "network.hpp"

//#include "printer.hpp"



using kcc::arch::Arch;
using kcc::arch::PeArray;
using kcc::arch::PsumBuffer;
using kcc::arch::PoolingEng;
using kcc::arch::ActivationEng;
using kcc::arch::StateBuffer;

using kcc::nets::Network;
using kcc::layers::Layer;
using kcc::schedule::Scheduler;
using kcc::memmgr::StateBufferMgr;
using kcc::codegen::CodeGen;



//------------------------------------------------

int
main(int argc, char* argv[])
{
    kcc::utils::breakFunc(44);
#if 0
    bool PrintLevels   = false;
    bool PrintSchedule = false;
    bool PrintDot      = false;
    bool PrintLayers   = false;
    bool TrivNet       = false;
    bool UseRelu       = false;
#endif
    bool DoBatching    = false;
    const char* JsonInFileName = nullptr;

    int i = 1;
    while (i < argc) {
        string arg(argv[i]);
#if 0
        if (arg == "--trivnet" or arg == "--triv") {
            TrivNet = true;
        } else if (arg == "--print-layers") {
            PrintLayers = true;
        } else if (arg == "--print-levels") {
            PrintLevels = true;
        } else if (arg == "--print-sched") {
            PrintSchedule = true;
        } else if (arg == "--no-print-sched") {
            PrintSchedule = false;
        } else if (arg == "--print-dot") {
            PrintDot = true;
        } else if (arg == "--relu") {
            UseRelu = true;
        } else if (arg == "--batch" or arg == "--batching") {
            DoBatching = true;
        } else
#endif
        if (arg == "--json") {
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
    Network network;
    Network* ntwk = &network;
    kcc::utils::breakFunc(44);
    {
        std::ifstream is(JsonInFileName);
        cereal::JSONInputArchive ar(is);
        ntwk->load(ar);
    }

    //------------------------------------------------
    std::cout << "Generating Arch\n";
    Arch* arch = new Arch();

    PeArray* peArray = arch->gPeArray();
    assert(peArray);
    PsumBuffer* psumBuf = arch->gPsumBuffer();
    assert(psumBuf);
    PoolingEng* pool = arch->gPoolingEng();
    assert(pool);
    ActivationEng* activ = arch->gActivationEng();
    assert(activ);
    StateBuffer* stbuf = arch->gStateBuffer();
    assert(stbuf);
    arch->gNumberPsumBanks();
    arch->gPsumBankEntries();
    arch->gNumberPeArrayRows();
    arch->gNumberPeArrayColumns();


    ntwk->rDoBatching(DoBatching);

    //--------------------------------------------------------
    Scheduler* scheduler = new Scheduler();
    std::cout << "Scheduling\n";
    scheduler->Schedule(ntwk);
    //ntwk->rLevels(scheduler->gLevels());

    //--------------------------------------------------------
    StateBufferMgr* sbmgr = new StateBufferMgr(arch, ntwk);
    std::cout << "Calculating FMAP addresses\n";
    sbmgr->calcLayerFmapAddresses();

    //--------------------------------------------------------
    {
        std::cout << "Reading NN\n";
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
        cereal::JSONOutputArchive ar(os);
        ntwk->save(ar);
    }

    //--------------------------------------------------------
    CodeGen* codegen = new CodeGen(ntwk, arch);
    std::cout << "Generating instructions";
    string objFileName(ntwk->gName());
    objFileName += ".tpb";
    codegen->generate(objFileName.c_str());

#if 0
    //--------------------------------------------------------
    printer = new Printer(ntwk);

    if (PrintLayers) {
        printer->printNetwork();
        std::cout << "\n";
    }

    if (PrintLevels) {
        std::cout << "By level\n";
        printer->printLevels();
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

    jsonFileName = ntwk.gName().lower() + ".json"
    printer.printJson(ntwk, jsonFileName)

    with open(jsonFileName) as f:
        jsonContent = f.read()

    jsonDict = json.loads(jsonContent)
    nn2 = Network.constructFromJson(jsonDict)

    jsonFileName2 = nn2.gName().lower() + "2.json"
    printer.printJson(nn2, jsonFileName2)
#endif

    return 0;
}

