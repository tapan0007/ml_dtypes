import sys
assert(sys.version_info.major >= 3)
import json

##################################################
from utils.printers             import Printer
from utils.debug                import breakFunc
from utils.consts               import *
from utils.datatype             import *

import layers.layer
from arch.arch import Arch
from memmgr.statebufmgr         import StateBufferMgr
from codegen.macroinstrgen      import MacroInstrGen

#from layers.layer               import rDoBatching

from schedule.scheduler         import Scheduler

##################################################
#print(sys.argv)

PrintLevels = False
PrintSchedule = False
PrintDot = False
PrintLayers = False
__DoBatching = False
DenseNet = False
ResNet = False
TrivNet = False
NetPyFile = None
UseRelu = False
JsonFileName = None

argv = sys.argv
nArgv = len(argv)
i = 1
while i < nArgv:
    arg = argv[i]
    if arg == "--densenet" or arg == "--dense":
        DenseNet = True
    elif arg == "--resnet" or arg == "--res":
        ResNet = True
    elif arg == "--trivnet" or arg == "--triv":
        TrivNet = True
    elif arg == "--print-layers":
        PrintLayers = True
    elif arg == "--print-levels":
        PrintLevels = True
    elif arg == "--no-print-sched":
        PrintSchedule = False
    elif arg == "--print-dot":
        PrintDot = True
    elif arg == "--batch" or arg == "--batching":
        __DoBatching = True
    elif arg == "--relu":
        UseRelu = True
    elif arg == "--json":
        JsonFileName = argv[i+1]
        i += 1
    else:
        sys.stderr.write("Wrong argument: " + arg + "\n")
        sys.exit(1)

    i += 1

from nets.network         import Network
if DenseNet:
    from nets.densenet.densenet     import DenseNet169
    ntwk = DenseNet169(DataTypeFloat16(), UseRelu)
elif ResNet:
    from nets.resnet.resnet         import ResNet50
    #ntwk = ResNet50(DataTypeFloat16(), UseRelu)
    ntwk = ResNet50(DataTypeInt8(), UseRelu)
    #ntwk = ResNet50(DataTypeInt16(), UseRelu)
elif TrivNet:
    #from nets.trivnet.trivnet_compiler import TrivNet
    from trivnet_compiler import TrivNet
    ntwk = TrivNet()
elif JsonFileName:
    with open(JsonFileName) as f:
        jsonContent = f.read()
    jsonDict = json.loads(jsonContent)
    ntwk = Network.constructFromJson(jsonDict)
else:
    sys.stderr.write("Must specify net\n")
    sys.exit(1)

##################################################
arch = Arch()

peArray = arch.gPeArray()
psumBuf = arch.gPsumBuffer()
pool = arch.gPoolingEng()
activ = arch.gActivationEng()
stbuf = arch.gStateBuffer()
arch.gNumberPsumBanks()
arch.gPsumBankEntries()
arch.gNumberPeArrayRows()
arch.gNumberPeArrayColumns()


##################################################
assert(ntwk)

ntwk.rDoBatching(__DoBatching)
if not ntwk.qConstructed():
    ntwk.construct()

scheduler = Scheduler()
scheduler.Schedule(ntwk)
ntwk.rLevels(scheduler.gLevels())

sbmgr = StateBufferMgr(arch, ntwk)
sbmgr.calcLayerFmapAddresses()

codegen = MacroInstrGen(ntwk, arch)
breakFunc(3)
codegen.generate(ntwk.gName().lower() + ".cpp")
print("")

##################################################
printer = Printer(ntwk)

### Printing
if PrintLayers:
    printer.printNetwork()
    print("")

if PrintLevels:
    print("By level")
    printer.printLevels()
    print("")

if PrintSchedule:
    print("By scheduling")
    printer.printSched()
    print("")

if PrintDot:
    print("Dot")
    printer.printDot()
    print("")

jsonFileName = ntwk.gName().lower() + ".json"
printer.printJson(ntwk, jsonFileName)

with open(jsonFileName) as f:
    jsonContent = f.read()

jsonDict = json.loads(jsonContent)
nn2 = Network.constructFromJson(jsonDict)

jsonFileName2 = nn2.gName().lower() + "2.json"
printer.printJson(nn2, jsonFileName2)

