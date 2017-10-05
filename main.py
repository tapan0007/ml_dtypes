import sys

from nets.densenet.densenet  import DenseNet169
from nets.resnet.resnet    import ResNet50
from schedule.scheduler      import Scheduler

print sys.argv
assert(len(sys.argv) == 2)

if sys.argv[1] == "--densenet" or sys.argv[1] == "--dense":
    ntwk = DenseNet169()
elif sys.argv[1] == "--resnet" or sys.argv[1] == "--res":
    ntwk = ResNet50()
else:
    sys.exit(1)

##################################################
ntwk.construct()
scheduler = Scheduler()
scheduler.schedule(ntwk)
ntwk.rLevels(scheduler.gLevels())

### Printing
ntwk.printMe()
ntwk.printLevels()
ntwk.printSched()

ntwk.printDot()


