#!/bin/bash

#--------| Source helper routines
source ${BASH_SOURCE[0]/install.sh/functions.sh}

# -------| Initialization
source ${BASH_SOURCE[0]/install.sh/init.sh}

# -------| Main

# Install the built RPM
install_rpms
