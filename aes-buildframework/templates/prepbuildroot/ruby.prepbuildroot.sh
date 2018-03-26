#!/bin/sh

# All libraries for this project
mkdir -p ${RPM_BUILD_ROOT}${SITE_RUBY}
rsync -avz --cvs-exclude src/lib/* ${RPM_BUILD_ROOT}${SITE_RUBY}.

# All scripts for this project
mkdir -p ${RPM_BUILD_ROOT}/usr/local/aes/${PKG_NAME}
cp -r src/bin/* ${RPM_BUILD_ROOT}/usr/local/aes/${PKG_NAME}

# Configuration files for this project
mkdir -p ${RPM_BUILD_ROOT}/etc/aes/${PKG_NAME}
cp -r resources/config/* ${RPM_BUILD_ROOT}/etc/aes/${PKG_NAME}

# Logrotate directives (if required)
mkdir -p ${RPM_BUILD_ROOT}/etc/logrotate.d
cp -r resources/logrotate.d/${PKG_NAME} ${RPM_BUILD_ROOT}/etc/logrotate.d/${PKG_NAME}

# Service scripts (if required)
/etc/rc.d/init.d/%{name}
mkdir -p ${RPM_BUILD_ROOT}/etc/rc.d/init.d/
cp -r resources/init.d/${PKG_NAME} ${RPM_BUILD_ROOT}/etc/rc.d/init.d/${PKG_NAME}
