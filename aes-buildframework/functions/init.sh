#!/bin/bash

DEFAULT_RPMDIR=${HOME}/rpmbuild
PREFIX=${PKG_NAME}-${PKG_VERSION}

BRANCH_NAME=${BRANCH_NAME:-$(git symbolic-ref HEAD 2>/dev/null | awk -F\/ '{print $NF}')}
PROJECT_SET=${PROJECT_SET:-manual}
PLATFORM=${PLATFORM:-$(uname -m | sed -e 's/i.86/i386/')}

# EC2 regions for per-region builds
if [ -n "$REGIONS" ] ; then
	echo =======================================================
	echo WARNING: REGIONS has been overridden
	echo =======================================================
else
	export REGIONS="nrt sin sfo dub local iad gru pdt pek pdx syd sea sea3lab $CUSTOM_REGIONS"
fi

# Determine the architecture of the build environment
# and set the base architecture accordingly.  The latter
# is used to determine from which of the 32- or 64-bit
# repositories the build artefacts will be served.
BUILD_ARCH=`uname -m`
case `uname -m` in
  i[3-9]86)
    BUILD_BASE_ARCH=i386
    ;;
  x86_64)
    BUILD_BASE_ARCH=x86_64
    ;;
  *)
    fatal "Unrecognised architecture: `uname -m`"
    ;;
esac
export BUILD_ARCH BUILD_BASE_ARCH

if [ -z "$PKG_ARCH" ] ; then
	PKG_ARCH=noarch
fi

if [ -n "$BRANCH_VERSION" ] ; then
	echo =======================================================
	echo WARNING: BRANCH_VERSION has been overridden
	echo =======================================================
else
	export BRANCH_VERSION=$(cat ${BUILDFRAMEWORKDIR:-.}/../version)
fi

if [ -z "$PKG_VERSION" ] ; then
	echo =======================================================
	echo ERROR: PKG_VERSION unset
	echo =======================================================
    exit 1
fi

if [ -z "$PKG_RELEASE" ] ; then
	echo =======================================================
	echo ERROR: PKG_RELEASE unset
	echo =======================================================
    exit 1
fi

if [ -z "$BUILDDIR" ] ; then
	BUILDDIR=build
fi

if [ -n "$SITE_RUBY" ] ; then
	echo =======================================================
	echo WARNING: SITE_RUBY has been overridden
	echo =======================================================
else
    export SITE_RUBY=/usr/lib/ruby/site_ruby
fi
