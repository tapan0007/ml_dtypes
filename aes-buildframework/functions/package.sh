#!/bin/bash
#--------| Source helper routines
source ${BASH_SOURCE[0]/package.sh/functions.sh}

# -------| Initialization
source ${BASH_SOURCE[0]/package.sh/init.sh}

# -------| Validation

# Validate RPMDIR is set
if [ -z "$RPMDIR" ] ; then
	info RPMDIR is not set, defaulting to ${DEFAULT_RPMDIR}
	RPMDIR=${DEFAULT_RPMDIR}
fi

# Validate RPMDIR is a directory
# Don't create it, in case it was a mistake.
if [ ! -d "$RPMDIR" ] ; then
	fatal ${RPMDIR} is not a directory or does not exist, exiting
fi

# Validate RPMDIR is writeable
if [ ! -w "$RPMDIR" ] ; then
	fatal "${RPMDIR} is not writeable, exiting. Set RPMDIR to a folder which is writeable"
fi

#
# Ensure that the folders needed for rpm builds exist
#
for d in BUILD RPMS SOURCES SRPMS SPECS tmp; do
   workingDir=${RPMDIR}/$d
   if [ ! -d "$workingDir" ] ; then
      warning "Mising directory. Creating $workingDir"
      mkdir $workingDir
   fi
   if [ ! -w "$workingDir" ]; then 
      fatal "$workingDir  is not writeable, exiting"
   fi
done

#
# RPM buils use a macro. Check for the macro file and copy it across
# Help setup first time builds - if fails, ignore.
if [ ! -f "$HOME/.rpmmacros" ]; then
   info "Setting up a default rpmmacros file in ~/.rpmmacros"
   workingDir=`dirname ${BASH_SOURCE[0]}`
   cp $workingDir/../resources/rpmmacros $HOME/.rpmmacros
fi

# Validate PKG_NAME and PKG_VERSION are set
for pkg_var in PKG_NAME PKG_VERSION ; do
	pkg_var_val=${!pkg_var}
	if [ -z "${pkg_var_val}" ] ; then
		echo ${pkg_var} is not set, exiting
		return 2
	fi
done


function run_package()
{

  # Package up a source .tar.gz for rpmbuild
  package_source_tarball

  # Create the RPM spec file
  create_spec

  # Optionally tweak it
  if is_function_defined tweak_spec; then
    tweak_spec
  fi

  # Invoke rpmbuild and check it succeeds
  build_rpm
  RETVAL=$?
  if [ $RETVAL -ne 0 ] ; then
    return $RETVAL
  fi

  # Success, retrieve the built RPMS and store them
  # under the build directory
  store_rpms

  # Retrieve build artifacts. Artifacts get retrieved only when
  # the $PKG_RETRIEVE_ARTIFACTS project variable is set to "true".
  # This variable should be set in the project file and is useful
  # for when one needs to do post-processing of build artifacts:
  # for instance, building a zip equivalent of the content of the
  # rpm
  store_artifacts
}

# -------| Main

if [ -n "$REGION_AWARE" ]; then
  info "* Region-aware package. Possible build regions: ${REGIONS}"

  # build region-specific packages for each region
  for REGION in $REGIONS; do
    export REGION
    export REGION_UC=`echo $REGION | tr a-z A-Z`

    info "* Preparing region: ${REGION}"
   
    #
    # SETUP REGION SPECIFIC MACROS
    # REGION is the airport code
    # PUBLIC_REGION is the region as viewed by user
    #
    case $REGION in
       nrt) PUBLIC_REGION="ap-northeast-1";;
       sin) PUBLIC_REGION="ap-southeast-1";;
       sfo) PUBLIC_REGION="us-west-1";;
       iad) PUBLIC_REGION="us-east-1";;
       dub) PUBLIC_REGION="eu-west-1";;
       gru) PUBLIC_REGION="sa-east-1";;
       pek) PUBLIC_REGION="ap-southeast-1";;
       pdt) PUBLIC_REGION="us-gov-west-1";;
       pdx) PUBLIC_REGION="us-west-2";;
       sea) PUBLIC_REGION="us-west-2";;
       syd) PUBLIC_REGION="ap-southeast-2";;
       sea3lab) PUBLIC_REGION="us-east-1";;
       local) PUBLIC_REGION="localhost";;
       *)  
          if is_function_defined init_region_env; then
             init_region_env $REGION
          else
             fatal "Unknown region - ${REGION}. Please provide an init_region_env function (in package.sh) to setup the custom region information"
	  fi
        ;;
    esac

    #
    # Setup the region based configs. Can be overloaded
    # in the init_region_env function
    #
    AWS_DOMAIN=${AWS_DOMAIN:-"amazonaws.com"}
    SERVICES_DOMAIN=${SERVICES_DOMAIN:-"$PUBLIC_REGION.ec2-services.$AWS_DOMAIN"}

    case $REGION in
       dub) AWS_S3_DOMAIN=${AWS_S3_DOMAIN:-"s3-external-3.$AWS_DOMAIN"};;
       iad) AWS_S3_DOMAIN=${AWS_S3_DOMAIN:-"s3.$AWS_DOMAIN"};;
       vpc) AWS_S3_DOMAIN=${AWS_S3_DOMAIN:-"s3.$AWS_DOMAIN"};;
       *)   AWS_S3_DOMAIN=${AWS_S3_DOMAIN:-"s3-$PUBLIC_REGION.$AWS_DOMAIN"};;
    esac

    RPM_BUILD_RT=$RPMDIR/tmp/$PKG_NAME-$REGION-$PKG_VERSION
    rm -rf $RPM_BUILD_RT

    if init_region ; then
      # Package has not opt'ed out of this region. Check the public region 
      # was set correctly. Its a late evaluation as it may be used in init_region
      if [ -z "$PUBLIC_REGION" ] ; then
	fatal "PUBLIC_REGION has not been defined for region ${REGION}. Perhaps you forgot to do this in your init_region_env?"
      fi

      info "* Building region: ${REGION}"
      run_package

      RETVAL=$?
      if [ $RETVAL -ne 0 ] ; then
        break
      fi
    fi

    # remove previous config from next loop context
    unset PUBLIC_REGION AWS_DOMAIN SERVICES_DOMAIN AWS_S3_DOMAIN
  done
else
  # not region aware
  RPM_BUILD_RT=$RPMDIR/tmp/$PKG_NAME-$PKG_VERSION
  rm -rf $RPM_BUILD_RT
  run_package
  RETVAL=$?
fi

return $RETVAL
