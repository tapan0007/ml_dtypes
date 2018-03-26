#!/bin/bash

function is_function_defined()
{
	local ftype
	ftype=$(type -t $1)
	if [ "$ftype" = function ]; then
		return 0
	fi
	return 1
}


#--------| Default function definitions
# We only define these if they don't already exist. This allows
# callers to "override" these functions.

if ! is_function_defined get_package_prefix; then
	function get_package_prefix()
	{
	if [ -n "$REGION_AWARE" ] && [ -n "$REGION" ] ; then
		# include region name in package name
		echo ${PKG_NAME}-${REGION}-${PKG_VERSION}
	else
		echo ${PKG_NAME}-${PKG_VERSION}
	fi
	}
fi

if ! is_function_defined warning; then
     function warning() { 
        echo -e "\033[1;33;40m$*\033[0m"
     }
fi

if ! is_function_defined info; then
     function info() { 
        echo -e "\033[1;34;40m$*\033[0m"
     }
fi

if ! is_function_defined fatal; then
     function fatal() { 
        echo -e "\033[1;31;40m$*\033[0m"
        exit 1;
     }
fi

if ! is_function_defined filter_region_config; then
     #
     # Given a file do a global search and replace in that file for the given variable.
     #
     function filter_region_config()
      {
          local config_file=$1;
          if [ -f "$config_file" ]
          then
              sed "s/__REGION_UC__/${REGION_UC}/g" $config_file -i              # SFO
              sed "s/__REGION__/${REGION}/g" $config_file -i			# sfo
              sed "s/__PUBLIC_REGION__/${PUBLIC_REGION}/g" $config_file -i      # us-east-1
              sed "s/__SERVICES_DOMAIN__/${SERVICES_DOMAIN}/g" $config_file -i	# us-east-1.amazon...
              sed "s/__AWS_DOMAIN__/${AWS_DOMAIN}/g" $config_file -i	        # amazonaws.com
              sed "s/__AWS_S3_DOMAIN__/${AWS_S3_DOMAIN}/g" $config_file -i	# s3...amazonaws.com
          else
              echo -e "\033[1;31;40mNo $config_file found. Proceeding...\033[0m"
          fi
      }
fi

if ! is_function_defined package_source_tarball; then
	# Prep the source tarball for rpmbuild. We need to create a symlink under the
	# build directory so when we tar up our source it unpacks into a directory
	# named as rpmbuild expects.
	function package_source_tarball()
	{
		local prefix
		prefix=$(get_package_prefix)
		
		mkdir -p ${BUILDDIR}
		ln -s ../ ${BUILDDIR}/${prefix}
		tar chzf ${RPMDIR}/SOURCES/${prefix}-src.tar.gz --exclude="${BUILDDIR}" -C ${BUILDDIR} ${prefix}/
		rm ${BUILDDIR}/${prefix}
	}
fi


if ! is_function_defined get_rpmspec_src; then
  # where to get the spec file for the project
  function get_rpmspec_src()
  {
    echo rpmspec/${PKG_NAME}.spec
  }
fi

if ! is_function_defined get_rpmspec_dst; then
  function get_rpmspec_dst()
  {
    echo ${BUILDDIR}/rpmspec/${PKG_NAME}.spec
  }
fi

if ! is_function_defined create_spec; then
	function create_spec()
	{
    local rpmspec_src=$(get_rpmspec_src)
    local rpmspec_dst=$(get_rpmspec_dst)
		mkdir -p `dirname ${rpmspec_dst}`
		rm -f ${rpmspec_dst}
		cp ${rpmspec_src} ${rpmspec_dst}
		sed "s/__RHEL_RELEASE__/${RHEL_RELEASE}/g" ${rpmspec_dst} -i
		sed "s/__PKG_NAME__/${PKG_NAME}/g" ${rpmspec_dst} -i
		sed "s/__PKG_VERSION__/${PKG_VERSION}/g" ${rpmspec_dst} -i
		sed "s/__BRANCH_VERSION__/${BRANCH_VERSION}/g" ${rpmspec_dst} -i
		sed "s/__PKG_RELEASE__/${PKG_RELEASE}/g" ${rpmspec_dst} -i
		sed "s/__PKG_ARCH__/${PKG_ARCH}/g" ${rpmspec_dst} -i
		sed "s/__BUILDDIR__/${BUILDDIR}/g" ${rpmspec_dst} -i
		sed "s/__REGION__/${REGION}/g" ${rpmspec_dst} -i
		sed "s/__REGION_UC__/${REGION_UC}/g" ${rpmspec_dst} -i
        sed "s!__SITE_RUBY__!${SITE_RUBY}!g" ${rpmspec_dst} -i

		# Check for a Package header, and warn if it exists. If it doesn't add one.
		local header=$(grep '^Packager:' ${rpmspec_dst})
		if [ "$header" != "" ] ; then
		    echo =======================================================
		    echo WARNING: Package header is already defined in rpm spec file
		    echo "  $header"
		    echo =======================================================
		else
		    sed -e "/^Name:/ iPackager: ${BRANCH_NAME}:${PROJECT_SET}:${PLATFORM}" -i ${rpmspec_dst}
		fi
	}
fi

if ! is_function_defined build_rpm; then
	function build_rpm()
	{
		local rpmspec_dst=$(get_rpmspec_dst)
		rpmbuild -ba ${rpmspec_dst} --buildroot $RPM_BUILD_RT
		return $?
	}
fi

if ! is_function_defined store_rpms; then
	function store_rpms()
	{
		local prefix
		prefix=$(get_package_prefix)
		
		mkdir -p ${BUILDDIR}/rpm
		cp ${RPMDIR}/RPMS/${PKG_ARCH}/${prefix}-${PKG_RELEASE}.${PKG_ARCH}.rpm ${BUILDDIR}/rpm/.
		cp ${RPMDIR}/SRPMS/${prefix}-${PKG_RELEASE}.src.rpm ${BUILDDIR}/rpm/.
	}
fi

if ! is_function_defined store_artifacts ; then
	function store_artifacts()
	{
		if [ "$PKG_RETRIEVE_ARTIFACTS" = "true" ]; then
			local prefix
			prefix=$(get_package_prefix)
			mkdir -p ${BUILDDIR}/artifacts
			cp -r ${RPMDIR}/BUILD/${prefix}/${BUILDDIR}/* ${BUILDDIR}/artifacts/.
		fi
	}
fi

if ! is_function_defined install_rpms; then
	function install_rpms()
	{
		sudo rpm -Uvh ${BUILDDIR}/rpm/${PKG_NAME}-${PKG_VERSION}-${PKG_RELEASE}.${PKG_ARCH}.rpm
	}
fi

if ! is_function_defined init_region; then
	function init_region()
  {
    # return 0 to build this region, anything else to skip it
    return 0
  }
fi
