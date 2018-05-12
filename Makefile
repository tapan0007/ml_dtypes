SHELL := /bin/bash
ifndef KAENA_PATH
$(error KAENA_PATH is not set)
endif
ifndef ARCH_ISA_PATH
$(error ARCH_ISA_PATH is not set)
endif
ifndef KAENA_RT_PATH
$(error KAENA_RT_PATH is not set)
endif
ifndef INKLING_PATH
$(error INKLING_PATH is not set)
endif
ifndef KAENA_EXT_PATH
$(error KAENA_EXT_PATH is not set)
endif


RPMDIR ?= ${KAENA_PATH}/build/rpmdir/
export RPMDIR
PKG_VERSION ?= 1.0.0
export PKG_VERSION
PKG_RELEASE ?= 1
export PKG_RELEASE
PYTHON=python3
SUBDIRS ?= \
    runtime \
    submodules/Inkling \
    compiler

.PHONY: build
build:
	@for subdir in $(SUBDIRS); do \
		($(MAKE) -j -C $$subdir PREFIX=${PREFIX} DESTDIR=$(DESTDIR) PYTHON=$(PYTHON)); \
	done
	
PREFIX=/tmp
.PHONY: install
install:
	@for subdir in $(SUBDIRS); do \
		($(MAKE) -C $$subdir install PREFIX=${PREFIX} DESTDIR=$(DESTDIR) PYTHON=$(PYTHON));\
	done

.PHONY: uninstall
uninstall:
	@for subdir in $(SUBDIRS); do \
		($(MAKE) -C $$subdir uninstall PREFIX=${PREFIX} DESTDIR=$(DESTDIR));\
	done


check:
	$(MAKE) check -C ${KAENA_PATH}/test/e2e

check_clean:
	$(MAKE) clean -C ${KAENA_PATH}/test/e2e

clean:  build_clean check_clean repo_clean


build_clean:
	@for subdir in $(SUBDIRS); do \
		($(MAKE) -C $$subdir clean  PYTHON=$(PYTHON)) \
	done


packages:
	mkdir -p $(RPMDIR) || exit 1;
	for subdir in $(SUBDIRS); do \
		pushd $$subdir; \
		./package.sh || exit 1; \
		popd; \
	done



REPO_TARBALL ?= kaena-$(PKG_VERSION)-$(PKG_RELEASE)-repo.tar.gz
REPO_TEST_TARBALL ?= $(USER)-test-kaena-$(PKG_VERSION)-$(PKG_RELEASE)-repo.tar.gz



# push a release tarball, $(2), to the upgrade bucket for a given region, $(1)
#   e.g. $(call _publish,iad,blackfoot-3.14.6-1-repo.tar.gz.aes)
#
# TODO: find a better way than hardcoding region names here.. 
define _publish
        exit_code=0; \
	region=`./get_public_region $(1)` || exit_code=$$?; \
        if [[ $$exit_code -eq 0 ]]; then \
		echo "region name is  $(1), public region name is $$region "; \
		aws s3 --profile kaena cp $(2) s3://kaena-release-bucket/$(2) \
                  --region=$$region;\
	fi
endef


# Publish the release tarball to all regions
# Usage: make publish
.PHONY: publish
publish: $(REPO_TARBALL)
	for region in `grep -v '^#' publish_regions | awk '{print $$1}'`; do \
		$(call _publish,$$region,$<); \
done


# Publish the test tarball to iad
# Usage: make publish_test
publish_test: $(REPO_TEST_TARBALL)
	$(call _publish,iad,$<)


.PHONY: create_repo
create_repo: packages
	for subdir in $(ARCHS); do \
	  createrepo -v $(CREATEREPO_OPTS) $(CREATEREPO_EXCLUDES) $(RPMDIR)/RPMS/$${subdir}; \
	done

# really creates tarball of the repo
.PHONY: repo
repo: create_repo
	tar -C $(RPMDIR)/RPMS -czv  -f $(REPO_TARBALL) .

# really creates test tarball
.PHONY: repo_test
repo_test: create_repo
	tar -C $(RPMDIR)/RPMS -czv -f $(REPO_TEST_TARBALL) .

$(REPO_TARBALL): repo
$(REPO_TEST_TARBALL): repo_test


.PHONY: repo-clean
repo_clean:
	rm -fr $(RPMDIR)/*
	rm -f $(REPO_TARBALL)
	rm -f $(REPO_TEST_TARBALL)
