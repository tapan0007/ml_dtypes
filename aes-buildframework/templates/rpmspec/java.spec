Summary: Sample spec file for a Java project.
Name: __PKG_NAME__
Version: __PKG_VERSION__
Release: __PKG_RELEASE__
BuildArchitectures: __PKG_ARCH__
License: Proprietary
Vendor: Amazon Execution Service __BRANCH_REV__.__PROJECT_REV__
Group: System Environment/Daemons
Source: %{name}-%{version}-src.tar.gz
Prereq: fileutils /sbin/chkconfig /etc/init.d
BuildRequires: java-sdk >= 1.5.0, log4j >= 1.2.12
Requires: java >= 1.5, log4j >= 1.2.12
Provides: %{name} = %{version}-%{release}
BuildRoot: %{_tmppath}/%{name}-%{version}-root

%description
blah blah blah

%prep
%setup -q -n %{name}-%{version}

%build
PKG_NAME=__PKG_NAME__ \
PKG_RELEASE=__PKG_RELEASE__ \
PKG_ARCH=__PKG_ARCH__ \
BUILDDIR=__BUILDDIR__ \
./build.sh

%install
PKG_NAME=__PKG_NAME__ \
PKG_RELEASE=__PKG_RELEASE__ \
PKG_ARCH=__PKG_ARCH__ \
BUILDDIR=__BUILDDIR__ \
./prepbuildroot.sh

%check

%clean
rm -rf $RPM_BUILD_ROOT

%pre

%post

%preun

%postun

%files
%defattr(-,root,root)
/usr/share/java/%{name}-%{version}.jar
/usr/share/java/%{name}.jar

%changelog
* Tue Nov 09 2005 James Greenfield <jamesg@amazon.com> 1.4.1rh-30
- Describe the magic you worked here.
