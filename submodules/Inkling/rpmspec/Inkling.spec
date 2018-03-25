Summary: Inkling simulator
Name: __PKG_NAME__
Version: __PKG_VERSION__
Release: __PKG_RELEASE__
BuildArch: __PKG_ARCH__
License: Proprietary
Vendor: Amazon
Group: AWS/Tonga/Kaena
Source: %{name}-%{version}-src.tar.gz
Provides: %{name} = %{version}-%{release}

BuildRoot: %{_tmppath}/%{name}-root


%description
Inkling simulator

%prep
%setup -q -n %{name}-%{version}

%build

%install
rm -rf %{buildroot}
mkdir -p ${RPM_BUILD_ROOT}
#mkdir -p ${RPM_BUILD_ROOT}/usr/local/kaena
make install PREFIX=${RPM_BUILD_ROOT}/usr

%clean
rm -rf %{buildroot}
[ ${RPM_BUILD_ROOT} != "/" ] && rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root)
#/usr/local/kaena/
/usr/bin/sim
