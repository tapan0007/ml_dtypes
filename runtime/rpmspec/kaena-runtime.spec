Summary: Kaena runtime
Name: __PKG_NAME__
Version: __PKG_VERSION__
Release: __PKG_RELEASE__
BuildArch: __PKG_ARCH__
License: Proprietary
Vendor: Amazon
Group: AWS/Tonga/Kaena
Source: %{name}-%{version}-src.tar.gz
Provides: %{name} = %{version}-%{release}
Requires: python36

BuildRoot: %{_tmppath}/%{name}-root

%global __python python3.6

%description
Kaena runtime

%prep
%setup -q -n %{name}-%{version}

%build
make build PYTHON=%{__python}

%install
rm -rf %{buildroot}
mkdir -p ${RPM_BUILD_ROOT}
echo %{__python}
make install PREFIX=${RPM_BUILD_ROOT}/usr PYTHON=%{__python}
cd %{buildroot}; find usr/lib/python3.6/ -type f > file-list; sed -i 's/^/\//' file-list; cd -; mv %{buildroot}/file-list ./


%clean
rm -rf %{buildroot}
[ ${RPM_BUILD_ROOT} != "/" ] && rm -rf $RPM_BUILD_ROOT

%files -f file-list
%defattr(-,root,root)
/usr/bin/runtime_tf
/usr/bin/runtime_sim
/usr/bin/nn_executor
