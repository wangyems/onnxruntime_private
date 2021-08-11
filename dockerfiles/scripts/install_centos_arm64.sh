yum-config-manager --enable extras
yum -y install centos-release-scl-rh
# EPEL support (for yasm)
if ! rpm -q --quiet epel-release ; then
  yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
fi
yum install -y devtoolset-10-binutils devtoolset-10-gcc devtoolset-10-gcc-c++ devtoolset-10-gcc aria2 python3-pip python3-wheel git python3-devel
aria2c -q -d /tmp -o cmake-3.21.1-linux-aarch64.tar.gz https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1-linux-aarch64.tar.gz && tar -zxf /tmp/cmake-3.21.1-linux-aarch64.tar.gz --strip=1 -C /usr
python3 -m pip install --upgrade pip
python3 -m pip install numpy
