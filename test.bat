set PROTOBUF_VERSION=3.6.1
set ARC=x64
set _CL_=/utf-8
set PATH=c:\Program Files\Git\usr\bin;c:\MinGW\bin;%PATH%
set CURRENT_PATH=%~dp0
set LIBRARY_PATH=%CURRENT_PATH%build\root\%ARC%

mkdir build
cd build

curl -O -L https://github.com/google/protobuf/releases/download/v%PROTOBUF_VERSION%/protobuf-cpp-%PROTOBUF_VERSION%.zip
unzip protobuf-cpp-%PROTOBUF_VERSION%.zip
cd protobuf-%PROTOBUF_VERSION%\cmake
cmake . -A %ARC% -DCMAKE_INSTALL_PREFIX=%LIBRARY_PATH%
cmake --build . --config Release --target install

cd ..\..
cmake .. -A %ARC% -DSPM_BUILD_TEST=ON -DSPM_ENABLE_SHARED=OFF -DCMAKE_INSTALL_PREFIX=%LIBRARY_PATH%
cmake --build . --config Release
ctest -C Release
cmake --build . --config Release --target install