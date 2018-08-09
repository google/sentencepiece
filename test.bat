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

cd ..\python
call :BuildPython C:\Python27-x64
call :BuildPython C:\Python35-x64
call :BuildPython C:\Python36-x64
call :BuildPython C:\Python37-x64
c:\Python37-x64\python setup.py sdist
exit

:BuildPython
%1\python setup.py build
%1\python setup.py test
%1\python setup.py bdist_wheel
rmdir /Q /S build
del /S *.pyd
exit /b
