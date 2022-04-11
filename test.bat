set PLATFORM=%1
if "%PLATFORM%"=="" set PLATFORM=x64
set PLATFORM_PREFIX=
if "%PLATFORM%"=="x64" set PLATFORM_PREFIX=-x64
set _CL_=/utf-8
set PATH=c:\Program Files\Git\usr\bin;c:\MinGW\bin;%PATH%
set CURRENT_PATH=%~dp0
set LIBRARY_PATH=%CURRENT_PATH%build\root

mkdir build
cd build

cmake .. -A %PLATFORM% -DSPM_BUILD_TEST=ON -DSPM_ENABLE_SHARED=OFF -DCMAKE_INSTALL_PREFIX=%LIBRARY_PATH%
cmake --build . --config Release --target install || goto :error
ctest -C Release || goto :error
cpack || goto :error

cd ..\python
rem call :BuildPython C:\Python27%PLATFORM_PREFIX%
call :BuildPython C:\Python35%PLATFORM_PREFIX%
call :BuildPython C:\Python36%PLATFORM_PREFIX%
call :BuildPython C:\Python37%PLATFORM_PREFIX%
call :BuildPython C:\Python38%PLATFORM_PREFIX%
call :BuildPython C:\Python39%PLATFORM_PREFIX%
call :BuildPython C:\Python310%PLATFORM_PREFIX%
c:\Python38%PLATFORM_PREFIX%\python setup.py sdist || goto :error
exit

:BuildPython
%1\python -m pip install wheel || goto :error
%1\python setup.py build || goto :error
%1\python setup.py bdist_wheel || goto :error
%1\python setup.py test || goto :error
rmdir /Q /S build
del /S *.pyd
exit /b

:error
exit /b %errorlevel%
