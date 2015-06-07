@echo off

set name=fgm

cd bindings\python

if exist %name%.pyd del %name%.pyd

call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat"

::initial includes and libpaths
set includes=/I %~dp0
set libpaths=

::set environment for eigen
set includes=%includes% /I %EIGEN_ROOT%

::set environment for numpy
set includes=%includes% /I C:\python27\lib\site-packages\numpy\core\include

::set environment for boost
set includes=%includes% /I %BOOST_ROOT%
set libpaths=%libpaths% /LIBPATH:"%BOOST_ROOT%\stage\lib"

::set environment for python extension
set includes=%includes% /I C:\python27\include
set libpaths=%libpaths% /LIBPATH:"C:\python27\libs"

set cmdline=cl /LDd /EHsc /DBOOST_PYTHON_STATIC_LIB /O2 /DNDEBUG %includes% module.cpp /link %libpaths% /out:%name%.pyd

echo %cmdline%

%cmdline%

python test_%name%.py -v

pause