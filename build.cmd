@echo off

set name=fgm

if exist %name%.pyd del %name%.pyd

call "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\vcvarsall.bat"

::initial includes and libs
set includes=
set libpaths=
set libs=

::set environment for eigen
set includes=%includes% /I %EIGEN_ROOT%

::set environment for numpy
set includes=%includes% /I C:\Python27\lib\site-packages\numpy\core\include

::set environment for boost
set includes=%includes% /I %BOOST_ROOT%
set libpaths=%libpaths% /LIBPATH:"%BOOST_ROOT%\stage\lib"

::set environment for python extension
set includes=%includes% /I C:\Python27\include
set libpaths=%libpaths% /LIBPATH:"C:\Python27\libs"

set cmdline=cl /LDd /EHsc /DBOOST_PYTHON_STATIC_LIB %includes% %name%.cpp /link %libpaths% /out:%name%.pyd

echo %cmdline%

%cmdline%

python test_%name%.py -v

pause