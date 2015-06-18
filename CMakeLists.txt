cmake_minimum_required(VERSION 2.8)
project(fgm-cpp)

if(CMAKE_COMPILER_IS_GNUCC)
  add_definitions("-fPIC -std=c++11")
endif()

if(WIN32 AND NOT CYGWIN)
  set(Boost_USE_STATIC_LIBS On)
  add_definitions(-DBOOST_PYTHON_STATIC_LIB)
endif()

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy;print numpy.get_include()"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python)
include_directories($ENV{EIGEN_ROOT} /usr/include/eigen3)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})

list(FIND PYTHON_LIBRARIES optimized PYTHON_OPTIMIZED_INDEX)
if (NOT ${PYTHON_OPTIMIZED_INDEX} EQUAL -1)
  math(EXPR PYTHON_OPTIMIZED_INDEX "${PYTHON_OPTIMIZED_INDEX}+1")
  list(GET PYTHON_LIBRARIES ${PYTHON_OPTIMIZED_INDEX} PYTHON_RELEASE_LIBRARY)
  link_libraries(${PYTHON_RELEASE_LIBRARY})
endif()

link_libraries(${Boost_LIBRARIES})

add_library(fgm SHARED bindings/python/module.cpp util.cpp util.hpp fgm.cpp fgm.hpp hungarian.cpp hungarian.hpp)

set_target_properties(fgm PROPERTIES PREFIX "")
if(WIN32 AND NOT CYGWIN)
  set_target_properties(fgm PROPERTIES SUFFIX ".pyd")
endif()
