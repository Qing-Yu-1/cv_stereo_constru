# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(opencv_video_project)
set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
#opencv
set(OpenCV_DIR /usr/opencv34/share/OpenCV)
# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#eigen
include_directories( "/usr/include/eigen3/" )

# pcl
find_package( PCL 1.8  REQUIRED COMPONENT common io )
#message(STATUS "pcl 111    libraries: ${PCL_LIBRARIES}")
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )
link_directories(${PCL_LIBRARY_DIRS})

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Declare the executable target built from your sources
add_executable(stereo_point_cloud_local_v1 10-28-stereo_piont_cloud_local.cpp)

# Link your application with OpenCV libraries
target_link_libraries(stereo_point_cloud_local_v1 ${OpenCV_LIBS} ${PCL_LIBRARIES})
