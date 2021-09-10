# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# License); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Copyright (c) 2021, OPEN AI LAB
# Author: fhfang@openailab.com
#

# Check CMake version
CMAKE_MINIMUM_REQUIRED (VERSION 3.10 FATAL_ERROR)

project(demo_run_graph_main)

set(CMAKE_CXX_STANDARD 17)

OPTION (BUILD_ON_AARCH64             "Build example on aarch64 platform"          OFF)
OPTION (BUILD_ON_A311D               "Build example on a311d platform"            OFF)
OPTION (USE_OPENCV4                  "Opencv version >= 4.0"            OFF)

# macro for adding test
FUNCTION (PIPE_EXAMPLE name file)
    ADD_EXECUTABLE (${name} "${CMAKE_CURRENT_SOURCE_DIR}/${file}")

	set_property(TARGET ${name} PROPERTY CXX_STANDARD 17)
    TARGET_COMPILE_FEATURES(${name} PRIVATE cxx_std_17)

    add_subdirectory(com_google_absl)

    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tengine/include")

    IF(USE_OPENCV4)
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "/usr/include/opencv4")
    ENDIF()

    TARGET_LINK_LIBRARIES (${name} PRIVATE desktop_tengine_calculators)
    TARGET_LINK_LIBRARIES (${name} PRIVATE opencv_highgui opencv_imgproc opencv_core opencv_videoio opencv_imgcodecs)
    TARGET_LINK_LIBRARIES (${name} PRIVATE absl::flags_parse absl::flags_internal absl::flags absl::strings)

    INSTALL (TARGETS ${name} DESTINATION bin)
ENDFUNCTION()

link_directories("${CMAKE_CURRENT_SOURCE_DIR}/libs")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/tengine/lib")
link_directories("/usr/lib")
    
IF(BUILD_ON_AARCH64)
    link_directories("/usr/lib/aarch64-linux-gnu")
ENDIF()

IF(BUILD_ON_A311D)
    link_directories("${CMAKE_CURRENT_SOURCE_DIR}/tengine/tim-vx/aarch64/lib")
ENDIF()

# add examples
PIPE_EXAMPLE (demo_run_detect_main   demo_run_detect_main.cc)

