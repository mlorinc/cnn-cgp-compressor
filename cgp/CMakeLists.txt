# Copyright 2024 Marián Lorinc
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     LICENSE.txt file
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# CMakeLists.txt: Building file.
# Add source to this project's executable.
add_executable (CGP 
    "Main.cpp" "Main.h"
    "Cgp.cpp" "Cgp.h"
    "Chromosome.cpp" "Chromosome.h"
    "Configuration.cpp" "Configuration.h"
    "CGPStream.h" "CGPStream.cpp"
    "Stream.cpp" "Stream.h"
    "StringTemplate.h" "StringTemplate.cpp" "Dataset.h" "Dataset.cpp" "Assert.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CGP PROPERTY CXX_STANDARD 20)
endif()

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
target_link_libraries(CGP OpenMP::OpenMP_CXX)
