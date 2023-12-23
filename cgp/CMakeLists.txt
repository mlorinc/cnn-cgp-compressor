# CMakeList.txt : CMake project for cgp, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (CGP 
    "Main.cpp" "Main.h"
    #"Model.cpp" "Model.h"
    "Cgp.cpp" "Cgp.h"
    "Chromosome.cpp" "Chromosome.h"
    "Configuration.cpp" "Configuration.h"
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CGP PROPERTY CXX_STANDARD 20)
endif()

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(CGP "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET CGP
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:CGP>)
endif (MSVC)