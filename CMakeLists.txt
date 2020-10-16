cmake_minimum_required(VERSION 2.8.9)
project(ICP)
set (CMAKE_CXX_STANDARD 17) # c++17

##############
#   LIBCSV   #
##############

include_directories(include)
file(GLOB LIBCSVSOURCES "src/libCSV/*.cpp")
if(CMAKE_COMPILER_IS_GNUCXX)
 add_compile_options(-Wall -Wextra -pedantic -Werror -g)
endif(CMAKE_COMPILER_IS_GNUCXX)
add_library(CSV ${LIBCSVSOURCES})

##############
#   LIBALG   #
##############

find_package(LAPACK REQUIRED) # sudo apt install liblapack-dev

include_directories(include)
file(GLOB LIBALGSOURCES "src/libalg/*.cpp")
if(CMAKE_COMPILER_IS_GNUCXX)
 add_compile_options(-Wall -Wextra -pedantic -Werror -g -lm)
endif(CMAKE_COMPILER_IS_GNUCXX)
add_library(alg ${LIBALGSOURCES})
target_link_libraries(alg lapack) # libalg uses lapack lib for SVD computation

##############
# TESTLIBALG #
##############

include_directories(include)
file(GLOB TESTLIBALGSOURCES "src/testlibalg/*.cpp")
if(CMAKE_COMPILER_IS_GNUCXX)
add_compile_options(-Wall -Wextra -pedantic -Werror -g)
endif(CMAKE_COMPILER_IS_GNUCXX)

foreach(file_path ${TESTLIBALGSOURCES})
 string(REPLACE ".cpp" "" bin_path_name ${file_path} )
 get_filename_component(bin_name ${bin_path_name} NAME)
 add_executable(${bin_name} ${file_path})
 target_link_libraries(${bin_name} CSV alg)
endforeach()

###########
#   CPU   #
###########

include_directories(include)
file(GLOB CPUSOURCES "src/cpu/*.cpp")
if(CMAKE_COMPILER_IS_GNUCXX)
 add_compile_options(-Wall -Wextra -pedantic -Werror -g)
endif(CMAKE_COMPILER_IS_GNUCXX)
add_executable(CPUICP ${CPUSOURCES})
target_link_libraries(CPUICP CSV alg)