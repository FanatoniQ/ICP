cmake_minimum_required(VERSION 2.8.9)
project(ICP)
set (CMAKE_CXX_STANDARD 17) # c++17

##############
#   LIBCSV   #
##############

include_directories(include)
file(GLOB LIBCSVSOURCES "src/libCSV/*.cpp")
add_compile_options(-Wall -Wextra -pedantic -Werror -g)
add_library(CSV ${LIBCSVSOURCES})

##############
#   LIBALG   #
##############

include_directories(include)
file(GLOB LIBALGSOURCES "src/libalg/*.cpp")
add_compile_options(-Wall -Wextra -pedantic -Werror -g -lm)
add_library(alg ${LIBALGSOURCES})

###########
#   CPU   #
###########

include_directories(include)
file(GLOB CPUSOURCES "src/cpu/*.cpp")
add_compile_options(-Wall -Wextra -pedantic -Werror -g)
add_executable(CPUICP ${CPUSOURCES})
target_link_libraries(CPUICP CSV alg)