cmake_minimum_required(VERSION 2.8.9)
project(ICP)
set (CMAKE_CXX_STANDARD 17) # c++17

###########
#   GPU   #
###########

find_package(CUDA)

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

###########
#   GPU   #
###########

if (CUDA_FOUND)
 set (CMAKE_CUDA_STANDARD 17)
 set(CUDA_SEPARABLE_COMPILATION ON)
 set(CUDA_HOST_COMPILER g++) #${CMAKE_CXX_COMPILER})
 set(CMAKE_CUDA_FLAGS RELEASE) # TODO: use CXX flag
 link_directories(/opt/cuda) # TODO: proper CUDA link dir var ?
 list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr" )
 list(APPEND CUDA_NVCC_FLAGS "-std=c++17" )

 ###############
 #  LIBGPUALG  #
 ###############
 # https://developer.nvidia.com/blog/building-cuda-applications-cmake/ #
 file(GLOB LIBALGGPUSOURCES "src/libgpualg/*.cu")
 #cuda_compile(gpualg ${LIBALGGPUSOURCES}) # NOT SURE
 cuda_add_library(gpualg STATIC ${LIBALGGPUSOURCES})
 target_link_libraries(gpualg ${CUDA_cusolver_LIBRARY})

 #################
 # TESTLIBGPUALG #
 #################
 #file(GLOB TESTLIBGPUALGSOURCES "src/testlibgpualg/*.cu")
 #foreach(file_path ${TESTLIBGPUALGSOURCES})
 #   string(REPLACE ".cu" "" bin_path_name ${file_path} )
 #   get_filename_component(bin_name ${bin_path_name} NAME)
 #   message("FILE IS: ${file_path}, BIN IS: ${bin_name}")
 #   CUDA_COMPILE(${bin_name} ${file_path})
 #   cuda_add_executable(${bin_name} ${file_path})
 #   target_link_libraries(${bin_name} CSV alg gpualg)
 #endforeach()

 # Working:
 CUDA_COMPILE(testgpumean "src/testlibgpualg/testgpumean.cu")
 cuda_add_executable(testgpumean "src/testlibgpualg/testgpumean.cu")
 target_link_libraries(testgpumean CSV alg gpualg)

 CUDA_COMPILE(testgpumult "src/testlibgpualg/testgpumult.cu")
 cuda_add_executable(testgpumult "src/testlibgpualg/testgpumult.cu")
 target_link_libraries(testgpumult CSV alg gpualg)
 
 CUDA_COMPILE(testgpuope "src/testlibgpualg/testgpuope.cu")
 cuda_add_executable(testgpuope "src/testlibgpualg/testgpuope.cu")
 target_link_libraries(testgpuope CSV alg gpualg)
 
 CUDA_COMPILE(testgpueuclid "src/testlibgpualg/testgpueuclid.cu")
 cuda_add_executable(testgpueuclid "src/testlibgpualg/testgpueuclid.cu")
 target_link_libraries(testgpueuclid CSV alg gpualg)

 CUDA_COMPILE(testgpusvd "src/testlibgpualg/testgpusvd.cu")
 cuda_add_executable(testgpusvd "src/testlibgpualg/testgpusvd.cu")
 target_link_libraries(testgpusvd CSV alg gpualg)

 ###############
 #  LIBGPUICP  #
 ###############
 file(GLOB LIBGPUICPSOURCES "src/libgpuicp/*.cu")
 #cuda_compile(gpuicp ${LIBGPUICPSOURCES}) # NOT SURE
 cuda_add_library(gpuicp STATIC ${LIBGPUICPSOURCES})
 target_link_libraries(gpuicp CSV alg gpualg)

 #################
 # TESTLIBGPUICP #
 #################
 CUDA_COMPILE(testgpucorresp "src/testlibgpuicp/testcorresp.cu")
 cuda_add_executable(testgpucorresp "src/testlibgpuicp/testcorresp.cu")
 target_link_libraries(testgpucorresp CSV alg gpualg gpuicp)

 CUDA_COMPILE(testgpucorresparray "src/testlibgpuicp/testcorresparray.cu")
 cuda_add_executable(testgpucorresparray "src/testlibgpuicp/testcorresparray.cu")
 target_link_libraries(testgpucorresparray CSV alg gpualg gpuicp)

 CUDA_COMPILE(testgpucrosscov "src/testlibgpuicp/testcrosscov.cu")
 cuda_add_executable(testgpucrosscov "src/testlibgpuicp/testcrosscov.cu")
 target_link_libraries(testgpucrosscov CSV alg gpualg gpuicp)

 CUDA_COMPILE(testgpudist "src/testlibgpuicp/testdist.cu")
 cuda_add_executable(testgpudist "src/testlibgpuicp/testdist.cu")
 target_link_libraries(testgpudist CSV alg gpualg gpuicp)
 
 CUDA_COMPILE(testgpucovsadd "src/testlibgpuicp/testcovsadd.cu")
 cuda_add_executable(testgpucovsadd "src/testlibgpuicp/testcovsadd.cu")
 target_link_libraries(testgpucovsadd CSV alg gpualg gpuicp)

 CUDA_COMPILE(testgpubatchcovs "src/testlibgpuicp/testbatchcovs.cu")
 cuda_add_executable(testgpubatchcovs "src/testlibgpuicp/testbatchcovs.cu")
 target_link_libraries(testgpubatchcovs CSV alg gpualg gpuicp)

 #######
 # GPU #
 #######

 file(GLOB GPUSOURCES "src/gpu/*.cu")
 CUDA_COMPILE(GPUICP ${GPUSOURCES})
 cuda_add_executable(GPUICP ${GPUSOURCES})
 target_link_libraries(GPUICP CSV alg gpualg)
else (CUDA_FOUND)
 message("CUDA NOT FOUND !, stick with CPU version")
endif (CUDA_FOUND)
