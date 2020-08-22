# MNN Ailibaba's lightweight embedded AI framework.
https://github.com/alibaba/MNN

This repo is about hwo to compile/cross-compile the MNN framework in Linux. (I just tested it in Ubuntu 16.04.6, and it might be something wrong in CentOS7.)

## (1) download the MNN: 
git clone https://github.com/alibaba/MNN
## (2) make sure that you have installed these:(https://www.yuque.com/mnn/en/build_linux)
cmake (version >=3.10 is recommended)  
protobuf (version >= 3.0 is required)  
gcc (version >= 4.9 is required)
## (3) compile converter(https://www.yuque.com/mnn/en/cvrt_linux)
cd MNN/  
./schema/generate.sh  
mkdir build && cd build  
cmake .. -DMNN_BUILD_CONVERTER=true && make -j4  
download the model from http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz  
./MNNConvert -f TF --modelFile ../../mobilenet_v1_1.0_224_frozen.pb --MNNModel mnv1.mnn --bizCode MNN  
__mnv1.mnn__ is needed for test.
## (4) cross-compile inference(https://www.yuque.com/mnn/en/build_linux)
MNN's compile is composed of three relative independent stages/programs(inference, ), if we juse want to realize inference, just compile inference and converter.  
cd MNN/  
./schema/generate.sh  
For cross-compile, I just downloaded the aarch64-toolchain from http://releases.linaro.org/components/toolchain/binaries/latest-7/  
export cross_compile_toolchain=/path/to/linaro/aarch64  
mkdir build && cd build  
cmake .. -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_C_COMPILER=$cross_compile_toolchain/bin/aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=$cross_compile_toolchain/bin/aarch64-linux-gnu-g++ -DCMAKE_ASM_COMPILER=$cross_compile_toolchain/bin/aarch64-linux-gnu-gcc  
make -j4  
__This step will generate some dynamic libs for use. If some libs cannot find, just go to cross-toolchain to find.__
## (5) cross-compile demo
cd ../demo/exec/  
add this __mnv1_mnn.cpp__ in it, and add these in CMakeLists.txt:  
add_executable(mnv1_mnn.out ${CMAKE_CURRENT_LIST_DIR}/mnv1_mnn.cpp)  
target_link_libraries(mnv1_mnn.out ${MNN_DEPS})  
cd ../../build/  
cmake .. -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_C_COMPILER=$cross_compile_toolchain/bin/aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=$cross_compile_toolchain/bin/aarch64-linux-gnu-g++ -DCMAKE_ASM_COMPILER=$cross_compile_toolchain/bin/aarch64-linux-gnu-gcc -DMNN_BUILD_DEMO=ON  
you will get the exe file  
## (6) Test it in board
./mnv1_mnn.out mnv1.mnn dog.png ./label.txt  
__we jsut need label.txt, some pics for test, mnv1.mnn, mnv1,out and some link libs.__




