mkdir build
cd build
cmake .. -DTRT_LIB_DIR=/workspace/zhanghandi/dx_fix_bug/envs/TensorRT-8.4.0.6/lib/ -DTRT_INC_DIR=/workspace/zhanghandi/dx_fix_bug/envs/TensorRT-8.4.0.6/include/ -DTRT_OUT_DIR=/workspace/zhanghandi/dx_fix_bug/envs/TensorRT/out/
make -j 40
