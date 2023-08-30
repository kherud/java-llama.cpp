git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cp llama.h ..
mkdir build
cd build
cmake -DLLAMA_METAL=ON -DBUILD_SHARED_LIBS=ON ..
cmake --build . --config Release
cd ../..
mv -f llama.cpp/build/libllama.dylib src/main/resources/
rm -rf llama.cpp