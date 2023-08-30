![llama.cpp b1112](https://img.shields.io/badge/llama.cpp-b1112-informational)

# Java Bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp)

## Quick Start

```
LlamaModel model = new LlamaModel("/path/to/model.bin");
```

#### Configuration

You can configure every option the library offers. 
Note however that most options aren't relevant to this Java binding (in particular everything that concerns command line interfacing). 

```
Parameters params = new Parameters.Builder()
                            .setInputPrefix("...")
                            .setLoraAdapter("/path/to/lora/adapter")
                            .setLoraBase("/path/to/lora/base")
                            .build();
LlamaModel model = new LlamaModel("/path/to/model.bin", params);
```

## Installation

First, make sure to appropriately install the `llama.cpp` shared library for your platform:

- `libllama.so` (linux)
- `libllama.dylib` (macos)
- `llama.dll` (windows)

Refer to the official [readme](https://github.com/ggerganov/llama.cpp#build) for details.
The library can be built with the `llama.cpp` project:

#### make

```
make libllama.so
```

Look for the shared library in the working directory `.`.

#### cmake

```
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON
cmake --build . --config Release
```

Look for the shared library in `build`.

> [!IMPORTANT]
> If you are running MacOS with Metal, you have to put the file `ggml-metal.metal` from `build/bin` in the same directory as the shared library.

Deployment to Maven Central is coming soon. The installation will also be improved soon. 

## Todo-List

- Grammar
- Guidance
- Caching
- Improve error handling
- Add synchronization
- Separate Parameters into ModelParameters and InferenceParameters
- Return nProp tokens with probability instead of only sampled one
