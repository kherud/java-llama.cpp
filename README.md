![llama.cpp b1112](https://img.shields.io/badge/llama.cpp-b1112-informational)

# Java Bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp)

## Quick Start

```java
public class Example {

    public static void main(String... args) throws IOException {
        Parameters params = new Parameters.Builder()
                .setNGpuLayers(1)
                .setTemperature(0.7f)
                .setPenalizeNl(true)
                .setMirostat(Parameters.MiroStat.V2)
                .build();

        String modelPath = "/path/to/gguf-model-q4_0.bin";
        String system = "This is a conversation between User and Llama, a friendly chatbot.\n" +
                "Llama is helpful, kind, honest, good at writing, and never fails to answer any " +
                "requests immediately and with precision.\n";
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        try (LlamaModel model = new LlamaModel(modelPath, params)) {
            String prompt = system;
            while (true) {
                prompt += "\nUser: ";
                System.out.print(prompt);
                String input = reader.readLine();
                prompt += input;
                System.out.print("Llama: ");
                prompt += "\nLlama: ";
                for (LlamaModel.Output output : model.generate(prompt)) {
                    System.out.print(output);
                }
                prompt = "";
            }
        }

    }
}
```

#### Configuration

You can configure every option the library offers. 
Note however that most options aren't relevant to this Java binding (in particular everything that concerns command line interfacing). 

```java
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

```shell
make libllama.so
```

Look for the shared library in the working directory `.`.

#### cmake

```shell
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON  # add any other arguments for your backend
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
