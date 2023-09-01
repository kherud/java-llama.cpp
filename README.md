![llama.cpp b1147](https://img.shields.io/badge/llama.cpp-%23b1147-informational)

# Java Bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp)

The main goal of llama.cpp is to run the LLaMA model using 4-bit integer quantization on a MacBook.
This repository provides Java bindings for the C++ library.

**You are welcome to contribute**

## Quick Start

Access this library via Maven:

```xml
<dependency>
    <groupId>de.kherud</groupId>
    <artifactId>llama</artifactId>
    <version>1.0.0</version>
</dependency>
```

You can then use this library. This is a short example: 

```java
public class Example {

    public static void main(String... args) throws IOException {
        Parameters params = new Parameters.Builder()
                .setNGpuLayers(43)
                .setTemperature(0.7f)
                .setPenalizeNl(true)
                .setMirostat(Parameters.MiroStat.V2)
                .setAntiPrompt(new String[]{"\n"})
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

Also have a look at the [examples](src/test/java/examples).

### Installing the llama.cpp library

Make sure the the `llama.cpp` shared library is appropriately installed for your platform:

- `libllama.so` (linux)
- `libllama.dylib` (macos)
- `llama.dll` (windows)

Refer to the official [readme](https://github.com/ggerganov/llama.cpp#build) for details.
The library can be built with the `llama.cpp` project:

```shell
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON  # add any other arguments for your backend
cmake --build . --config Release
```

Look for the shared library in `build`.

> [!IMPORTANT]
> If you are running MacOS with Metal, you have to put the file `ggml-metal.metal` from `build/bin` in the same directory as the shared library.

Depending on your platform, either:

- Move the file then to the correct directory, e.g., `/usr/local/lib` for most linux distributions. 
If you're not sure where to put it, just run the code. Java will throw an error explaining where it looks.
- Set the JVM option `-Djna.library.path="/path/to/library/"` (IDEs like IntelliJ make this easy)

## Documentation



### Model Configuration

You can configure most options the library offers.
Note however that most options aren't relevant to this Java binding yet (in particular everything that concerns command line interfacing).

```java
Parameters params = new Parameters.Builder()
                            .setInputPrefix("...")
                            .setLoraAdapter("/path/to/lora/adapter")
                            .setLoraBase("/path/to/lora/base")
                            .build();
LlamaModel model = new LlamaModel("/path/to/model.bin", params);
```

### Logging

Both Java and C++ logging can be configured via the static method `LlamaModel.setLogger`:

```java
// The method accepts a BiConsumer<LogLevel, String>.
LlamaModel.setLogger((level, message) -> System.out.println(level.name() + ": " + message));
// This can also be set to null to disable Java logging.
// However, in this case the C++ side will still output to stdout/stderr.
LlamaModel.setLogger(null);
// To completely silence any output, pass a no-op.
LlamaModel.setLogger((level, message) -> {});

// Similarly, a progress callback can be set (only the C++ side will call this).
// I think this is only used to report progress loading the model with a value of 0-1.
// It is thus state specific and can be done via the parameters.
new Parameters.Builder()
        .setProgressCallback(progress -> System.out.println("progress: " + progress))
        .build();
```
