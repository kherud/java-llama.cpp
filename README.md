![llama.cpp b1170](https://img.shields.io/badge/llama.cpp-%23b1170-informational)

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
    <version>1.1.0</version>
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

Make sure the `llama.cpp` shared library is appropriately installed for your platform:

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


### Inference

There are multiple inference tasks. In general, `LlamaModel` is stateful, i.e., unless `LlamaModel#reset()` is called,
each subsequent call takes the previous requests and responses into context.

```java
try (LlamaModel model = new LlamaModel("/path/to/gguf-model")) {
    // Stream a response and access more information about each output.
    for (LlamaModel.Output output : model.generate("Tell me a joke.")) {
        System.out.print(output);
    }
    // Calculate a whole response before returning it.
    String response = model.complete("Tell me another one");
    // Returns the hidden representation of the context + prompt.
    float[] embedding = model.getEmbedding("Embed this");
}
```

If the model runs out of context at any point, it truncates the saved context to keep half of the maximal context size.

> [!NOTE]
> Since llama.cpp allocates memory that can't be garbage collected by the JVM, `LlamaModel` is implemented as an
> AutoClosable. If you use the objects with `try-with` blocks like the examples, the memory will be automatically
> freed when the model is no longer needed. This isn't strictly required, but avoids memory leaks if you use different
> models throughout the lifecycle of your application.

### Model Information

There is some information you can access of your loaded model.

```java
try (LlamaModel model = new LlamaModel("/path/to/gguf-model")) {
    // the maximal amount of tokens this model can process
    int contextSize = model.getContextSize();
    // the hidden dimensionality of this model 
    int embeddingSize = model.getEmbeddingSize();
    // the total amount of tokens known in the vocabulary
    int vocabularySize = model.getVocabularySize();
    // the tokenization method of the model, i.e., sentence piece or byte pair encoding
    VocabularyType vocabType = model.getVocabularyType();
}
```

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

### Debugging Information

There are some methods to debug your shared library:

```java
// Returns some information like "AVX = 1 | AVX2 = 1 | AVX512 = 0 | ...".
String systemInfo = LlamaLibrary.llama_print_system_info();
// I think this returns the amount of logical cores llama.cpp can use (not completely sure though).
int maxDevices = llama_max_devices();
// These two methods return a C bool, which is a byte in Java (0 = false, >0 = true).
boolean mmapSupported = llama_mmap_supported() > 0;
boolean mLockSupported = llama_mlock_supported() > 0;
// Returns a timestamp of the llama.cpp backend 
long time = LlamaLibrary.llama_time_us();
```

### Backend

The `llama.cpp` backend is statically initialized upon accessing `LlamaModel`. If you want to de-allocate and maybe 
later re-initialize it for whatever reason, there are two methods: 

```
// This method takes a bool (byte, 0 = false, >0 = true) to enable NUMA optimizations.
// Per default, they are off. If you want to enable them, first free the backend, then initialize it again with 1.
LLamaLibrary.llama_backend_init((byte) 0);
LLamaLibrary.llama_backend_free();
```
