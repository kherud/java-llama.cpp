![Java 11+](https://img.shields.io/badge/Java-11%2B-informational)
![llama.cpp b3534](https://img.shields.io/badge/llama.cpp-%23b3534-informational)

# Java Bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp)

Inference of Meta's LLaMA model (and others) in pure C/C++.

**You are welcome to contribute**

1. [Quick Start](#quick-start)  
    1.1 [No Setup required](#no-setup-required)   
    1.2 [Setup required](#setup-required)
2. [Documentation](#documentation)  
    2.1 [Example](#example)  
    2.2 [Inference](#inference)  
    2.3 [Infilling](#infilling)  
3. [Android](#importing-in-android)

> [!NOTE]
> Now with support for Llama 3, Phi-3, and flash attention

## Quick Start

Access this library via Maven:

```xml
<dependency>
    <groupId>de.kherud</groupId>
    <artifactId>llama</artifactId>
    <version>3.4.1</version>
</dependency>
```

Bu default the default library artifact is built only with CPU inference support. To enable CUDA, use a `cuda12-linux-x86-64` maven classifier:

```xml
<dependency>
    <groupId>de.kherud</groupId>
    <artifactId>llama</artifactId>
    <version>3.4.1</version>
    <classifier>cuda12-linux-x86-64</classifier>
</dependency>
```

There are multiple [examples](src/test/java/examples).

### No Setup required

We support CPU inference for the following platforms out of the box:

- Linux x86-64, aarch64
- MacOS x86-64, aarch64 (M-series)
- Windows x86-64, x64, arm (32 bit)

For GPU inference, we support:

- Linux x86-64 with CUDA 12.1+

If any of these match your platform, you can include the Maven dependency and get started.

### Setup required

If none of the above listed platforms matches yours, currently you have to compile the library yourself (also if you 
want GPU acceleration).

This consists of two steps: 1) Compiling the libraries and 2) putting them in the right location.

##### Library Compilation

First, have a look at [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md) to know which build arguments to use (e.g. for CUDA support).
Any build option of llama.cpp works equivalently for this project.
You then have to run the following commands in the directory of this repository (java-llama.cpp):

```shell
mvn compile  # don't forget this line
cmake -B build # add any other arguments for your backend, e.g. -DGGML_CUDA=ON
cmake --build build --config Release
```

> [!TIP]
> Use `-DGGML_CURL=ON` to download models via Java code using `ModelParameters#setModelUrl(String)`.

All compiled libraries will be put in a resources directory matching your platform, which will appear in the cmake output. For example something like:

```shell
--  Installing files to /java-llama.cpp/src/main/resources/de/kherud/llama/Linux/x86_64
```

#### Library Location

This project has to load three shared libraries:

- ggml
- llama
- jllama

Note, that the file names vary between operating systems, e.g., `ggml.dll` on Windows, `libggml.so` on Linux, and `libggml.dylib` on macOS.

The application will search in the following order in the following locations:

- In **de.kherud.llama.lib.path**: Use this option if you want a custom location for your shared libraries, i.e., set VM option `-Dde.kherud.llama.lib.path=/path/to/directory`.
- In **java.library.path**: These are predefined locations for each OS, e.g., `/usr/java/packages/lib:/usr/lib64:/lib64:/lib:/usr/lib` on Linux.
  You can find out the locations using `System.out.println(System.getProperty("java.library.path"))`.
  Use this option if you want to install the shared libraries as system libraries.
- From the **JAR**: If any of the libraries weren't found yet, the application will try to use a prebuilt shared library.
  This of course only works for the [supported platforms](#no-setup-required) .

Not all libraries have to be in the same location.
For example, if you already have a llama.cpp and ggml version you can install them as a system library and rely on the jllama library from the JAR.
This way, you don't have to compile anything. 

#### CUDA 

On Linux x86-64 with CUDA 12.1+, the library assumes that your CUDA libraries are findable in `java.library.path`. If you have CUDA installed in a non-standard location, then point the `java.library.path` to the directory containing the `libcudart.so.12` library.

## Documentation

### Example

This is a short example on how to use this library:

```java
public class Example {

    public static void main(String... args) throws IOException {
        ModelParameters modelParams = new ModelParameters()
                .setModelFilePath("/path/to/model.gguf")
                .setNGpuLayers(43);

        String system = "This is a conversation between User and Llama, a friendly chatbot.\n" +
                "Llama is helpful, kind, honest, good at writing, and never fails to answer any " +
                "requests immediately and with precision.\n";
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        try (LlamaModel model = new LlamaModel(modelParams)) {
            System.out.print(system);
            String prompt = system;
            while (true) {
                prompt += "\nUser: ";
                System.out.print("\nUser: ");
                String input = reader.readLine();
                prompt += input;
                System.out.print("Llama: ");
                prompt += "\nLlama: ";
                InferenceParameters inferParams = new InferenceParameters(prompt)
                        .setTemperature(0.7f)
                        .setPenalizeNl(true)
                        .setMirostat(InferenceParameters.MiroStat.V2)
                        .setAntiPrompt("\n");
                for (LlamaOutput output : model.generate(inferParams)) {
                    System.out.print(output);
                    prompt += output;
                }
            }
        }
    }
}
```

Also have a look at the other [examples](src/test/java/examples).

### Inference

There are multiple inference tasks. In general, `LlamaModel` is stateless, i.e., you have to append the output of the 
model to your prompt in order to extend the context. If there is repeated content, however, the library will internally
cache this, to improve performance.

```java
ModelParameters modelParams = new ModelParameters().setModelFilePath("/path/to/model.gguf");
InferenceParameters inferParams = new InferenceParameters("Tell me a joke.");
try (LlamaModel model = new LlamaModel(modelParams)) {
    // Stream a response and access more information about each output.
    for (LlamaOutput output : model.generate(inferParams)) {
        System.out.print(output);
    }
    // Calculate a whole response before returning it.
    String response = model.complete(inferParams);
    // Returns the hidden representation of the context + prompt.
    float[] embedding = model.embed("Embed this");
}
```

> [!NOTE]
> Since llama.cpp allocates memory that can't be garbage collected by the JVM, `LlamaModel` is implemented as an
> AutoClosable. If you use the objects with `try-with` blocks like the examples, the memory will be automatically
> freed when the model is no longer needed. This isn't strictly required, but avoids memory leaks if you use different
> models throughout the lifecycle of your application.

### Infilling

You can simply set `InferenceParameters#setInputPrefix(String)` and `InferenceParameters#setInputSuffix(String)`.

### Model/Inference Configuration

There are two sets of parameters you can configure, `ModelParameters` and `InferenceParameters`. Both provide builder 
classes to ease configuration. `ModelParameters` are once needed for loading a model, `InferenceParameters` are needed
for every inference task. All non-specified options have sensible defaults.

```java
ModelParameters modelParams = new ModelParameters()
        .setModelFilePath("/path/to/model.gguf")
        .setLoraAdapter("/path/to/lora/adapter")
        .setLoraBase("/path/to/lora/base");
String grammar = """
		root  ::= (expr "=" term "\\n")+
		expr  ::= term ([-+*/] term)*
		term  ::= [0-9]""";
InferenceParameters inferParams = new InferenceParameters("")
        .setGrammar(grammar)
        .setTemperature(0.8);
try (LlamaModel model = new LlamaModel(modelParams)) {
    model.generate(inferParams);
}
```

### Logging

Per default, logs are written to stdout.
This can be intercepted via the static method `LlamaModel.setLogger(LogFormat, BiConsumer<LogLevel, String>)`. 
There is text- and JSON-based logging. The default is JSON.
Note, that text-based logging will include additional output of the GGML backend, while JSON-based logging
only provides request logs (while still writing GGML messages to stdout).
To only change the log format while still writing to stdout, `null` can be passed for the callback. 
Logging can be disabled by passing an empty callback.

```java
// Re-direct log messages however you like (e.g. to a logging library)
LlamaModel.setLogger(LogFormat.TEXT, (level, message) -> System.out.println(level.name() + ": " + message));
// Log to stdout, but change the format
LlamaModel.setLogger(LogFormat.TEXT, null);
// Disable logging by passing a no-op
LlamaModel.setLogger(null, (level, message) -> {});
```

## Importing in Android

You can use this library in Android project.
1. Add java-llama.cpp as a submodule in your android `app` project directory
```shell
git submodule add https://github.com/kherud/java-llama.cpp 
```
2. Declare the library as a source in your build.gradle
```gradle
android {
    val jllamaLib = file("java-llama.cpp")

    // Execute "mvn compile" if folder target/ doesn't exist at ./java-llama.cpp/
    if (!file("$jllamaLib/target").exists()) {
        exec {
            commandLine = listOf("mvn", "compile")
            workingDir = file("java-llama.cpp/")
        }
    }

    ...
    defaultConfig {
	...
        externalNativeBuild {
            cmake {
		// Add an flags if needed
                cppFlags += ""
                arguments += ""
            }
        }
    }

    // Declare c++ sources
    externalNativeBuild {
        cmake {
            path = file("$jllamaLib/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    // Declare java sources
    sourceSets {
        named("main") {
            // Add source directory for java-llama.cpp
            java.srcDir("$jllamaLib/src/main/java")
        }
    }
}
```
3. Exclude `de.kherud.llama` in proguard-rules.pro
```proguard
keep class de.kherud.llama.** { *; }
```
