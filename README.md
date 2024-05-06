![Java 11+](https://img.shields.io/badge/Java-11%2B-informational)
![llama.cpp b2797](https://img.shields.io/badge/llama.cpp-%23b2797-informational)

# Java Bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp)

The main goal of llama.cpp is to run the LLaMA model using 4-bit integer quantization on a MacBook.
This repository provides Java bindings for the C++ library.

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
    <version>3.0.2</version>
</dependency>
```

There are multiple [examples](src/test/java/examples):

### No Setup required

We support CPU inference for the following platforms out of the box:

- Linux x86-64, aarch64
- MacOS x86-64, aarch64 (M1)
- Windows x86-64, x64, arm (32 bit)

If any of these match your platform, you can include the Maven dependency and get started.

### Setup required

If none of the above listed platforms matches yours, currently you have to compile the library yourself (also if you 
want GPU acceleration, see below).

This requires:

- Git
- A C++11 conforming compiler
- The [cmake](https://www.cmake.org/) build system
- Java, Maven, and setting [JAVA_HOME](https://www.baeldung.com/java-home-on-windows-7-8-10-mac-os-x-linux)

Make sure everything works by running

```
g++ -v  # depending on your compiler
java -version
mvn -v
echo $JAVA_HOME # for linux/macos
echo %JAVA_HOME% # for windows
```

Then, checkout [llama.cpp](https://github.com/ggerganov/llama.cpp) to know which build arguments to use (e.g. for CUDA support).
Finally, you have to run following commands in the directory of this repository (java-llama.cpp).
Remember to add your build arguments in the fourth line (`cmake ..`):

```shell
mvn compile
mkdir build
cd build
cmake .. # add any other arguments for your backend
cmake --build . --config Release
```

> [!TIP]
> Use `-DLLAMA_CURL=ON` to download models via Java code using `ModelParameters#setModelUrl(String)`.

All required files will be put in a resources directory matching your platform, which will appear in the cmake output. For example something like:

```shell
--  Installing files to /java-llama.cpp/src/main/resources/de/kherud/llama/Linux/x86_64
```

This includes:

- Linux: `libllama.so`, `libjllama.so`
- MacOS: `libllama.dylib`, `libjllama.dylib`, `ggml-metal.metal`
- Windows: `llama.dll`, `jllama.dll`

If you then compile your own JAR from this directory, you are ready to go. Otherwise, if you still want to use the library
as a Maven dependency, see below how to set the necessary paths in order for Java to find your compiled libraries.

### Custom llama.cpp Setup (GPU acceleration)

This repository provides default support for CPU based inference. You can compile `llama.cpp` any way you want, however (see [Setup Required](#setup-required)).
In order to use your self-compiled library, set either of the [JVM options](https://www.jetbrains.com/help/idea/tuning-the-ide.html#configure-jvm-options):

- `de.kherud.llama.lib.path`, for example `-Dde.kherud.llama.lib.path=/directory/containing/lib`
- `java.library.path`, for example `-Djava.library.path=/directory/containing/lib`

This repository uses [`System#mapLibraryName`](https://docs.oracle.com/javase%2F7%2Fdocs%2Fapi%2F%2F/java/lang/System.html) to determine the name of the shared library for you platform.
If for any reason your library has a different name, you can set it with

- `de.kherud.llama.lib.name`, for example `-Dde.kherud.llama.lib.name=myname.so`

For compiling `llama.cpp`, refer to the official [readme](https://github.com/ggerganov/llama.cpp#build) for details.
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
                for (String output : model.generate(inferParams)) {
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
    for (String output : model.generate(inferParams)) {
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
