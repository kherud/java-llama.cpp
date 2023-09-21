![Java 11+](https://img.shields.io/badge/Java-11%2B-informational)
![llama.cpp b1261](https://img.shields.io/badge/llama.cpp-%23b1261-informational)

# Java Bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp)

The main goal of llama.cpp is to run the LLaMA model using 4-bit integer quantization on a MacBook.
This repository provides Java bindings for the C++ library.

**You are welcome to contribute**

> [!NOTE]
> Version 2.0 of this library just released and introduces some breaking changes. The underlying Java/C++ interfacing 
> technology changed from JNA to JNI with a focus on performance. If you prefer to use the old JNA version, you can 
> still access it as 1.x.x in the [jna branch](https://github.com/kherud/java-llama.cpp/tree/jna).

## Quick Start

Access this library via Maven:

```xml
<dependency>
    <groupId>de.kherud</groupId>
    <artifactId>llama</artifactId>
    <version>2.0.0</version>
</dependency>
```

Here is a short example:

```java
public class Example {

    public static void main(String... args) throws IOException {
		LlamaModel.setLogger((level, message) -> System.out.print(message));
		ModelParameters modelParams = new ModelParameters.Builder()
				.setNGpuLayers(43)
				.build();
		InferenceParameters inferParams = new InferenceParameters.Builder()
				.setTemperature(0.7f)
				.setPenalizeNl(true)
				.setMirostat(InferenceParameters.MiroStat.V2)
				.setAntiPrompt(new String[]{"\n"})
				.build();

		String modelPath = "/run/media/konstantin/Seagate/models/llama2/llama-2-13b-chat/ggml-model-q4_0.gguf";
		String system = "This is a conversation between User and Llama, a friendly chatbot.\n" +
				"Llama is helpful, kind, honest, good at writing, and never fails to answer any " +
				"requests immediately and with precision.\n";
		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
		try (LlamaModel model = new LlamaModel(modelPath, modelParams)) {
			System.out.print(system);
			String prompt = system;
			while (true) {
				prompt += "\nUser: ";
				System.out.print("\nUser: ");
				String input = reader.readLine();
				prompt += input;
				System.out.print("Llama: ");
				prompt += "\nLlama: ";
				for (String output : model.generate(prompt, inferParams)) {
					System.out.print(output);
					prompt += output;
				}
			}
		}
    }
}
```

Also have a look at the [examples](src/test/java/examples).

### No Setup required

We support CPU inference for the following platforms out of the box:

- Linux x86-64
- MacOS x86-64, arm64 (M1)
- Windows x86-64

If any of these match your platform, you can include the Maven dependency and get started.

### Setup required

If none of the above listed platforms matches yours, currently you have to compile the library yourself (also if you 
want GPU acceleration, see below). More support is planned soon.

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
echo $JAVA_HOME # for inlux/macos
echo %JAVA_HOME% # for windows
```

Then, run the following commands in the directory of this repository (java-llama.cpp):

```shell
mvn compile
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON  # add any other arguments for your backend
cmake --build . --config Release
```

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

This repository provides default support for CPU based inference. You can compile `llama.cpp` any way you want, however.
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

### Inference

There are multiple inference tasks. In general, `LlamaModel` is stateless, i.e., you have to append the output of the 
model to your prompt in order to extend the context. If there is repeated content, however, the library will internally
cache this, to improve performance.

```java
try (LlamaModel model = new LlamaModel("/path/to/gguf-model")) {
    // Stream a response and access more information about each output.
    for (String output : model.generate("Tell me a joke.")) {
        System.out.print(output);
    }
    // Calculate a whole response before returning it.
    String response = model.complete("Tell me another one");
    // Returns the hidden representation of the context + prompt.
    float[] embedding = model.embed("Embed this");
}
```

> [!NOTE]
> Since llama.cpp allocates memory that can't be garbage collected by the JVM, `LlamaModel` is implemented as an
> AutoClosable. If you use the objects with `try-with` blocks like the examples, the memory will be automatically
> freed when the model is no longer needed. This isn't strictly required, but avoids memory leaks if you use different
> models throughout the lifecycle of your application.

### Model/Inference Configuration

There are two sets of parameters you can configure, `ModelParameters` and `InferenceParameters`. Both provide builder 
classes to ease configuration. All non-specified options have sensible defaults.

```java
ModelParameters modelParams = new ModelParameters.Builder()
                            .setLoraAdapter("/path/to/lora/adapter")
                            .setLoraBase("/path/to/lora/base")
                            .build();
InferenceParameters inferParams = new InferenceParameters.Builder()
		.setGrammar(new File("/path/to/grammar.gbnf"))
        .setTemperature(0.8)
		.build();
LlamaModel model = new LlamaModel("/path/to/model.bin", modelParams);
model.generate(prompt, inferParams)
```

### Logging

Both Java and C++ logging can be configured via the static method `LlamaModel.setLogger`:

```java
// The method accepts a BiConsumer<LogLevel, String>.
LlamaModel.setLogger((level, message) -> System.out.println(level.name() + ": " + message));
// To completely silence any output, pass a no-op.
LlamaModel.setLogger((level, message) -> {});

// Similarly, a progress callback can be set (only the C++ side will call this).
// I think this is only used to report progress loading the model with a value of 0-1.
// It is thus state specific and can be done via the parameters.
new ModelParameters.Builder()
        .setProgressCallback(progress -> System.out.println("progress: " + progress))
        .build();
```
