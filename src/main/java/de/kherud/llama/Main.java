package de.kherud.llama;

import de.kherud.llama.foreign.LlamaLibrary;

public class Main {

	public static void main(String... args) {
		System.out.println(LlamaLibrary.llama_print_system_info());
		LlamaModel model = new LlamaModel("/Users/konstantin.herud/denkbares/projects/llama.cpp/models/13B/ggml-model-q4_0.bin");
		model.forward(model.tokenize("Hello, how are you?"));
	}


}
