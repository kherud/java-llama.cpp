package de.kherud.llama;

import de.kherud.llama.foreign.LlamaLibrary;

public class Main {

	public static void main(String... args) {
//		Parameters params = new Parameters.Builder()
//				.
		System.out.println(LlamaLibrary.llama_print_system_info());
		LlamaModel model = new LlamaModel("/run/media/konstantin/Seagate/models/llama2/llama-2-13b-chat/ggml-model-q4_0.bin");
		model.eval(model.tokenize("Hello, how are you?"));
		int token = model.sample();
		System.out.println(token);

	}


}
