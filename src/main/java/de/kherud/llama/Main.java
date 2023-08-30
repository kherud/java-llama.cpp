package de.kherud.llama;

import de.kherud.llama.foreign.LlamaLibrary;

public class Main {

	public static void main(String... args) {
		Parameters params = new Parameters.Builder()
				.setNProbs(0)
				.setTopK(40)
				.setTfsZ(1)
				.setTypicalP(1)
				.setTopP(0.5f)
				.setTemperature(0.7f)
				.setNGpuLayers(1)
//				.setTemperature(0.0f)
//				.setMirostat(Parameters.MiroStat.V2)
				.build();
		System.out.println(LlamaLibrary.llama_print_system_info());
		// LlamaModel model = new LlamaModel("/run/media/konstantin/Seagate/models/llama2/llama-2-13b-chat/gguf-model-q4_0.bin", params);
		LlamaModel model = new LlamaModel("/Users/konstantin.herud/denkbares/projects/llama.cpp/models/13B/gguf-model-q4_0.bin", params);

		model.tokenize("This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.\\n\\nUser: Hello, how are you?\\nLlama:");
		model.hasNextToken = true;
		while (model.hasNextToken) {
			LlamaModel.Output token = model.nextToken();
			System.out.print(token);
		}

//		System.out.println(model.getEmbeddingSize());
//		IntBuffer tokens = model.tokenize("Hello world, how are you?");
//		FloatByReference buffer = LlamaLibrary.llama_get_logits(model.ctx);
//		buffer.setValue(0);
//		buffer.getPointer().setFloat(Float.BYTES, 1f);
////		buffer.getPointer().share(Float.BYTES).setFloat();
//		System.out.println(Arrays.toString(buffer.getPointer().getFloatArray(0, model.getVocabularySize())));
//		System.out.println(buffer);
//		int result = LlamaLibrary.llama_eval(model.ctx, tokens, tokens.capacity(), 0, 1);
//		System.out.println(result);
//		System.out.println(Arrays.toString(buffer.getPointer().getFloatArray(0, model.getVocabularySize())));
//		model.eval(model.tokenize("Hello, how are you?"));
//		int token = model.sample();
//		System.out.println(token);

	}


}
