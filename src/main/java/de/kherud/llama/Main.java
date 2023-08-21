package de.kherud.llama;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;

import com.sun.jna.ptr.FloatByReference;

import de.kherud.llama.foreign.LlamaLibrary;

public class Main {

	public static void main(String... args) {
//		Parameters params = new Parameters.Builder()
//				.
		System.out.println(LlamaLibrary.llama_print_system_info());
		LlamaModel model = new LlamaModel("/run/media/konstantin/Seagate/models/llama2/llama-2-13b-chat/ggml-model-q4_0.bin");
		IntBuffer tokens = model.tokenize("Hello world, how are you?");
		FloatByReference buffer = LlamaLibrary.llama_get_logits(model.ctx);
		buffer.setValue(0);
		buffer.getPointer().setFloat(Float.BYTES, 1f);
//		buffer.getPointer().share(Float.BYTES).setFloat();
		System.out.println(Arrays.toString(buffer.getPointer().getFloatArray(0, model.getVocabularySize())));
		System.out.println(buffer);
		int result = LlamaLibrary.llama_eval(model.ctx, tokens, tokens.capacity(), 0, 1);
		System.out.println(result);
		System.out.println(Arrays.toString(buffer.getPointer().getFloatArray(0, model.getVocabularySize())));
//		model.eval(model.tokenize("Hello, how are you?"));
//		int token = model.sample();
//		System.out.println(token);

	}


}
