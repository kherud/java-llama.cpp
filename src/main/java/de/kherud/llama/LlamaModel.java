package de.kherud.llama;

import com.sun.jna.Pointer;

public class LlamaModel implements AutoCloseable {

//	private final Pointer ctx;

	public LlamaModel(String filePath) {
//		this(filePath, LlamaCpp.INSTANCE.llama_context_default_params());
	}

//	public LlamaModel(String filePath, ContextParameters parameters) {
//		this.ctx = LlamaCpp.INSTANCE.llama_load_model_from_file(filePath, parameters);
//	}

	@Override
	public void close() throws Exception {
//		LlamaCpp.INSTANCE.llama_free_model(ctx);
	}
}
