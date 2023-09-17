package de.kherud.llama;

import java.lang.annotation.Native;
import java.util.Iterator;
import java.util.NoSuchElementException;


public class LlamaModel {

	private static final ModelParameters defaultModelParams = new ModelParameters.Builder().build();
	private static final InferenceParameters defaultInferenceParams = new InferenceParameters.Builder().build();

	@Native
	private long ctx;

	/**
	 * Load a <b>gguf</b> llama.cpp model from a given file path with default {@link ModelParameters}.
	 *
	 * @param filePath a file path pointing to the model
	 * @throws LlamaException if no model could be loaded from the given file path
	 */
	public LlamaModel(String filePath) {
		this(filePath, defaultModelParams);
	}

	/**
	 * Load a <b>gguf</b> llama.cpp model from a given file path with custom {@link ModelParameters}.
	 *
	 * @param filePath a file path pointing to the model
	 * @param parameters the set of previously configured options
	 * @throws LlamaException if no model could be loaded from the given file path
	 */
	public LlamaModel(String filePath, ModelParameters parameters) {
		loadModel(filePath, parameters);
	}

	static {
		LlamaLoader.initialize();
	}

	public String complete(String prompt) {
		return complete(prompt, defaultInferenceParams);
	}

	public String complete(String prompt, InferenceParameters parameters) {
		return getFull(prompt, parameters);
	}

	public Iterable<String> generate(String prompt) {
		return generate(prompt, defaultInferenceParams);
	}

	public Iterable<String> generate(String prompt, InferenceParameters parameters) {
		return () -> new LlamaIterator(prompt, parameters);
	}

	private native void loadModel(String filePath, ModelParameters parameters) throws LlamaException;
	private native void setupInference(String prompt, InferenceParameters parameters);
	private native String getFull(String prompt, InferenceParameters parameters);
	private native String getNext(LlamaIterator iterator);

	// fields are modified by native code and thus should not be final
	@SuppressWarnings("FieldMayBeFinal")
	private final class LlamaIterator implements Iterator<String> {

		@Native
		private boolean hasNext = true;
		@Native
		private long generatedCount = 0;
		@Native
		private long tokenIndex = 0;

		private LlamaIterator(String prompt, InferenceParameters parameters) {
			setupInference(prompt, parameters);
		}

		@Override
		public boolean hasNext() {
			return hasNext;
		}

		@Override
		public String next() {
			if (!hasNext) {
				throw new NoSuchElementException();
			}
			return getNext(this);
		}
	}

}
