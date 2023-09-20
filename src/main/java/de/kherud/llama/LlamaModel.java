package de.kherud.llama;

import java.lang.annotation.Native;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.function.BiConsumer;

import org.jetbrains.annotations.Nullable;

/**
 * This class is a wrapper around the llama.cpp functionality.
 * Upon being created, it natively allocates memory for the model context.
 * Thus, this class is an {@link AutoCloseable}, in order to de-allocate the memory when it is no longer being needed.
 * <p>
 * The main functionality of this class is:
 * <ul>
 *     <li>Streaming answers (and probabilities) via {@link #generate(String)}</li>
 *     <li>Creating whole responses to prompts via {@link #complete(String)}</li>
 *     <li>Creating embeddings via {@link #embed(String)} (make sure to configure {@link ModelParameters.Builder#setEmbedding(boolean)}</li>
 *     <li>Accessing the tokenizer via {@link #encode(String)} and {@link #decode(int[])}</li>
 * </ul>
 */
public class LlamaModel implements AutoCloseable {

	static {
		LlamaLoader.initialize();
	}

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

	/**
	 * Generate and return a whole answer with default parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @param prompt the LLM prompt
	 * @return an LLM response
	 */
	public String complete(String prompt) {
		return complete(prompt, defaultInferenceParams);
	}

	/**
	 * Generate and return a whole answer with custom parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @param prompt the LLM prompt
	 * @return an LLM response
	 */
	public String complete(String prompt, InferenceParameters parameters) {
		byte[] bytes = getFull(prompt, parameters);
		return new String(bytes, StandardCharsets.UTF_8);
	}

	/**
	 * Generate and stream outputs with default inference parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @param prompt the LLM prompt
	 * @return iterable LLM outputs
	 */
	public Iterable<String> generate(String prompt) {
		return generate(prompt, defaultInferenceParams);
	}

	/**
	 * Generate and stream outputs with custom inference parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @param prompt the LLM prompt
	 * @return iterable LLM outputs
	 */
	public Iterable<String> generate(String prompt, InferenceParameters parameters) {
		return () -> new LlamaIterator(prompt, parameters);
	}

	/**
	 * Get the embedding of a string. Note, that the prompt isn't preprocessed in any way, nothing like
	 * "User: ", "###Instruction", etc. is added.
	 *
	 * @param prompt the string to embed
	 * @return an embedding float array
	 * @throws IllegalStateException if embedding mode was not activated (see {@link ModelParameters.Builder#setEmbedding(boolean)})
	 */
	public native float[] embed(String prompt);

	/**
	 * Tokenize a prompt given the native tokenizer
	 *
	 * @param prompt the prompt to tokenize
	 * @return an array of integers each representing a token id
	 */
	public native int[] encode(String prompt);

	/**
	 * Convert an array of token ids to its string representation
	 *
	 * @param tokens an array of tokens
	 * @return the token ids decoded to a string
	 */
	public String decode(int[] tokens)  {
		byte[] bytes = decodeBytes(tokens);
		return new String(bytes, StandardCharsets.UTF_8);
	}

	/**
	 * Sets a callback for both Java and C++ log messages. Can be set to {@code null} to disable logging.
	 *
	 * @param callback a method to call for log messages
	 */
	public static native void setLogger(@Nullable BiConsumer<LogLevel, String> callback);

	@Override
	public void close() {
		delete();
	}

	private native void loadModel(String filePath, ModelParameters parameters) throws LlamaException;
	private native void setupInference(String prompt, InferenceParameters parameters);
	private native byte[] getFull(String prompt, InferenceParameters parameters);
	private native byte[] getNext(LlamaIterator iterator);
	private native byte[] decodeBytes(int[] tokens);
	private native void delete();

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
			byte[] bytes = getNext(this);
			return new String(bytes, StandardCharsets.UTF_8);
		}
	}

}
