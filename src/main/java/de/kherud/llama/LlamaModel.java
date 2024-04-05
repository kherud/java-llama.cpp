package de.kherud.llama;

import java.lang.annotation.Native;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;

import org.jetbrains.annotations.NotNull;

/**
 * This class is a wrapper around the llama.cpp functionality.
 * Upon being created, it natively allocates memory for the model context.
 * Thus, this class is an {@link AutoCloseable}, in order to de-allocate the memory when it is no longer being needed.
 * <p>
 * The main functionality of this class is:
 * <ul>
 *     <li>Streaming answers (and probabilities) via {@link #generate(InferenceParameters)}</li>
 *     <li>Creating whole responses to prompts via {@link #complete(InferenceParameters)}</li>
 *     <li>Creating embeddings via {@link #embed(String)} (make sure to configure {@link ModelParameters#setEmbedding(boolean)}</li>
 *     <li>Accessing the tokenizer via {@link #encode(String)} and {@link #decode(int[])}</li>
 * </ul>
 */
public class LlamaModel implements AutoCloseable {

	static {
		LlamaLoader.initialize();
	}

	@Native
	private long ctx;

	/**
	 * Load with the given {@link ModelParameters}. Make sure to either set
	 * <ul>
	 *     <li>{@link ModelParameters#setModelFilePath(String)}</li>
	 *     <li>{@link ModelParameters#setModelUrl(String)}</li>
	 *     <li>{@link ModelParameters#setHuggingFaceRepository(String)}}, {@link ModelParameters#setHuggingFaceFile(String)}</li>
	 * </ul>
	 *
	 * @param parameters the set of options
	 * @throws LlamaException if no model could be loaded from the given file path
	 */
	public LlamaModel(ModelParameters parameters) {
		loadModel(parameters.toString());
	}

	/**
	 * Generate and return a whole answer with custom parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @return an LLM response
	 */
	public String complete(InferenceParameters parameters) {
		parameters.setStream(false);
		int taskId = requestCompletion(parameters.toString());
		Output output = receiveCompletion(taskId);
		return output.text;
	}

	/**
	 * Generate and stream outputs with custom inference parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @return iterable LLM outputs
	 */
	public Iterable<Output> generate(InferenceParameters parameters) {
		return () -> new LlamaIterator(parameters);
	}

	/**
	 * Get the embedding of a string. Note, that the prompt isn't preprocessed in any way, nothing like
	 * "User: ", "###Instruction", etc. is added.
	 *
	 * @param prompt the string to embed
	 * @return an embedding float array
	 * @throws IllegalStateException if embedding mode was not activated (see {@link ModelParameters#setEmbedding(boolean)})
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

//	/**
//	 * Sets a callback for both Java and C++ log messages. Can be set to {@code null} to disable logging.
//	 *
//	 * @param callback a method to call for log messages
//	 */
//	public static native void setLogger(@Nullable BiConsumer<LogLevel, String> callback);

	@Override
	public void close() {
		delete();
	}

	// don't overload native methods since the C++ function names get nasty
	private native void loadModel(String parameters) throws LlamaException;
	private native int requestCompletion(String params) throws LlamaException;
	private native Output receiveCompletion(int taskId) throws LlamaException;
	private native byte[] decodeBytes(int[] tokens);
	private native void delete();

	/**
	 * A generated output of the LLM. Note that you have to configure {@link InferenceParameters#setNProbs(int)}
	 * in order for probabilities to be returned.
	 */
	public static final class Output {

		@NotNull
		public final String text;
		@NotNull
		public final Map<String, Float> probabilities;
		private final boolean stop;

		private Output(byte[] generated, @NotNull Map<String, Float> probabilities, boolean stop) {
			this.text = new String(generated, StandardCharsets.UTF_8);
			this.probabilities = probabilities;
			this.stop = stop;
		}

		@Override
		public String toString() {
			return text;
		}

	}

	private final class LlamaIterator implements Iterator<Output> {

		private final int taskId;

		@Native
		@SuppressWarnings("FieldMayBeFinal")
		private boolean hasNext = true;

		private LlamaIterator(InferenceParameters parameters) {
			parameters.setStream(true);
			taskId = requestCompletion(parameters.toString());
		}

		@Override
		public boolean hasNext() {
			return hasNext;
		}

		@Override
		public Output next() {
			if (!hasNext) {
				throw new NoSuchElementException();
			}
			Output output = receiveCompletion(taskId);
			hasNext = !output.stop;
			return output;
		}
	}

}
