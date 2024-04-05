package de.kherud.llama;

import java.lang.annotation.Native;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;

import org.jetbrains.annotations.NotNull;

import de.kherud.llama.args.InferenceParameters;
import de.kherud.llama.args.ModelParameters;

/**
 * This class is a wrapper around the llama.cpp functionality.
 * Upon being created, it natively allocates memory for the model context.
 * Thus, this class is an {@link AutoCloseable}, in order to de-allocate the memory when it is no longer being needed.
 * <p>
 * The main functionality of this class is:
 * <ul>
 *     <li>Streaming answers (and probabilities) via {@link #generate(String)}</li>
 *     <li>Creating whole responses to prompts via {@link #complete(String)}</li>
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
	 * Generate and return a whole answer with default parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @param prompt the LLM prompt
	 * @return an LLM response
	 */
	public String complete(String prompt) {
		return complete(prompt, new InferenceParameters());
	}

	/**
	 * Generate and return a whole answer with custom parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @param prompt the LLM prompt
	 * @return an LLM response
	 */
	public String complete(String prompt, InferenceParameters parameters) {
		byte[] bytes = getAnswer(prompt, parameters.toString());
		return new String(bytes, StandardCharsets.UTF_8);
	}

	/**
	 * Infill a whole answer with default parameters. Note, that the prompt isn't preprocessed in any
	 * way. Nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @param prefix the prefix prompt of the completion to infill
	 * @param suffix the suffix prompt of the completion to infill
	 * @return an LLM response
	 */
	public String complete(String prefix, String suffix) {
		return complete(prefix, suffix, new InferenceParameters());
	}

	/**
	 * Infill a whole answer with custom parameters. Note, that the prompt isn't preprocessed in any
	 * way. Nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @param prefix the prefix prompt of the completion to infill
	 * @param suffix the suffix prompt of the completion to infill
	 * @return an LLM response
	 */
	public String complete(String prefix, String suffix, InferenceParameters parameters) {
		byte[] bytes = getInfill(prefix, suffix, parameters.toString());
		return new String(bytes, StandardCharsets.UTF_8);
	}

	/**
	 * Generate and stream outputs with default inference parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @param prompt the LLM prompt
	 * @return iterable LLM outputs
	 */
	public Iterable<Output> generate(String prompt) {
		return generate(prompt, new InferenceParameters());
	}

	/**
	 * Generate and stream outputs with custom inference parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @param prompt the LLM prompt
	 * @return iterable LLM outputs
	 */
	public Iterable<Output> generate(String prompt, InferenceParameters parameters) {
		return () -> new LlamaIterator(prompt, parameters);
	}

	/**
	 * Infill and stream outputs with default inference parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @param prefix the prefix prompt of the completion to infill
	 * @param suffix the suffix prompt of the completion to infill
	 * @return iterable LLM outputs
	 */
	public Iterable<Output> generate(String prefix, String suffix) {
		return generate(prefix, suffix, new InferenceParameters());
	}

	/**
	 * Infill and stream outputs with custom inference parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @param prefix the prefix prompt of the completion to infill
	 * @param suffix the suffix prompt of the completion to infill
	 * @return iterable LLM outputs
	 */
	public Iterable<Output> generate(String prefix, String suffix, InferenceParameters parameters) {
		return () -> new LlamaIterator(prefix, suffix, parameters);
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
	private native void newAnswerIterator(String prompt, String parameters);
	private native void newInfillIterator(String prefix, String suffix, String parameters);
	private native Output getNext(LlamaIterator iterator);
	private native byte[] getAnswer(String prompt, String parameters);
	private native byte[] getInfill(String prefix, String suffix, String parameters);
	private native byte[] decodeBytes(int[] tokens);
	private native void delete();

	/**
	 * A generated output of the LLM. Note that you have to configure {@link InferenceParameters#setNPredict(int)}
	 * in order for probabilities to be returned.
	 * For multibyte outputs (unicode characters like emojis) only the last generated token and its probabilities
	 * are returned.
	 */
	public static final class Output {

		public final int token;
		@NotNull
		public final String text;
		@NotNull
		public final Map<Integer, Float> probabilities;

		private Output(int token, byte[] generated, @NotNull Map<Integer, Float> probabilities) {
			this.token = token;
			this.text = new String(generated, StandardCharsets.UTF_8);
			this.probabilities = probabilities;
		}

		@Override
		public String toString() {
			return text;
		}

	}

	// fields are modified by native code and thus should not be final
	@SuppressWarnings("FieldMayBeFinal")
	private final class LlamaIterator implements Iterator<Output> {

		@Native
		private boolean hasNext = true;
		@Native
		private long generatedCount = 0;
		@Native
		private long tokenIndex = 0;

		private LlamaIterator(String prompt, InferenceParameters parameters) {
			newAnswerIterator(prompt, parameters.toString());
		}

		private LlamaIterator(String prefix, String suffix, InferenceParameters parameters) {
			newInfillIterator(prefix, suffix, parameters.toString());
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
			return getNext(this);
		}
	}

}
