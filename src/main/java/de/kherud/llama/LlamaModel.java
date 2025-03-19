package de.kherud.llama;

import de.kherud.llama.args.LogFormat;
import org.jetbrains.annotations.Nullable;

import java.lang.annotation.Native;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * This class is a wrapper around the llama.cpp functionality.
 * Upon being created, it natively allocates memory for the model context.
 * Thus, this class is an {@link AutoCloseable}, in order to de-allocate the memory when it is no longer being needed.
 * <p>
 * The main functionality of this class is:
 * <ul>
 *     <li>Streaming answers (and probabilities) via {@link #generate(InferenceParameters)}</li>
 *     <li>Creating whole responses to prompts via {@link #complete(InferenceParameters)}</li>
 *     <li>Creating embeddings via {@link #embed(String)} (make sure to configure {@link ModelParameters#enableEmbedding()}</li>
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
	 *     <li>{@link ModelParameters#setModel(String)}</li>
	 *     <li>{@link ModelParameters#setModelUrl(String)}</li>
	 *     <li>{@link ModelParameters#setHfRepo(String)}, {@link ModelParameters#setHfFile(String)}</li>
	 * </ul>
	 *
	 * @param parameters the set of options
	 * @throws LlamaException if no model could be loaded from the given file path
	 */
	public LlamaModel(ModelParameters parameters) {
		loadModel(parameters.toArray());
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
		LlamaOutput output = receiveCompletion(taskId);
		return output.text;
	}
	
	/**
	 * Generate and return a whole answer with custom parameters. 
	 * Please remember this will apply template and will only look at messages 
	 *
	 * @return an LLM response
	 */
	public String completeChat(InferenceParameters parameters) {
		parameters.setStream(false);
		
		int taskId = requestChat(parameters.toString());
		String output = receiveChatCompletion(taskId);
		return output;
	}

	/**
	 * Generate and stream outputs with custom inference parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @return iterable LLM outputs
	 */
	public LlamaIterable generate(InferenceParameters parameters) {
		return () -> new LlamaIterator(this, parameters);
	}
	
	/**
	 * Generate and stream outputs with custom inference parameters.
	 * Please remember this will apply template and will only look at messages
	 * @return iterable LLM outputs
	 */
	public LlamaIterable generateChat(InferenceParameters parameters) {
		String prompt = applyTemplate(parameters);
		parameters.setPrompt(prompt);
		return () -> new LlamaIterator(this, parameters);
	}
	
	
    
	/**
	 * Get the embedding of a string. Note, that the prompt isn't preprocessed in any way, nothing like
	 * "User: ", "###Instruction", etc. is added.
	 *
	 * @param prompt the string to embed
	 * @return an embedding float array
	 * @throws IllegalStateException if embedding mode was not activated (see {@link ModelParameters#enableEmbedding()})
	 */
	public  native float[] embed(String prompt);
		

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
	public String decode(int[] tokens) {
		byte[] bytes = decodeBytes(tokens);
		return new String(bytes, StandardCharsets.UTF_8);
	}

	/**
	 * Sets a callback for native llama.cpp log messages.
	 * Per default, log messages are written in JSON to stdout. Note, that in text mode the callback will be also
	 * invoked with log messages of the GGML backend, while JSON mode can only access request log messages.
	 * In JSON mode, GGML messages will still be written to stdout.
	 * To only change the log format but keep logging to stdout, the given callback can be <code>null</code>.
	 * To disable logging, pass an empty callback, i.e., <code>(level, msg) -> {}</code>.
	 *
	 * @param format the log format to use
	 * @param callback a method to call for log messages
	 */
	public static native void setLogger(LogFormat format, @Nullable BiConsumer<LogLevel, String> callback);

	@Override
	public void close() {
		delete();
	}

	// don't overload native methods since the C++ function names get nasty
	native int requestCompletion(String params) throws LlamaException;
	
	native int requestChat(String params) throws LlamaException;

	native LlamaOutput receiveCompletion(int taskId) throws LlamaException;
	
	native String receiveChatCompletion(int taskId) throws LlamaException;
	
	native void cancelCompletion(int taskId);

	native byte[] decodeBytes(int[] tokens);

	private native void loadModel(String... parameters) throws LlamaException;

	private native void delete();
	
	native void releaseTask(int taskId);

	private static native byte[] jsonSchemaToGrammarBytes(String schema);
	
	public static String jsonSchemaToGrammar(String schema) {
		return new String(jsonSchemaToGrammarBytes(schema), StandardCharsets.UTF_8);
	}
	
	public List<Pair<String, Float>> rerank(boolean reRank, String query, String ... documents) {
		LlamaOutput output = rerank(query, documents);
		
		Map<String, Float> scoredDocumentMap = output.probabilities;
		
		List<Pair<String, Float>> rankedDocuments = new ArrayList<>();
		
		if (reRank) {
            // Sort in descending order based on Float values
            scoredDocumentMap.entrySet()
                    .stream()
                    .sorted((a, b) -> Float.compare(b.getValue(), a.getValue())) // Descending order
                    .forEach(entry -> rankedDocuments.add(new Pair<>(entry.getKey(), entry.getValue())));
        } else {
            // Copy without sorting
            scoredDocumentMap.forEach((key, value) -> rankedDocuments.add(new Pair<>(key, value)));
        }
		
		return rankedDocuments;
	}
	
	public native LlamaOutput rerank(String query, String... documents);
	
	public  String applyTemplate(InferenceParameters parameters) {
		return applyTemplate(parameters.toString());
	}
	public native String applyTemplate(String parametersJson);
}
