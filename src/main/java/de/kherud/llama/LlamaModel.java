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
     * Load a model with the given parameters.
     * 
     * @param params Command line-style parameters for model loading
     */
    public native void loadModel(String[] params);

    /**
     * Clean up resources and unload the model.
     */
    public native void delete();

    /**
     * Set a logger to receive log messages from the native library.
     * 
     * @param logFormat The format of log messages (JSON or TEXT)
     * @param callback Callback to receive log messages
     */
    public static native void setLogger(LogFormat logFormat, BiConsumer<LogLevel, String> callback);

    // Server Information Endpoints

    /**
     * Get the server health status.
     * Equivalent to GET /health endpoint.
     * 
     * @return JSON string with health information
     */
    public native String getHealth();

    /**
     * Get detailed server metrics.
     * Equivalent to GET /metrics endpoint.
     * 
     * @return JSON string with metrics information
     */
    public native String getMetrics();

    /**
     * Get model properties.
     * Equivalent to GET /props endpoint.
     * 
     * @return JSON string with model properties
     */
    public native String getProps();

    /**
     * Update model properties.
     * Equivalent to POST /props endpoint.
     * 
     * @param propsJson JSON string with properties to update
     */
    public native void updateProps(String propsJson);

    /**
     * Get the list of available models.
     * Equivalent to GET /models or GET /v1/models endpoints.
     * 
     * @return JSON string with model information
     */
    public native String getModels();

    /**
     * Get the current server state.
     * 
     * @return String indicating server state ("UNLOADED", "LOADING_MODEL", "READY")
     */
    public native String getServerState();

    // Text Generation Endpoints

    /**
     * Handle standard completions request.
     * Equivalent to POST /completions endpoint.
     * 
     * @param requestData JSON string with completion parameters
     * @param stream Whether to stream the results
     * @return JSON string with task information or completion results
     */
    public native String handleCompletions(String requestData, boolean stream);

    /**
     * Handle OpenAI compatible completions request.
     * Equivalent to POST /v1/completions endpoint.
     * 
     * @param requestData JSON string with OpenAI format completion parameters
     * @param stream Whether to stream the results
     * @return JSON string with task information or completion results in OpenAI format
     */
    public native String handleCompletionsOai(String requestData, boolean stream);

    /**
     * Handle chat completions request.
     * Equivalent to POST /chat/completions or POST /v1/chat/completions endpoints.
     * 
     * @param requestData JSON string with chat parameters
     * @param stream Whether to stream the results
     * @return JSON string with task information or chat completion results
     */
    public native String handleChatCompletions(String requestData, boolean stream);

    /**
     * Handle text infill request (completing text with given prefix and suffix).
     * Equivalent to POST /infill endpoint.
     * 
     * @param requestData JSON string with infill parameters
     * @param stream Whether to stream the results
     * @return JSON string with task information or infill results
     */
    public native String handleInfill(String requestData, boolean stream);

    /**
     * Get the next chunk of streaming results for a completion task.
     * 
     * @param taskId The ID of the task to get results for
     * @return JSON string with the next chunk of results
     */
    public native String getNextStreamResult(int taskId);

    /**
     * Release resources associated with a task.
     * 
     * @param taskId The ID of the task to release
     */
    public native void releaseTask(int taskId);

    /**
     * Cancel an ongoing completion.
     * 
     * @param taskId The ID of the task to cancel
     */
    public native void cancelCompletion(int taskId);

    // Embeddings and Reranking Endpoints

    /**
     * Handle embeddings request.
     * Equivalent to POST /embeddings endpoint.
     * 
     * @param requestData JSON string with embedding parameters
     * @param oaiCompat Whether to use OpenAI compatible format
     * @return JSON string with embedding results
     */
    public native String handleEmbeddings(String requestData, boolean oaiCompat);

    /**
     * Handle reranking request.
     * Equivalent to POST /rerank, POST /reranking, POST /v1/rerank, or POST /v1/reranking endpoints.
     * 
     * @param requestData JSON string with reranking parameters
     * @return JSON string with reranking results
     */
    public native String handleRerank(String requestData);

    // Tokenization Endpoints

    /**
     * Handle tokenization request.
     * Equivalent to POST /tokenize endpoint.
     * 
     * @param content The text to tokenize
     * @param addSpecial Whether to add special tokens
     * @param withPieces Whether to include token pieces in the response
     * @return JSON string with tokenization results
     */
    public native String handleTokenize(String content, boolean addSpecial, boolean withPieces);

    /**
     * Handle detokenization request.
     * Equivalent to POST /detokenize endpoint.
     * 
     * @param tokens Array of token IDs to detokenize
     * @return JSON string with detokenization results
     */
    public native String handleDetokenize(int[] tokens);

    /**
     * Apply a chat template to messages.
     * Equivalent to POST /apply-template endpoint.
     * 
     * @param requestData JSON string with template parameters
     * @return String with the template applied to the messages
     */
    public native String applyTemplate(String requestData);

    // LoRA Adapters Endpoints

    /**
     * Get the list of available LoRA adapters.
     * Equivalent to GET /lora-adapters endpoint.
     * 
     * @return JSON string with LoRA adapter information
     */
    public native String getLoraAdapters();

    /**
     * Apply LoRA adapters to the model.
     * Equivalent to POST /lora-adapters endpoint.
     * 
     * @param adaptersJson JSON string with LoRA adapter parameters
     * @return boolean indicating success
     */
    public native boolean applyLoraAdapters(String adaptersJson);

    // Slots Management Endpoints

    /**
     * Handle slot management operations.
     * Consolidates GET /slots and POST /slots/:id_slot endpoints.
     * 
     * @param action Action to perform: 0=GET (list), 1=SAVE, 2=RESTORE, 3=ERASE
     * @param slotId Slot ID (ignored for GET action)
     * @param filename Filename for save/restore (ignored for GET and ERASE actions)
     * @return JSON string with operation results
     */
    public native String handleSlotAction(int action, int slotId, String filename);

    // Constants for slot actions
    public static final int SLOT_ACTION_GET = 0;
    public static final int SLOT_ACTION_SAVE = 1;
    public static final int SLOT_ACTION_RESTORE = 2;
    public static final int SLOT_ACTION_ERASE = 3;
    // Utility Methods

    /**
     * Convert a JSON schema to a grammar.
     * 
     * @param schema JSON string with schema definition
     * @return Byte array with the grammar
     */
    public static native byte[] jsonSchemaToGrammarBytes(String schema);

	@Override
	public void close() throws Exception {
		delete();
		
	}
	
	/**
	 * Tokenize a prompt given the native tokenizer
	 *
	 * @param prompt the prompt to tokenize
	 * @return an array of integers each representing a token id
	 */
	public native int[] encode(String prompt);
}
