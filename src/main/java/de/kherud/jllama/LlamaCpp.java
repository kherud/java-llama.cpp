package de.kherud.jllama;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.ByteByReference;

public interface LlamaCpp extends Library {

	LlamaCpp INSTANCE = Native.load("llama", LlamaCpp.class);

	// Set callback for all future logging events.
	// If this is not called, or NULL is supplied, everything is output on stderr.
	void llama_log_set(LogCallback log_callback, Pointer user_data);

	int llama_max_devices();

	ContextParameters llama_context_default_params();

	QuantizationParameters llama_model_quantize_default_params();

	byte llama_mmap_supported();

	byte llama_mlock_supported();

	// Initialize the llama + ggml backend
	// If numa is true, use NUMA optimizations
	// Call once at the start of the program
	void llama_backend_init(byte numa);

	// Call once at the end of the program - currently only used for MPI
	void llama_backend_free();

	long llama_time_us();

	Pointer llama_load_model_from_file(String path_model, ContextParameters params);

	void llama_free_model(Pointer model);

	Pointer llama_new_context_with_model(Pointer model, ContextParameters params);

	// Various functions for loading a ggml llama model.
	// Allocate (almost) all memory needed for the model.
	// Return NULL on failure
//	DEPRECATED(Pointer llama_init_from_file(
//			String path_model,
//						 struct llama_context_params   params),
//            "please use llama_load_model_from_file combined with llama_new_context_with_model instead");

	// Frees all allocated memory
	void llama_free(Pointer ctx);

	// Returns 0 on success
	int llama_model_quantize(String fname_inp, String fname_out, QuantizationParameters.ByReference params);

	// Apply a LoRA adapter to a loaded model
	// path_base_model is the path to a higher quality model to use as a base for
	// the layers modified by the adapter. Can be NULL to use the current loaded model.
	// The model needs to be reloaded before applying a new adapter, otherwise the adapter
	// will be applied on top of the previous one
	// Returns 0 on success
//	DEPRECATED(int llama_apply_lora_from_file(
//			Pointer ctx,
//                      String path_lora,
//                      String path_base_model,
//						 int   n_threads),
//            "please use llama_model_apply_lora_from_file instead");

	int llama_model_apply_lora_from_file(Pointer model, String path_lora, String path_base_model, int n_threads);

	// Returns the number of tokens in the KV cache
	int llama_get_kv_cache_token_count(Pointer ctx);

	// Sets the current rng seed.
	void llama_set_rng_seed(Pointer ctx, int seed);

	// Returns the maximum size in bytes of the state (rng, logits, embedding
	// and kv_cache) - will often be smaller after compacting tokens
	NativeSize llama_get_state_size(Pointer ctx);

	// Copies the state to the specified destination address.
	// Destination needs to have allocated enough memory.
	// Returns the number of bytes copied
	// Note that the original data type for "src" was uint8_t, and here is int8
	NativeSize llama_copy_state_data(Pointer ctx, ByteByReference dst);

	// Set the state reading from the specified address
	// Returns the number of bytes read
	// Note that the original data type for "src" was uint8_t, and here is int8
	NativeSize llama_set_state_data(Pointer ctx, ByteByReference src);

	// Save/load session file
	byte llama_load_session_file(Pointer ctx, String path_session, int[] tokens_out, NativeSize n_token_capacity, NativeSizeByReference n_token_count_out);

	byte llama_save_session_file(Pointer ctx, String path_session, int[] tokens, NativeSize n_token_count);

	// Run the llama inference to obtain the logits and probabilities for the next token.
	// tokens + n_tokens is the provided batch of new tokens to process
	// n_past is the number of tokens to use from previous eval calls
	// Returns 0 on success
	int llama_eval(Pointer ctx,int[] tokens,int n_tokens,int n_past,int n_threads);

	// Same as llama_eval, but use float matrix input directly.
	int llama_eval_embd(Pointer ctx,float[] embd,int n_tokens,int n_past,int n_threads);

	// Export a static computation graph for context of 511 and batch size of 1
	// NOTE: since this functionality is mostly for debugging and demonstration purposes, we hard-code these
	//       parameters here to keep things simple
	// IMPORTANT: do not use for anything else other than debugging and testing!
	int llama_eval_export(Pointer ctx, String fname);

	// Convert the provided text into tokens.
	// The tokens pointer must be large enough to hold the resulting tokens.
	// Returns the number of tokens on success, no more than n_max_tokens
	// Returns a negative number on failure - the number of tokens that would have been returned
	int llama_tokenize(Pointer ctx,String text,int[] tokens,int n_max_tokens,byte add_bos);

	int llama_tokenize_with_model(Pointer model,String text,int[] tokens,int n_max_tokens,byte add_bos);

	int llama_n_vocab(Pointer ctx);

	int llama_n_ctx(Pointer ctx);

	int llama_n_embd(Pointer ctx);

	int llama_n_vocab_from_model(Pointer model);

	int llama_n_ctx_from_model(Pointer model);

	int llama_n_embd_from_model(Pointer model);

	// Get the vocabulary as output parameters.
	// Returns number of results.
	int llama_get_vocab(Pointer ctx,String[] strings,float[] scores,int capacity);

	int llama_get_vocab_from_model(Pointer model,String[] strings,float[] scores,int capacity);

	// Token logits obtained from the last call to llama_eval()
	// The logits for the last token are stored in the last row
	// Can be mutated in order to change the probabilities of the next token
	// Rows: n_tokens
	// Cols: n_vocab
	float[] llama_get_logits(Pointer ctx);

	// Get the embeddings for the input
	// shape: [n_embd] (1-dimensional)
	float[] llama_get_embeddings(Pointer ctx);

	// Token Id -> String. Uses the vocabulary in the provided context
	String llama_token_to_str(Pointer ctx, int token);

	String llama_token_to_str_with_model(Pointer model, int token);

	// Special tokens
	int llama_token_bos();  // beginning-of-sentence

	int llama_token_eos();  // end-of-sentence

	int llama_token_nl();   // next-line

	// Grammar
	//
	Grammar.ByReference llama_grammar_init(GrammarElement.ByReference rules, NativeSize n_rules,NativeSize start_rule_index);

	void llama_grammar_free(GrammarElement.ByReference grammar);

	// Sampling functions

	/// @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
	void llama_sample_repetition_penalty(Pointer ctx, TokenDataArray.ByReference candidates, int[] last_tokens, NativeSize last_tokens_size, float penalty);

	/// @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
	void llama_sample_frequency_and_presence_penalties(Pointer ctx, TokenDataArray.ByReference candidates, int[] last_tokens, NativeSize last_tokens_size, float alpha_frequency, float alpha_presence);

	/// @details Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806
	/// @param candidates A vector of `llama_token_data` containing the candidate tokens, the logits must be directly extracted from the original generation context without being sorted.
	/// @params guidance_ctx A separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
	/// @params scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.
	void llama_sample_classifier_free_guidance(Pointer ctx,TokenDataArray candidates,Pointer guidance_ctx,float scale);

	/// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
	void llama_sample_softmax(Pointer ctx, TokenDataArray.ByReference candidates);

	/// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
	void llama_sample_top_k(Pointer ctx, TokenDataArray.ByReference candidates, int k, NativeSize min_keep);

	/// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
	void llama_sample_top_p(Pointer ctx, TokenDataArray.ByReference candidates, float p, NativeSize min_keep);

	/// @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
	void llama_sample_tail_free(Pointer ctx, TokenDataArray.ByReference candidates, float z, NativeSize min_keep);

	/// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
	void llama_sample_typical(Pointer ctx, TokenDataArray.ByReference candidates, float p, NativeSize min_keep);

	void llama_sample_temperature(Pointer ctx, TokenDataArray.ByReference candidates, float temp);

	/// @details Apply constraints from grammar
	void llama_sample_grammar(Pointer ctx, TokenDataArray.ByReference candidates, Grammar.ByReference grammar);

	/// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
	/// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
	/// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
	/// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
	/// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
	/// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
	int llama_sample_token_mirostat(Pointer ctx, TokenDataArray.ByReference candidates, float tau, float eta, int m, float[] mu);

	/// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
	/// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
	/// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
	/// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
	/// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
	int llama_sample_token_mirostat_v2(Pointer ctx, TokenDataArray.ByReference candidates, float tau, float eta, float[] mu);

	/// @details Selects the token with the highest probability.
	int llama_sample_token_greedy(Pointer ctx, TokenDataArray.ByReference candidates);

	/// @details Randomly selects a token from the candidates based on their probabilities.
	int llama_sample_token(Pointer ctx, TokenDataArray.ByReference candidates);

	/// @details Accepts the sampled token into the grammar
	void llama_grammar_accept_token(Pointer ctx, Grammar.ByReference grammar, int token);

	// Performance information
	Timings	llama_get_timings(Pointer ctx);

	void llama_print_timings(Pointer ctx);

	void llama_reset_timings(Pointer ctx);

	// Print system information
	String llama_print_system_info();
}
