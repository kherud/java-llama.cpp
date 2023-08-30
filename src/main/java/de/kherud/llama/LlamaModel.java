package de.kherud.llama;

import com.sun.jna.Pointer;
import org.jetbrains.annotations.NotNull;

import de.kherud.llama.foreign.LlamaLibrary;
import de.kherud.llama.foreign.NativeSize;
import de.kherud.llama.foreign.llama_token_data;
import de.kherud.llama.foreign.llama_token_data_array;

import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;


public class LlamaModel implements AutoCloseable {

	private final Parameters params;
	private final LlamaLibrary.llama_model model;
	final LlamaLibrary.llama_context ctx;

	// cache some things for performance
	private final Pointer logitsPointer;
	private final IntBuffer tokenBuffer;
	private final byte[] tokenPieceBuffer;

	private final llama_token_data.ByReference[] candidateData;
	private final llama_token_data_array candidates;

	private final int nVocab;

	private final int tokenBos;
	private final int tokenEos;
	private final int tokenNl;

	private int nPast = 0;
	private int nPromptTokens = 0;
	private int nKeep = 0;
	private int nRemain = 0;
	boolean hasNextToken = false;
	private int nContext = 0;

	static {
		LlamaLibrary.llama_backend_init((byte) 0);
	}

	public LlamaModel(String filePath) {
		this(filePath, new Parameters.Builder().build());
	}

	public LlamaModel(String filePath, Parameters params) {
		// load the model
		this.params = params;
		model = LlamaLibrary.llama_load_model_from_file(filePath, params.ctx);
		if (model == null) {
			throw new RuntimeException("error: unable to load model");
		}
		ctx = LlamaLibrary.llama_new_context_with_model(model, params.ctx);

		// load lora adapter if configured
		if (params.loraAdapter != null
				&& params.loraBase != null
				&& LlamaLibrary.llama_model_apply_lora_from_file(model, params.loraAdapter, params.loraBase, params.nThreads) != 0) {
			throw new RuntimeException("error: unable to apply lora");
		}

		// setup some cached variables used throughout lifecycle
		tokenBuffer = IntBuffer.allocate(params.ctx.n_ctx);
		tokenPieceBuffer = new byte[64];
		nVocab = getVocabularySize();

		logitsPointer = LlamaLibrary.llama_get_logits(ctx).getPointer();

		candidateData = (llama_token_data.ByReference[]) new llama_token_data.ByReference().toArray(nVocab);
		candidates = new llama_token_data_array();

		tokenBos = LlamaLibrary.llama_token_bos(ctx);
		tokenEos = LlamaLibrary.llama_token_eos(ctx);
		tokenNl = LlamaLibrary.llama_token_nl(ctx);

		// do one empty run to warm up the model (taken from main.cpp)
//		warmup();
	}

	public Iterable<Output> generate(String prompt) {
		LlamaLibrary.llama_reset_timings(ctx);
		return new Iterable<>() {
			@NotNull
			@Override
			public Iterator<Output> iterator() {
				return new LlamaIterator(prompt);
			}
		};
	}

	public float[] getEmbedding() {
		return null;
	}

	/**
	 * Tokenize a prompt given the native tokenizer
	 *
	 * @param prompt the prompt to tokenize
	 * @return an array of integers each representing a token id (see {@link #getVocabularySize()})
	 */
	public int[] encode(String prompt) {
		// IntBuffer tokens = tokenize(prompt);
		// // return tokens without padding
		// for (int i = 0; i < tokens.capacity(); i++) {
		// 	if (tokens.get(i) == 0) {
		// 		return tokens.slice(0, i).array();
		// 	}
		// }
		// return tokens.array();
		return null;
	}

	/**
	 * Convert an array of token ids to its string representation
	 *
	 * @param tokens an array of tokens conforming to {@link #getVocabularySize()}
	 * @return the token ids decoded to a string
	 */
	public String decode(int[] tokens) {
		StringBuilder builder = new StringBuilder();
		for (int token : tokens) {
			String decoded = LlamaLibrary.llama_token_get_text(ctx, token);
			builder.append(decoded);
		}
		return builder.toString();
	}

	/**
	 * Returns the context size of the loaded model, e.g., how many tokens the LLM can process at most in one request.
	 * E.g., for llama 1 this size is 2048.
	 *
	 * @return the context size of the loaded model
	 */
	public int getContextSize() {
		return LlamaLibrary.llama_n_ctx(ctx);
	}

	/**
	 * Returns the hidden dimensionality of the loaded model, which corresponds to the size of {@link #getEmbedding()}.
	 * This size typically depends on the amount of parameters in the model, e.g., for llama 2 13b this size is 5120.
	 *
	 * @return the amount of embedding dimensions
	 */
	public int getEmbeddingSize() {
		return LlamaLibrary.llama_n_embd(ctx);
	}

	/**
	 * Returns the total amount of tokens in the vocabulary.
	 *
	 * @return the vocabulary size
	 */
	public int getVocabularySize() {
		return LlamaLibrary.llama_n_vocab(ctx);
	}


	void tokenize(String prompt) {
		// Add a space in front of the first character to match OG llama tokenizer behavior (taken from main.cpp)
		if (nContext == 0 && !prompt.startsWith(" ")) {
			prompt = " " + prompt;
		}

		nPromptTokens = LlamaLibrary.llama_tokenize(ctx, prompt, tokenBuffer, params.ctx.n_ctx, (byte) 1);
		if (nPromptTokens < 0) {
			throw new RuntimeException("tokenization failed due to unknown reasons");
		}
		nKeep = params.nKeep < 0 ? nPromptTokens : params.nKeep;
		nKeep = Math.min(params.ctx.n_ctx - 4, nKeep);
		if (nPromptTokens >= params.ctx.n_ctx) {
			int nLeft = (params.ctx.n_ctx - nKeep) / 2;

			int erasedBlocks = (nPromptTokens - nKeep - nLeft - 1) / nLeft;
		}
		nContext = nPromptTokens;
	}

	private void logPrompt(String prompt, IntBuffer tokens) {
		if (params.verbosePrompt) {
			StringBuilder msgBuilder = new StringBuilder();
			msgBuilder.append("prompt: '")
					.append(prompt)
					.append("'\nnumber of tokens in prompt =")
					.append(tokens.capacity())
					.append("\n");
			for (int i = 0; i < tokens.capacity(); i++) {
				int tokenId = tokens.get(i);
				String tokenStr = LlamaLibrary.llama_token_get_text(ctx, tokenId);
				String msg = String.format("%6d -> '%s'", tokens.get(i), tokenStr);
				msgBuilder.append(msg);
			}
			params.logCallback.accept(LlamaLibrary.llama_log_level.LLAMA_LOG_LEVEL_INFO, msgBuilder.toString());
			/*
			if (ctx_guidance) {
				fprintf(stderr, "\n");
				fprintf(stderr, "%s: negative prompt: '%s'\n", __func__, params.cfg_negative_prompt.c_str());
				fprintf(stderr, "%s: number of tokens in negative prompt = %zu\n", __func__, guidance_inp.size());
				for (int i = 0; i < (int) guidance_inp.size(); i++) {
					fprintf(stderr, "%6d -> '%s'\n", guidance_inp[i], llama_token_to_str(ctx, guidance_inp[i]));
				}
			}

			if (params.n_keep > 0) {
				fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
				for (int i = 0; i < params.n_keep; i++) {
					fprintf(stderr, "%s", llama_token_to_str(ctx, embd_inp[i]));
				}
				fprintf(stderr, "'\n");
			}
			fprintf(stderr, "\n");
			 */
		}
	}

	Output nextToken() {
		Output result;

		if (tokenBuffer.capacity() >= params.ctx.n_ctx) {
			// ...
		}

		while (nPast < nContext) {
			int nEval = nContext - nPast;
			if (nEval > params.ctx.n_batch) {
				nEval = params.ctx.n_batch;
			}
//			System.out.printf("eval %d to %d\n", nPast, nPast + nEval);
			if (LlamaLibrary.llama_eval(ctx, tokenBuffer.slice(nPast, nEval), nEval, nPast, params.nThreads) != 0) {
				String msg = String.format("evaluation failed (%d to evaluate, %d past, %d threads)", nEval, nPast, params.nThreads);
				params.logCallback.accept(LlamaLibrary.llama_log_level.LLAMA_LOG_LEVEL_ERROR, msg);
				hasNextToken = false;
				return new Output(-1, 0);
			}
			nPast += nEval;
		}

		if (params.nPredict == 0) {
			hasNextToken = false;
			return new Output(tokenEos, 0);
		}

		result = sample();
		nRemain--;

		if (result.token == tokenEos) {
			hasNextToken = false;
			return result;
		}
		tokenBuffer.put(nContext, result.token);
		nContext++;

		hasNextToken = params.nPredict == -1 || nRemain > 0;
		return result;
	}

	@Override
	public void close() {
		LlamaLibrary.llama_free_model(model);
		LlamaLibrary.llama_free(ctx);
	}

//	void eval(IntBuffer tokens) throws RuntimeException {
//		int nTokens = tokens.capacity();
//		for (int b = 0; b < nTokens; b += params.ctx.n_batch) {
//			applyLogitBias(logitsPointer);
//			int batchEnd = Math.min(nTokens, b + params.ctx.n_batch);
//			IntBuffer batch = tokens.slice(b, batchEnd);
//			int nBatch = batchEnd - b;
//			int nPast = Math.min(params.ctx.n_ctx - nBatch, inputIds.length);
//			int result = LlamaLibrary.llama_eval(ctx, batch, nBatch, nPast, params.nThreads);
//			if (result != 0) {
//				throw new RuntimeException("llama_eval returned " + result);
//			}
//			for (int i = this.nTokens; i < this.nTokens + nBatch; i++) {
//				inputIds[i] = batch.get(i);
//			}
//			int rows, offset;
//			if (params.ctx.logits_all > 0) {
//				offset = 0;
//				rows = nBatch;
//			} else {
//				offset = nBatch - 1;
//				rows = 1;
//			}
//
//			// Save only the last token logits if logits_all is false
//			float[] logits = logitsPointer.getFloatArray(0, rows * nVocab);
//			int destPos = (this.nTokens + offset) * nVocab;
//			System.arraycopy(logits, 0, scores, destPos, nBatch);
//			this.nTokens += nBatch;
//		}
//	}

	// todo: should be private
	Output sample() {
		float[] logits = LlamaLibrary.llama_get_logits(ctx).getPointer().getFloatArray(0, nVocab);
		// I'm not sure why anything but `setLogit` has to be called here for `candidateData` and `candidates` again.
		// Otherwise, the results are garbage, however. Maybe the JVM/JIT is moving stuff around.
		for (int i = 0; i < nVocab; i++) {
			candidateData[i].setId(i);
			candidateData[i].setLogit(logits[i]);
//			candidateData[i].setP(0f);
		}
		candidates.setData(candidateData[0]);
		candidates.setSize(new NativeSize(nVocab));
		candidates.setSorted((byte) 0);

		samplePenalty();


		if (params.grammar != null) {
			// todo: how to create a native grammar object? How to obtain grammar rules etc?
			// LlamaLibrary.llama_sample_grammar(ctx, candidates, params.grammar);
			throw new IllegalStateException("grammar not yet supported");
		}

		int token;
		if (params.temperature == 0) {
			token = sampleGreedy();
		} else if (params.mirostat == Parameters.MiroStat.V1) {
			token = sampleMirostatV1();
		} else if (params.mirostat == Parameters.MiroStat.V2) {
			token = sampleMirostatV2();
		} else {
			token = sampleTopK();
		}

		if (params.grammar != null) {
			// todo: see above, not yet implemented
			// LlamaLibrary.llama_grammar_accept_token(ctx, params.grammar, token);
			throw new IllegalStateException("grammar not yet implemented");
		}

		return new Output(token, candidateData[token].p);
	}

	private void samplePenalty() {
//		float nlLogit = scores[scores.length - nVocab + tokenNl];
		int repeat_last_n = params.repeatLastN < 0 ? params.ctx.n_ctx : params.repeatLastN;
		int last_n_repeat = Math.min(Math.min(nContext, repeat_last_n), params.ctx.n_ctx);
		NativeSize nTokens = new NativeSize(last_n_repeat);
		IntBuffer lastTokens = tokenBuffer.slice(nContext - last_n_repeat, last_n_repeat);
		LlamaLibrary.llama_sample_repetition_penalty(
				ctx,
				candidates,
				lastTokens,
				nTokens,
				params.repeatPenalty
		);
		LlamaLibrary.llama_sample_frequency_and_presence_penalties(
				ctx,
				candidates,
				lastTokens,
				nTokens,
				params.frequencyPenalty,
				params.presencePenalty
		);
//		if (!params.penalizeNl) {
//			token_data[nlOffset].setLogit(nlLogit);
//		}
	}

	private int sampleGreedy() {
		int token = LlamaLibrary.llama_sample_token_greedy(ctx, candidates);
		if (params.nProbs > 0) {
			LlamaLibrary.llama_sample_softmax(ctx, candidates);
		}
		return token;
	}

	private int sampleMirostatV1() {
		LlamaLibrary.llama_sample_temperature(ctx, candidates, params.temperature);
		return LlamaLibrary.llama_sample_token_mirostat(
				ctx,
				candidates,
				params.mirostatTau,
				params.mirostatEta,
				params.mirostatM,
				params.mirostatMu
		);
	}

	private int sampleMirostatV2() {
		LlamaLibrary.llama_sample_temperature(ctx, candidates, params.temperature);
		return LlamaLibrary.llama_sample_token_mirostat_v2(
				ctx,
				candidates,
				params.mirostatTau,
				params.mirostatEta,
				params.mirostatMu
		);
	}

	private int sampleTopK() {
		NativeSize minKeep = new NativeSize(Math.max(1, params.nProbs));
		LlamaLibrary.llama_sample_top_k(
				ctx,
				candidates,
				params.topK <= 0 ? nVocab : params.topK,
				minKeep
		);
		LlamaLibrary.llama_sample_tail_free(
				ctx,
				candidates,
				params.tfsZ,
				minKeep
		);
		LlamaLibrary.llama_sample_typical(
				ctx,
				candidates,
				params.typicalP,
				minKeep
		);
		LlamaLibrary.llama_sample_top_p(ctx,
				candidates,
				params.topP,
				minKeep
		);
		LlamaLibrary.llama_sample_temperature(
				ctx,
				candidates,
				params.temperature
		);
		return LlamaLibrary.llama_sample_token(ctx, candidates);
	}

	private void validateParams() {
		if (params.ctx.rope_freq_base != 10000.0) {
			System.out.printf("warning: changing RoPE frequency base to %g (default 10000.0)\n", params.ctx.rope_freq_base);
		}

		if (params.ctx.rope_freq_scale != 1.0) {
			System.out.printf("warning: scaling RoPE frequency by %g (default 1.0)\n", params.ctx.rope_freq_scale);
		}

		if (params.ctx.n_ctx > 2048) {
			System.out.printf("warning: base model only supports context sizes no greater than 2048 tokens (%d specified)\n", params.ctx.n_ctx);
		} else if (params.ctx.n_ctx < 8) {
			System.out.println("warning: minimum context size is 8, using minimum size.");
			params.ctx.setN_ctx(8);
		}
	}

	private void predict(IntBuffer tokens) {
		// Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
		// --prompt or --file which uses the same value.
		int nTokens = tokens.capacity();
		int maxEmbedSize = params.ctx.n_ctx - 4;
		// Ensure the input doesn't exceed the context size by truncating the input if necessary.
		if (nTokens > maxEmbedSize) {
			int skipTokens = nTokens - params.ctx.n_ctx;
			String msg = String.format("<<input too long: skipped %d token%s>>", skipTokens, skipTokens == 1 ? "" : "s");
			params.logCallback.accept(LlamaLibrary.llama_log_level.LLAMA_LOG_LEVEL_WARN, msg);
			tokens = tokens.slice(0, maxEmbedSize);
		}

		// infinite text generation via context swapping
		// if we run out of context:
		// - take the n_keep first tokens from the original prompt (via n_past)
		// - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches

	}

	private void applyLogitBias(Pointer logits) {
		params.logitBias.forEach((id, bias) -> logits.setFloat(id * Float.BYTES, bias));
	}

	private void warmup() {
		IntBuffer intBuffer = IntBuffer.wrap(new int[]{tokenBos});
//		eval(intBuffer);
		LlamaLibrary.llama_reset_timings(ctx);
	}

	private class LlamaIterator implements Iterator<Output> {

		private IntBuffer tokens;
		private int nPast = 0;
		private int nRemain = params.nPredict;

		public LlamaIterator(String prompt) {

		}

		@Override
		public boolean hasNext() {
			return nRemain != 0 && !isAntiprompt();
		}

		@Override
		public Output next() {
			return null;
		}

		private void setup() {
			int maxTokens;
			if (params.nPredict < 0) {
				maxTokens = params.ctx.n_ctx - tokens.capacity();
			}

//			stopTokens = new int[1 + params.antiprompt.size()];
//			stopTokens[0] = LlamaLibrary.llama_token_eos();


		}

		private boolean isAntiprompt() {
			return false;
		}
	}

	public final class Output {
		public final int token;
		public final float probability;

		private Output(int token, float probability) {
			this.token = token;
			this.probability = probability;
		}

		@Override
		public String toString() {
			int size = LlamaLibrary.llama_token_to_piece(ctx, token, tokenPieceBuffer, tokenPieceBuffer.length);
			return new String(tokenPieceBuffer, 0, size, StandardCharsets.UTF_8);
		}
	}
}
