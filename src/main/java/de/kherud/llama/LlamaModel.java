package de.kherud.llama;

import com.sun.jna.ptr.FloatByReference;
import de.kherud.llama.foreign.LlamaLibrary;
import de.kherud.llama.foreign.llama_context_params;

import java.nio.IntBuffer;
import java.util.Iterator;


public class LlamaModel implements AutoCloseable {

	private final llama_context_params.ByValue params;
	private final LlamaLibrary.llama_model model;
	private final LlamaLibrary.llama_context ctx;
	private final int[] inputIds;
	private final int[][] scores;
	private int nTokens = 0;

	private int nThreads = 1;
	private int nVocab;


	public LlamaModel(String filePath) {
		this(filePath, LlamaLibrary.llama_context_default_params());
	}

	public LlamaModel(String filePath, llama_context_params.ByValue params) {
		this.params = params;
		model = LlamaLibrary.llama_load_model_from_file(filePath, params);
		ctx = LlamaLibrary.llama_new_context_with_model(model, params);
		inputIds = new int[params.n_ctx];
		nVocab = getVocabularySize();
		scores = new int[params.n_ctx][nVocab];
	}

	public int[] encode(String prompt) {
		return tokenize(prompt).array();
	}

	public String decode(int[] tokens) {
		StringBuilder builder = new StringBuilder();
		for (int token : tokens) {
			String decoded = LlamaLibrary.llama_token_to_str(ctx, token);
			builder.append(decoded);
		}
		return builder.toString();
	}

	public IntBuffer tokenize(String prompt) {
		IntBuffer tokens = IntBuffer.allocate(params.n_ctx);
		int nTokens = LlamaLibrary.llama_tokenize(ctx, prompt, tokens, params.n_ctx, (byte) 1);
		if (nTokens < 0) {
			throw new RuntimeException("tokenization failed");
		}
		return tokens.slice(0, nTokens);
	}

	public int getContextSize() {
		return LlamaLibrary.llama_n_ctx(ctx);
	}

	public int getEmbeddingSize() {
		return LlamaLibrary.llama_n_embd(ctx);
	}

	public int getVocabularySize() {
		return LlamaLibrary.llama_n_vocab(ctx);
	}

	@Override
	public void close() throws Exception {
		LlamaLibrary.llama_free_model(model);
		LlamaLibrary.llama_free(ctx);
	}

	public void forward(IntBuffer tokens) throws RuntimeException {
		int nTokens = tokens.capacity();
		for (int b = 0; b < nTokens; b += params.n_batch) {
			int batchEnd = Math.min(nTokens, b + params.n_batch);
			IntBuffer batch = tokens.slice(b, batchEnd);
			int batchSize = batchEnd - b;
			int nPast = Math.min(params.n_ctx - batchSize, inputIds.length);
			int result = LlamaLibrary.llama_eval(ctx, batch, nTokens, nPast, nThreads);
			if (result != 0) {
				throw new RuntimeException("llama_eval returned " + result);
			}
			for (int i = this.nTokens; i < this.nTokens + nTokens; i++) {
				inputIds[i] = batch.get(i);
			}
//			int offset = params.logits_all > 0 ? 0 : 1;
//			FloatByReference logits = LlamaLibrary.llama_get_logits(ctx);
//			logits.
//			for (int i = this.nTokens + offset; i < this.nTokens + nTokens; i++) {
//				for (int j = 0; j < nVocab; j++) {
//					scores[i][j];
//				}
//			}
		}
	}

//	private class LlamaIterator implements Iterator<String> {
//
//		public LlamaIterator(String prompt) {
//			MemorySegment promptSegment = arena.allocateUtf8String(prompt);
//			MemorySegment tokenized = tokenize(promptSegment);
//		}
//
//
//		@Override
//		public boolean hasNext() {
//			return false;
//		}
//
//		@Override
//		public String next() {
//			return null;
//		}
//	}
}
