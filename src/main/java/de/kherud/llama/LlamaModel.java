package de.kherud.llama;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.ptr.FloatByReference;
import de.kherud.llama.foreign.LlamaLibrary;
import de.kherud.llama.foreign.NativeSize;
import de.kherud.llama.foreign.llama_context_params;
import de.kherud.llama.foreign.llama_token_data;
import de.kherud.llama.foreign.llama_token_data_array;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Iterator;


public class LlamaModel implements AutoCloseable {

	private final llama_context_params.ByValue params;
	private final LlamaLibrary.llama_model model;
	private final LlamaLibrary.llama_context ctx;
	private final int[] inputIds;
	private final float[] scores;
	private int nTokens = 0;

	private int nThreads = 1;
	private int nVocab;
	private int topK = 1;
	private byte sortedCandidates = 0;


	public LlamaModel(String filePath) {
		this(filePath, LlamaLibrary.llama_context_default_params());
	}

	public LlamaModel(String filePath, llama_context_params.ByValue params) {
		this.params = params;
		model = LlamaLibrary.llama_load_model_from_file(filePath, params);
		ctx = LlamaLibrary.llama_new_context_with_model(model, params);
		inputIds = new int[params.n_ctx];
		nVocab = getVocabularySize();
		scores = new float[params.n_ctx * nVocab];
	}

	public Iterator<String> generate(String prompt) {
		return new LlamaIterator(prompt);
	}

	public String complete(String prompt) {
		return "";
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

	void eval(IntBuffer tokens) throws RuntimeException {
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
			int rows, offset;
			if (params.logits_all > 0) {
				offset = 0;
				rows = nTokens;
			} else {
				offset = (nTokens - 1) * nVocab;
				rows = 1;
			}
			Pointer logitsPointer = LlamaLibrary.llama_get_logits(ctx).getPointer();
			float[] logits = logitsPointer.getFloatArray(0, rows * nVocab);
			System.arraycopy(logits, 0, scores, offset, logits.length);
		}
	}

	private void sample() {
		int topK = this.topK <= 0 ? nVocab : this.topK;
		IntBuffer tokens = IntBuffer.wrap(inputIds);
		llama_token_data[] tokenData = (llama_token_data[]) new llama_token_data().toArray(nVocab);
		for (int i = 0; i < nVocab; i++) {
			tokenData[i].setLogit(scores[scores.length - nVocab + i]);
		}
		llama_token_data_array llamaTokenDataArray = new llama_token_data_array();
		llamaTokenDataArray.setData((llama_token_data.ByReference) tokenData[0]);
		llamaTokenDataArray.setSize(new NativeSize(nVocab));
		llamaTokenDataArray.setSorted(sortedCandidates);
		LlamaLibrary.llama_sample_repetition_penalty(ctx, llamaTokenDataArray, tokens, new NativeSize(tokens.capacity()), 1);
	}

	private void setupCandidates() {
//		Structure[] tokenData = new llama_token_data().toArray(nVocab);
//		llama_token_data_array llamaTokenDataArray = new llama_token_data_array();
//		llamaTokenDataArray.setData((llama_token_data.ByReference) tokenData[0]);
//		llamaTokenDataArray.setSize(new NativeSize(nVocab));
//		llamaTokenDataArray.setSorted(sortedCandidates);
	}

	private class LlamaIterator implements Iterator<String> {

		private int token = -1;
		private int eosToken = LlamaLibrary.llama_token_eos();

		public LlamaIterator(String prompt) {

		}

		@Override
		public boolean hasNext() {
			return false;
		}

		@Override
		public String next() {
			return null;
		}
	}
}
