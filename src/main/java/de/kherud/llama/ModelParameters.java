package de.kherud.llama;

import org.jetbrains.annotations.Nullable;

/**
 * Parameters used for initializing a {@link LlamaModel}.
 */
public final class ModelParameters {

	public final int nThreads;

	public final int seed;
	public final int nCtx; // text context
	public final int nBatch; // prompt processing batch size
	public final int nGpuLayers; // number of layers to store in VRAM
	public final int mainGpu; // the GPU that is used for scratch and small tensors
	public final float[] tensorSplit; // how to split layers across multiple GPUs (size: LLAMA_MAX_DEVICES)
	public final float ropeFreqBase; // RoPE base frequency
	public final float ropeFreqScale; // RoPE frequency scaling factor
	//	public final llama_progress_callback progress_callback;
//	public final Pointer progress_callback_user_data;
	public final boolean lowVram; // if true, reduce VRAM usage at the cost of performance
	public final boolean mulMatQ; // if true, use experimental mul_mat_q kernels
	public final boolean f16Kv; // use fp16 for KV cache
	public final boolean logitsAll; // the llama_eval() call computes all logits, not just the last one
	public final boolean vocabOnly; // only load the vocabulary, no weights
	public final boolean useMmap; // use mmap if possible
	public final boolean useMlock; // force system to keep model in RAM
	public final boolean embedding; // embedding mode only
	@Nullable
	public final String loraAdapter;  // lora adapter path
	@Nullable
	public final String loraBase;  // base model path for the lora adapter
	public final boolean hellaswag; // compute HellaSwag score over random tasks from datafile supplied in prompt
	public final short hellaswagTasks;   // number of tasks to use when computing the HellaSwag score
	public final boolean memoryF16;  // use f16 instead of f32 for memory kv
	public final boolean memTest; // compute maximum memory usage
	public final boolean numa; // attempt optimizations that help on some NUMA systems
	public final boolean verbosePrompt; // log prompt tokens before generation

	/**
	 * Private constructor to build immutable parameters object. Called via {@link Builder}.
	 */
	private ModelParameters(
			int nThreads,
			int seed,
			int nCtx,
			int nBatch,
			int nGpuLayers,
			int mainGpu,
			float[] tensorSplit,
			float ropeFreqBase,
			float ropeFreqScale,
			boolean lowVram,
			boolean mulMatQ,
			boolean f16Kv,
			boolean logitsAll,
			boolean vocabOnly,
			boolean useMmap,
			boolean useMlock,
			boolean embedding,
			@Nullable String loraAdapter,
			@Nullable String loraBase,
			boolean hellaswag,
			short hellaswagTasks,
			boolean memoryF16,
			boolean memTest,
			boolean numa,
			boolean verbosePrompt
	) {
		this.seed = seed;
		this.nCtx = nCtx;
		this.nBatch = nBatch;
		this.nGpuLayers = nGpuLayers;
		this.mainGpu = mainGpu;
		this.tensorSplit = tensorSplit;
		this.ropeFreqBase = ropeFreqBase;
		this.ropeFreqScale = ropeFreqScale;
		this.lowVram = lowVram;
		this.mulMatQ = mulMatQ;
		this.f16Kv = f16Kv;
		this.logitsAll = logitsAll;
		this.vocabOnly = vocabOnly;
		this.useMmap = useMmap;
		this.useMlock = useMlock;
		this.embedding = embedding;
		this.nThreads = nThreads;
		this.loraAdapter = loraAdapter;
		this.loraBase = loraBase;
		this.hellaswag = hellaswag;
		this.hellaswagTasks = hellaswagTasks;
		this.memoryF16 = memoryF16;
		this.memTest = memTest;
		this.numa = numa;
		this.verbosePrompt = verbosePrompt;
	}

	/**
	 * The builder class used for creating new {@link ModelParameters} of a {@link LlamaModel}.
	 */
	public static class Builder {

		private int nThreads = Runtime.getRuntime().availableProcessors();
		public int seed = -1;
		public int nCtx = 512; // text context
		public int nBatch = 512; // prompt processing batch size
		public int nGpuLayers = -1; // number of layers to store in VRAM
		public int mainGpu = 0; // the GPU that is used for scratch and small tensors
		public float[] tensorSplit = null; // how to split layers across multiple GPUs (size: LLAMA_MAX_DEVICES)
		public float ropeFreqBase = 10000.0f; // RoPE base frequency
		public float ropeFreqScale = 1.0f; // RoPE frequency scaling factor
		//	public llama_progress_callback progress_callback;
		//	public Pointer progress_callback_user_data;
		public boolean lowVram = false; // if true, reduce VRAM usage at the cost of performance
		public boolean mulMatQ = true; // if true, use experimental mul_mat_q kernels
		public boolean f16Kv; // use fp16 for KV cache
		public boolean logitsAll; // the llama_eval() call computes all logits, not just the last one
		public boolean vocabOnly = false; // only load the vocabulary, no weights
		public boolean useMmap = true; // use mmap if possible
		public boolean useMlock = false; // force system to keep model in RAM
		public boolean embedding = false; // embedding mode only
		private String loraAdapter = null;  // lora adapter path
		private String loraBase = null;  // base model path for the lora adapter

		private boolean hellaswag = false; // compute HellaSwag score over random tasks from datafile supplied in prompt
		private short hellaswagTasks = 400;   // number of tasks to use when computing the HellaSwag score

		private boolean memoryF16 = true;  // use f16 instead of f32 for memory kv
		private boolean memTest = false; // compute maximum memory usage
		private boolean numa = false; // attempt optimizations that help on some NUMA systems
		private boolean verbosePrompt = false; // print prompt tokens before generation

		/**
		 * Constructs the immutable {@link ModelParameters} objects with the configured options.
		 * Note, that all options not configured have sensible defaults.
		 *
		 * @return an immutable parameters object
		 */
		public ModelParameters build() {
			return new ModelParameters(
					nThreads,
					seed,
					nCtx,
					nBatch,
					nGpuLayers,
					mainGpu,
					tensorSplit,
					ropeFreqBase,
					ropeFreqScale,
					lowVram,
					mulMatQ,
					f16Kv,
					logitsAll,
					vocabOnly,
					useMmap,
					useMlock,
					embedding,
					loraAdapter,
					loraBase,
					hellaswag,
					hellaswagTasks,
					memoryF16,
					memTest,
					numa,
					verbosePrompt
			);
		}

		public Builder setNThreads(int nThreads) {
			this.nThreads = nThreads;
			return this;
		}

		public Builder setLoraAdapter(@Nullable String loraAdapter) {
			this.loraAdapter = loraAdapter;
			return this;
		}

		public Builder setLoraBase(@Nullable String loraBase) {
			this.loraBase = loraBase;
			return this;
		}

		public Builder setHellaswag(boolean hellaswag) {
			this.hellaswag = hellaswag;
			return this;
		}

		public Builder setHellaswagTasks(short hellaswagTasks) {
			this.hellaswagTasks = hellaswagTasks;
			return this;
		}

		public Builder setMemoryF16(boolean memoryF16) {
			this.memoryF16 = memoryF16;
			return this;
		}

		public Builder setMemTest(boolean memTest) {
			this.memTest = memTest;
			return this;
		}

		public Builder setNuma(boolean numa) {
			this.numa = numa;
			return this;
		}

		public Builder setVerbosePrompt(boolean verbosePrompt) {
			this.verbosePrompt = verbosePrompt;
			return this;
		}

		/**
		 * Set a callback that will be used to report progress loading the model with a float value of 0-1.
		 *
		 * @return this builder object
		 */
//		public Builder setProgressCallback(@Nullable Consumer<Float> progressCallback) {
//			// Similarly to setting the logger, we don't allow passing any user data to the progress callback, since
//			// the JVM might move the object around in the memory, thus invalidating any pointers.
//			if (progressCallback == null) {
//				ctxParams.setProgress_callback(null);
//			} else {
//				ctxParams.setProgress_callback((progress, ctx) -> progressCallback.accept(progress));
//			}
//			return this;
//		}

		public Builder setSeed(int seed) {
			this.seed = seed;
			return this;
		}

		public Builder setNCtx(int nCtx) {
			this.nCtx = nCtx;
			return this;
		}

		public Builder setNBbatch(int nBatch) {
			this.nBatch = nBatch;
			return this;
		}

		public Builder setNGpuLayers(int nGpuLayers) {
			this.nGpuLayers = nGpuLayers;
			return this;
		}

		public Builder setMainGpu(int mainGpu) {
			this.mainGpu = mainGpu;
			return this;
		}

		public Builder setTensorSplit(float[] tensorSplit) {
			this.tensorSplit = tensorSplit;
			return this;
		}

		public Builder setRopeFreqBase(float ropeFreqBase) {
			this.ropeFreqBase = ropeFreqBase;
			return this;
		}

		public Builder setRopeFreqScale(float ropeFreqScale) {
			this.ropeFreqScale = ropeFreqScale;
			return this;
		}

//		public Builder setProgressCallback(LlamaLibrary.llama_progress_callback progress_callback) {
//			ctxParams.setProgress_callback(progress_callback);
//			return this;
//		}

//		public Builder setProgressCallbackUserData(Pointer progress_callback_user_data) {
//			ctxParams.setProgress_callback_user_data(progress_callback_user_data);
//			return this;
//		}

		public Builder setLowVram(boolean lowVram) {
			this.lowVram = lowVram;
			return this;
		}

		public Builder setMulMatQ(boolean mulMatQ) {
			this.mulMatQ = mulMatQ;
			return this;
		}

		/**
		 * use fp16 for KV cache
		 */
		public Builder setF16Kv(boolean f16Kv) {
			this.f16Kv = f16Kv;
			return this;
		}

		/**
		 * the llama_eval() call computes all logits, not just the last one
		 */
		public Builder setLogitsAll(boolean logitsAll) {
			this.logitsAll = logitsAll;
			return this;
		}

		/**
		 * only load the vocabulary, no weights
		 */
		public Builder setVocabOnly(boolean vocabOnly) {
			this.vocabOnly = vocabOnly;
			return this;
		}

		/**
		 * use mmap if possible
		 */
		public Builder setUseMmap(boolean useMmap) {
			this.useMmap = useMmap;
			return this;
		}

		/**
		 * force system to keep model in RAM
		 */
		public Builder setUseMLock(boolean useMlock) {
			this.useMlock = useMlock;
			return this;
		}

		/**
		 * embedding mode only
		 */
		public Builder setEmbedding(boolean embedding) {
			this.embedding = embedding;
			return this;
		}
	}
}
