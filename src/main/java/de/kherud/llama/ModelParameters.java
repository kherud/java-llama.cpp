package de.kherud.llama;

import org.jetbrains.annotations.Nullable;

/**
 * Parameters used for initializing a {@link LlamaModel}.
 */
public final class ModelParameters {

	private int nThreads = Runtime.getRuntime().availableProcessors();

	private int seed = -1;
	// text context
	private int nCtx = 512;
	// prompt processing batch size
	private int nBatch = 512;
	// number of layers to store in VRAM
	private int nGpuLayers = -1;
	// the GPU that is used for scratch and small tensors
	private int mainGpu = 0;
	// how to split layers across multiple GPUs (size: LLAMA_MAX_DEVICES)
	private float[] tensorSplit = null;
	// RoPE base frequency
	private float ropeFreqBase = 0f;
	// RoPE frequency scaling factor
	private float ropeFreqScale = 0f;
	// if true, use experimental mul_mat_q kernels
	private boolean mulMatQ = true;
	// use fp16 for KV cache
	private boolean f16Kv = false;
	// the llama_eval() call computes all logits, not just the last one
	private boolean logitsAll = false;
	// only load the vocabulary, no weights
	private boolean vocabOnly = false;
	// use mmap if possible
	private boolean useMmap = true;
	// force system to keep model in RAM
	private boolean useMlock = false;
	// embedding mode
	private boolean embedding = false;
	// lora adapter path
	@Nullable
	private String loraAdapter = null;
	// base model path for the lora adapter
	@Nullable
	private String loraBase = null;
	// use f16 instead of f32 for memory kv
	private boolean memoryF16 = true;
	// compute maximum memory usage
	private boolean memTest = false;
	// attempt optimizations that help on some NUMA systems
	private boolean numa = false;
	private boolean verbosePrompt = false; // log prompt tokens before generation

	public ModelParameters setNThreads(int nThreads) {
		this.nThreads = nThreads;
		return this;
	}

	public ModelParameters setLoraAdapter(@Nullable String loraAdapter) {
		this.loraAdapter = loraAdapter;
		return this;
	}

	public ModelParameters setLoraBase(@Nullable String loraBase) {
		this.loraBase = loraBase;
		return this;
	}

	public ModelParameters setMemoryF16(boolean memoryF16) {
		this.memoryF16 = memoryF16;
		return this;
	}

	public ModelParameters setMemTest(boolean memTest) {
		this.memTest = memTest;
		return this;
	}

	public ModelParameters setNuma(boolean numa) {
		this.numa = numa;
		return this;
	}

	public ModelParameters setVerbosePrompt(boolean verbosePrompt) {
		this.verbosePrompt = verbosePrompt;
		return this;
	}

	/**
	 * Set a callback that will be used to report progress loading the model with a float value of 0-1.
	 *
	 * @return this builder object
	 */
//		public ModelParameters setProgressCallback(@Nullable Consumer<Float> progressCallback) {
//			// Similarly to setting the logger, we don't allow passing any user data to the progress callback, since
//			// the JVM might move the object around in the memory, thus invalidating any pointers.
//			if (progressCallback == null) {
//				ctxParams.setProgress_callback(null);
//			} else {
//				ctxParams.setProgress_callback((progress, ctx) -> progressCallback.accept(progress));
//			}
//			return this;
//		}

	public ModelParameters setSeed(int seed) {
		this.seed = seed;
		return this;
	}

	public ModelParameters setNCtx(int nCtx) {
		this.nCtx = nCtx;
		return this;
	}

	public ModelParameters setNBbatch(int nBatch) {
		this.nBatch = nBatch;
		return this;
	}

	public ModelParameters setNGpuLayers(int nGpuLayers) {
		this.nGpuLayers = nGpuLayers;
		return this;
	}

	public ModelParameters setMainGpu(int mainGpu) {
		this.mainGpu = mainGpu;
		return this;
	}

	public ModelParameters setTensorSplit(float[] tensorSplit) {
		this.tensorSplit = tensorSplit;
		return this;
	}

	public ModelParameters setRopeFreqBase(float ropeFreqBase) {
		this.ropeFreqBase = ropeFreqBase;
		return this;
	}

	public ModelParameters setRopeFreqScale(float ropeFreqScale) {
		this.ropeFreqScale = ropeFreqScale;
		return this;
	}

//		public ModelParameters setProgressCallback(LlamaLibrary.llama_progress_callback progress_callback) {
//			ctxParams.setProgress_callback(progress_callback);
//			return this;
//		}

//		public ModelParameters setProgressCallbackUserData(Pointer progress_callback_user_data) {
//			ctxParams.setProgress_callback_user_data(progress_callback_user_data);
//			return this;
//		}

	public ModelParameters setMulMatQ(boolean mulMatQ) {
		this.mulMatQ = mulMatQ;
		return this;
	}

	/**
	 * use fp16 for KV cache
	 */
	public ModelParameters setF16Kv(boolean f16Kv) {
		this.f16Kv = f16Kv;
		return this;
	}

	/**
	 * the llama_eval() call computes all logits, not just the last one
	 */
	public ModelParameters setLogitsAll(boolean logitsAll) {
		this.logitsAll = logitsAll;
		return this;
	}

	/**
	 * only load the vocabulary, no weights
	 */
	public ModelParameters setVocabOnly(boolean vocabOnly) {
		this.vocabOnly = vocabOnly;
		return this;
	}

	/**
	 * use mmap if possible
	 */
	public ModelParameters setUseMmap(boolean useMmap) {
		this.useMmap = useMmap;
		return this;
	}

	/**
	 * force system to keep model in RAM
	 */
	public ModelParameters setUseMLock(boolean useMlock) {
		this.useMlock = useMlock;
		return this;
	}

	/**
	 * embedding mode only
	 */
	public ModelParameters setEmbedding(boolean embedding) {
		this.embedding = embedding;
		return this;
	}

	public int getNThreads() {
		return nThreads;
	}

	public int getSeed() {
		return seed;
	}

	public int getNCtx() {
		return nCtx;
	}

	public int getNBatch() {
		return nBatch;
	}

	public int getNGpuLayers() {
		return nGpuLayers;
	}

	public int getMainGpu() {
		return mainGpu;
	}

	public float[] getTensorSplit() {
		return tensorSplit;
	}

	public float getRopeFreqBase() {
		return ropeFreqBase;
	}

	public float getRopeFreqScale() {
		return ropeFreqScale;
	}

	public boolean isMulMatQ() {
		return mulMatQ;
	}

	public boolean isF16Kv() {
		return f16Kv;
	}

	public boolean isLogitsAll() {
		return logitsAll;
	}

	public boolean isVocabOnly() {
		return vocabOnly;
	}

	public boolean isUseMmap() {
		return useMmap;
	}

	public boolean isUseMlock() {
		return useMlock;
	}

	public boolean isEmbedding() {
		return embedding;
	}

	public @Nullable String getLoraAdapter() {
		return loraAdapter;
	}

	public @Nullable String getLoraBase() {
		return loraBase;
	}

	public boolean isMemoryF16() {
		return memoryF16;
	}

	public boolean isMemTest() {
		return memTest;
	}

	public boolean isNuma() {
		return numa;
	}

	public boolean isVerbosePrompt() {
		return verbosePrompt;
	}
}
