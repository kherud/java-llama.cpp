package de.kherud.llama.args;

import java.lang.annotation.Native;

import de.kherud.llama.LlamaModel;

/**
 * Parameters used for initializing a {@link LlamaModel}.
 */
public final class ModelParameters {

	@Native
	private int seed = -1; // RNG seed
	@Native
	private int nThreads = Runtime.getRuntime().availableProcessors();
	@Native
	private int nThreadsBatch = -1; // number of threads to use for batch processing (-1 = use n_threads)
	@Native
	private String modelFilePath; // model path
	@Native
	private String modelUrl; // model url to download
	@Native
	private String huggingFaceRepository; // HF repo
	@Native
	private String huggingFaceFile; // HF file
	@Native
	private String modelAlias; // model alias
	@Native
	private String systemPromptFile;
	@Native
	private int nCtx = 512; // context size
	@Native
	private int nBatch = 2048; // logical batch size for prompt processing (must be >=32 to use BLAS)
	@Native
	private int nUBatch = 512; // physical batch size for prompt processing (must be >=32 to use BLAS)
	@Native
	private int nParallel = 1; // number of parallel sequences to decode
	@Native
	private int nPredict = -1; // new tokens to predict
	@Native
	private GpuSplitMode gpuSplitMode = GpuSplitMode.LAYER; // how to split the model across GPUs
	@Native
	private int nGpuLayers = -1; // number of layers to store in VRAM (-1 - use default)
	@Native
	private int mainGpu = 0; // the GPU that is used for scratch and small tensors
	@Native
	private float[] tensorSplit = null; // // how split tensors should be distributed across GPUs
	@Native
	private RopeScalingType ropeScalingType = RopeScalingType.UNSPECIFIED;
	@Native
	private float ropeFreqBase = 0f; // RoPE base frequency
	@Native
	private float ropeFreqScale = 0f; // RoPE frequency scaling factor
	@Native
	private float yarnExtFactor = -1.0f;
	@Native
	private float yarnAttnFactor = 1.0f;
	@Native
	private float yarnBetaFast = 32.0f;
	@Native
	private float yarnBetaSlow = 1.0f;
	@Native
	private PoolingType poolingType = PoolingType.UNSPECIFIED; // pooling type for embeddings
	@Native
	private float defragmentationThreshold = -1.0f; // KV cache defragmentation threshold
	@Native
	private int groupAttnN = 1;
	@Native
	private int groupAttnW = 512;
	@Native
	private boolean useMmap = true; // use mmap if possible
	@Native
	private boolean useMlock = false; // force system to keep model in RAM
	@Native
	private boolean noKVOffload = false;
	@Native
	private boolean embedding = false; // embedding mode
	@Native
	private boolean continuousBatching = true; // insert new sequences for decoding on-the-fly
	@Native
	private NumaStrategy numa = NumaStrategy.NONE; // attempt optimizations that help on some NUMA systems
	@Native
	private LogFormat logFormat = LogFormat.TEXT;
	@Native
	private boolean verbose = false;

//	@Nullable
//	private String loraAdapter = null;
//	@Nullable
//	private String loraBase = null;

	/**
	 * Set the RNG seed
	 */
	public ModelParameters setSeed(int seed) {
		this.seed = seed;
		return this;
	}

	/**
	 * Set the total amount of threads ever used
	 */
	public ModelParameters setNThreads(int nThreads) {
		this.nThreads = nThreads;
		return this;
	}

	/**
	 * number of threads to use for batch processing (-1 = use {@link #nThreads})
	 */
	public ModelParameters setNThreadsBatch(int nThreadsBatch) {
		this.nThreadsBatch = nThreadsBatch;
		return this;
	}

	/**
	 * Set a file path to load the model from
	 */
	public ModelParameters setModelFilePath(String modelFilePath) {
		this.modelFilePath = modelFilePath;
		return this;
	}

	/**
	 * Set a URL to load the model from
	 */
	public ModelParameters setModelUrl(String modelUrl) {
		this.modelUrl = modelUrl;
		return this;
	}

	/**
	 * Set a HuggingFace repository to load a model from (see {@link #setHuggingFaceFile(String)})
	 */
	public ModelParameters setHuggingFaceRepository(String huggingFaceRepository) {
		this.huggingFaceRepository = huggingFaceRepository;
		return this;
	}

	/**
	 * Set a HuggingFace file to load a model from (see {@link #setHuggingFaceRepository(String)})
	 */
	public ModelParameters setHuggingFaceFile(String huggingFaceFile) {
		this.huggingFaceFile = huggingFaceFile;
		return this;
	}

	/**
	 * Set the model alias
	 */
	public ModelParameters setModelAlias(String modelAlias) {
		this.modelAlias = modelAlias;
		return this;
	}

	/**
	 * Set a file path to load a system prompt from
	 */
	public ModelParameters setSystemPrompt(String systemPromptFile) {
		this.systemPromptFile = systemPromptFile;
		return this;
	}

	/**
	 * Set the context size
	 */
	public ModelParameters setNCtx(int nCtx) {
		this.nCtx = nCtx;
		return this;
	}

	/**
	 * Set the logical batch size for prompt processing (must be >=32 to use BLAS)
	 */
	public ModelParameters setNBatch(int nBatch) {
		this.nBatch = nBatch;
		return this;
	}

	/**
	 * Set the physical batch size for prompt processing (must be >=32 to use BLAS)
	 */
	public ModelParameters setNUBatch(int nUBatch) {
		this.nUBatch = nUBatch;
		return this;
	}

	/**
	 * Set how the number of parallel sequences to decode
	 */
	public ModelParameters setNParallel(int nParallel) {
		this.nParallel = nParallel;
		return this;
	}

	/**
	 * Set the amount of new tokens to predict
	 */
	public ModelParameters setNPredict(int nPredict) {
		this.nPredict = nPredict;
		return this;
	}

	/**
	 * Set how to split the model across GPUs
	 */
	public ModelParameters setGpuSplitMode(GpuSplitMode gpuSplitMode) {
		this.gpuSplitMode = gpuSplitMode;
		return this;
	}

	/**
	 * Set the number of layers to store in VRAM (-1 - use default)
	 */
	public ModelParameters setNGpuLayers(int nGpuLayers) {
		this.nGpuLayers = nGpuLayers;
		return this;
	}

	/**
	 * Set the GPU that is used for scratch and small tensors
	 */
	public ModelParameters setMainGpu(int mainGpu) {
		this.mainGpu = mainGpu;
		return this;
	}

	/**
	 * Set how split tensors should be distributed across GPUs
	 */
	public ModelParameters setTensorSplit(float[] tensorSplit) {
		this.tensorSplit = tensorSplit;
		return this;
	}

	/**
	 * Set the RoPE scaling type
	 */
	public ModelParameters setRopeScalingType(RopeScalingType ropeScalingType) {
		this.ropeScalingType = ropeScalingType;
		return this;
	}

	/**
	 * Set the RoPE base frequency
	 */
	public ModelParameters setRopeFreqBase(float ropeFreqBase) {
		this.ropeFreqBase = ropeFreqBase;
		return this;
	}

	/**
	 * Set the RoPE frequency scaling factor
	 */
	public ModelParameters setRopeFreqScale(float ropeFreqScale) {
		this.ropeFreqScale = ropeFreqScale;
		return this;
	}

	/**
	 * Set the YaRN extrapolation mix factor
	 */
	public ModelParameters setYarnExtrapolationFactor(float yarnExtFactor) {
		this.yarnExtFactor = yarnExtFactor;
		return this;
	}

	/**
	 * Set the YaRN magnitude scaling factor
	 */
	public ModelParameters setYarnMagnitudeFactor(float yarnAttnFactor) {
		this.yarnAttnFactor = yarnAttnFactor;
		return this;
	}

	/**
	 * Set the YaRN low correction dim
	 */
	public ModelParameters setYarnBetaFast(float yarnBetaFast) {
		this.yarnBetaFast = yarnBetaFast;
		return this;
	}

	/**
	 * Set the YaRN high correction dim
	 */
	public ModelParameters setYarnBetaSlow(float yarnBetaSlow) {
		this.yarnBetaSlow = yarnBetaSlow;
		return this;
	}

	/**
	 * Set the pooling type for embeddings
	 */
	public ModelParameters setPoolingType(PoolingType poolingType) {
		this.poolingType = poolingType;
		return this;
	}

	/**
	 * Set the KV cache defragmentation threshold
	 */
	public ModelParameters setDefragmentationThreshold(float defragmentationThreshold) {
		this.defragmentationThreshold = defragmentationThreshold;
		return this;
	}

	/**
	 * Set the group-attention factor
	 */
	public ModelParameters setGroupAttnN(int groupAttnN) {
		this.groupAttnN = groupAttnN;
		return this;
	}

	/**
	 * Set the group-attention width
	 */
	public ModelParameters setGroupAttnW(int groupAttnW) {
		this.groupAttnW = groupAttnW;
		return this;
	}

	/**
	 * Whether to use mmap for faster loads
	 */
	public ModelParameters setUseMmap(boolean useMmap) {
		this.useMmap = useMmap;
		return this;
	}

	/**
	 * Whether to use mlock to keep model in memory
	 */
	public ModelParameters setUseMlock(boolean useMlock) {
		this.useMlock = useMlock;
		return this;
	}

	/**
	 * Whether to disable KV offloading
	 */
	public ModelParameters setNoKVOffload(boolean noKVOffload) {
		this.noKVOffload = noKVOffload;
		return this;
	}

	/**
	 * Whether to only get sentence embeddings
	 */
	public ModelParameters setEmbedding(boolean embedding) {
		this.embedding = embedding;
		return this;
	}

	/**
	 * Whether to insert new sequences for decoding on-the-fly
	 */
	public ModelParameters setContinuousBatching(boolean continuousBatching) {
		this.continuousBatching = continuousBatching;
		return this;
	}

	/**
	 * Set a numa strategy if compiled with NUMA support
	 */
	public ModelParameters setNumaStrategy(NumaStrategy numa) {
		this.numa = numa;
		return this;
	}

	/**
	 * Set the log format
	 */
	public ModelParameters setLogFormat(LogFormat logFormat) {
		this.logFormat = logFormat;
		return this;
	}

	/**
	 * Whether to log additional output (if compiled with <code>LLAMA_VERBOSE</code>)
	 */
	public ModelParameters setVerbose(boolean verbose) {
		this.verbose = verbose;
		return this;
	}

	public int getSeed() {
		return seed;
	}

	public int getNThreads() {
		return nThreads;
	}

	public int getNThreadsBatch() {
		return nThreadsBatch;
	}

	public String getModelFilePath() {
		return modelFilePath;
	}

	public String getModelUrl() {
		return modelUrl;
	}

	public String getHuggingFaceRepository() {
		return huggingFaceRepository;
	}

	public String getHuggingFaceFile() {
		return huggingFaceFile;
	}

	public String getModelAlias() {
		return modelAlias;
	}

	public String getSystemPromptFile() {
		return systemPromptFile;
	}

	public int getNCtx() {
		return nCtx;
	}

	public int getNBatch() {
		return nBatch;
	}

	public int getNUBatch() {
		return nUBatch;
	}

	public int getNParallel() {
		return nParallel;
	}

	public int getNPredict() {
		return nPredict;
	}

	public GpuSplitMode getGpuSplitMode() {
		return gpuSplitMode;
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

	public RopeScalingType getRopeScalingType() {
		return ropeScalingType;
	}

	public float getRopeFreqBase() {
		return ropeFreqBase;
	}

	public float getRopeFreqScale() {
		return ropeFreqScale;
	}

	public float getYarnExtFactor() {
		return yarnExtFactor;
	}

	public float getYarnAttnFactor() {
		return yarnAttnFactor;
	}

	public float getYarnBetaFast() {
		return yarnBetaFast;
	}

	public float getYarnBetaSlow() {
		return yarnBetaSlow;
	}

	public PoolingType getPoolingType() {
		return poolingType;
	}

	public float getDefragmentationThreshold() {
		return defragmentationThreshold;
	}

	public int getGroupAttnN() {
		return groupAttnN;
	}

	public int getGroupAttnW() {
		return groupAttnW;
	}

	public boolean isUseMmap() {
		return useMmap;
	}

	public boolean isUseMlock() {
		return useMlock;
	}

	public boolean isNoKVOffload() {
		return noKVOffload;
	}

	public boolean isEmbedding() {
		return embedding;
	}

	public NumaStrategy getNuma() {
		return numa;
	}

	public LogFormat getLogFormat() {
		return logFormat;
	}

	public boolean isVerbose() {
		return verbose;
	}
}
