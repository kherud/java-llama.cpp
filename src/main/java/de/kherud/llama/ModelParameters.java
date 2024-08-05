package de.kherud.llama;

import java.util.Map;

import de.kherud.llama.args.GpuSplitMode;
import de.kherud.llama.args.NumaStrategy;
import de.kherud.llama.args.PoolingType;
import de.kherud.llama.args.RopeScalingType;

/***
 * Parameters used for initializing a {@link LlamaModel}.
 */
public final class ModelParameters extends JsonParameters {

	private static final String PARAM_SEED = "seed";
	private static final String PARAM_N_THREADS = "n_threads";
	private static final String PARAM_N_THREADS_DRAFT = "n_threads_draft";
	private static final String PARAM_N_THREADS_BATCH = "n_threads_batch";
	private static final String PARAM_N_THREADS_BATCH_DRAFT = "n_threads_batch_draft";
	private static final String PARAM_N_PREDICT = "n_predict";
	private static final String PARAM_N_CTX = "n_ctx";
	private static final String PARAM_N_BATCH = "n_batch";
	private static final String PARAM_N_UBATCH = "n_ubatch";
	private static final String PARAM_N_KEEP = "n_keep";
	private static final String PARAM_N_DRAFT = "n_draft";
	private static final String PARAM_N_CHUNKS = "n_chunks";
	private static final String PARAM_N_PARALLEL = "n_parallel";
	private static final String PARAM_N_SEQUENCES = "n_sequences";
	private static final String PARAM_P_SPLIT = "p_split";
	private static final String PARAM_N_GPU_LAYERS = "n_gpu_layers";
	private static final String PARAM_N_GPU_LAYERS_DRAFT = "n_gpu_layers_draft";
	private static final String PARAM_SPLIT_MODE = "split_mode";
	private static final String PARAM_MAIN_GPU = "main_gpu";
	private static final String PARAM_TENSOR_SPLIT = "tensor_split";
	private static final String PARAM_GRP_ATTN_N = "grp_attn_n";
	private static final String PARAM_GRP_ATTN_W = "grp_attn_w";
	private static final String PARAM_ROPE_FREQ_BASE = "rope_freq_base";
	private static final String PARAM_ROPE_FREQ_SCALE = "rope_freq_scale";
	private static final String PARAM_YARN_EXT_FACTOR = "yarn_ext_factor";
	private static final String PARAM_YARN_ATTN_FACTOR = "yarn_attn_factor";
	private static final String PARAM_YARN_BETA_FAST = "yarn_beta_fast";
	private static final String PARAM_YARN_BETA_SLOW = "yarn_beta_slow";
	private static final String PARAM_YARN_ORIG_CTX = "yarn_orig_ctx";
	private static final String PARAM_DEFRAG_THOLD = "defrag_thold";
	private static final String PARAM_NUMA = "numa";
	private static final String PARAM_ROPE_SCALING_TYPE = "rope_scaling_type";
	private static final String PARAM_POOLING_TYPE = "pooling_type";
	private static final String PARAM_MODEL = "model";
	private static final String PARAM_MODEL_DRAFT = "model_draft";
	private static final String PARAM_MODEL_ALIAS = "model_alias";
	private static final String PARAM_MODEL_URL = "model_url";
	private static final String PARAM_HF_REPO = "hf_repo";
	private static final String PARAM_HF_FILE = "hf_file";
	private static final String PARAM_LOOKUP_CACHE_STATIC = "lookup_cache_static";
	private static final String PARAM_LOOKUP_CACHE_DYNAMIC = "lookup_cache_dynamic";
	private static final String PARAM_LORA_ADAPTER = "lora_adapter";
	private static final String PARAM_EMBEDDING = "embedding";
	private static final String PARAM_CONT_BATCHING = "cont_batching";
	private static final String PARAM_FLASH_ATTENTION = "flash_attn";
	private static final String PARAM_INPUT_PREFIX_BOS = "input_prefix_bos";
	private static final String PARAM_IGNORE_EOS = "ignore_eos";
	private static final String PARAM_USE_MMAP = "use_mmap";
	private static final String PARAM_USE_MLOCK = "use_mlock";
	private static final String PARAM_NO_KV_OFFLOAD = "no_kv_offload";
	private static final String PARAM_SYSTEM_PROMPT = "system_prompt";
	private static final String PARAM_CHAT_TEMPLATE = "chat_template";

	/**
	 * Set the RNG seed
	 */
	public ModelParameters setSeed(int seed) {
		parameters.put(PARAM_SEED, String.valueOf(seed));
		return this;
	}

	/**
	 * Set the number of threads to use during generation (default: 8)
	 */
	public ModelParameters setNThreads(int nThreads) {
		parameters.put(PARAM_N_THREADS, String.valueOf(nThreads));
		return this;
	}

	/**
	 * Set the number of threads to use during draft generation (default: same as {@link #setNThreads(int)})
	 */
	public ModelParameters setNThreadsDraft(int nThreadsDraft) {
		parameters.put(PARAM_N_THREADS_DRAFT, String.valueOf(nThreadsDraft));
		return this;
	}

	/**
	 * Set the number of threads to use during batch and prompt processing (default: same as {@link #setNThreads(int)})
	 */
	public ModelParameters setNThreadsBatch(int nThreadsBatch) {
		parameters.put(PARAM_N_THREADS_BATCH, String.valueOf(nThreadsBatch));
		return this;
	}

	/**
	 * Set the number of threads to use during batch and prompt processing (default: same as
	 * {@link #setNThreadsDraft(int)})
	 */
	public ModelParameters setNThreadsBatchDraft(int nThreadsBatchDraft) {
		parameters.put(PARAM_N_THREADS_BATCH_DRAFT, String.valueOf(nThreadsBatchDraft));
		return this;
	}

	/**
	 * Set the number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
	 */
	public ModelParameters setNPredict(int nPredict) {
		parameters.put(PARAM_N_PREDICT, String.valueOf(nPredict));
		return this;
	}

	/**
	 * Set the size of the prompt context (default: 512, 0 = loaded from model)
	 */
	public ModelParameters setNCtx(int nCtx) {
		parameters.put(PARAM_N_CTX, String.valueOf(nCtx));
		return this;
	}

	/**
	 * Set the logical batch size for prompt processing (must be >=32 to use BLAS)
	 */
	public ModelParameters setNBatch(int nBatch) {
		parameters.put(PARAM_N_BATCH, String.valueOf(nBatch));
		return this;
	}

	/**
	 * Set the physical batch size for prompt processing (must be >=32 to use BLAS)
	 */
	public ModelParameters setNUbatch(int nUbatch) {
		parameters.put(PARAM_N_UBATCH, String.valueOf(nUbatch));
		return this;
	}

	/**
	 * Set the number of tokens to keep from the initial prompt (default: 0, -1 = all)
	 */
	public ModelParameters setNKeep(int nKeep) {
		parameters.put(PARAM_N_KEEP, String.valueOf(nKeep));
		return this;
	}

	/**
	 * Set the number of tokens to draft for speculative decoding (default: 5)
	 */
	public ModelParameters setNDraft(int nDraft) {
		parameters.put(PARAM_N_DRAFT, String.valueOf(nDraft));
		return this;
	}

	/**
	 * Set the maximal number of chunks to process (default: -1, -1 = all)
	 */
	public ModelParameters setNChunks(int nChunks) {
		parameters.put(PARAM_N_CHUNKS, String.valueOf(nChunks));
		return this;
	}

	/**
	 * Set the number of parallel sequences to decode (default: 1)
	 */
	public ModelParameters setNParallel(int nParallel) {
		parameters.put(PARAM_N_PARALLEL, String.valueOf(nParallel));
		return this;
	}

	/**
	 * Set the number of sequences to decode (default: 1)
	 */
	public ModelParameters setNSequences(int nSequences) {
		parameters.put(PARAM_N_SEQUENCES, String.valueOf(nSequences));
		return this;
	}

	/**
	 * Set the speculative decoding split probability (default: 0.1)
	 */
	public ModelParameters setPSplit(float pSplit) {
		parameters.put(PARAM_P_SPLIT, String.valueOf(pSplit));
		return this;
	}

	/**
	 * Set the number of layers to store in VRAM (-1 - use default)
	 */
	public ModelParameters setNGpuLayers(int nGpuLayers) {
		parameters.put(PARAM_N_GPU_LAYERS, String.valueOf(nGpuLayers));
		return this;
	}

	/**
	 * Set the number of layers to store in VRAM for the draft model (-1 - use default)
	 */
	public ModelParameters setNGpuLayersDraft(int nGpuLayersDraft) {
		parameters.put(PARAM_N_GPU_LAYERS_DRAFT, String.valueOf(nGpuLayersDraft));
		return this;
	}

	/**
	 * Set how to split the model across GPUs
	 */
	public ModelParameters setSplitMode(GpuSplitMode splitMode) {
//		switch (splitMode) {
//			case NONE: parameters.put(PARAM_SPLIT_MODE, "\"none\""); break;
//			case ROW: parameters.put(PARAM_SPLIT_MODE, "\"row\""); break;
//			case LAYER: parameters.put(PARAM_SPLIT_MODE, "\"layer\""); break;
//		}
		parameters.put(PARAM_SPLIT_MODE, String.valueOf(splitMode.ordinal()));
		return this;
	}

	/**
	 * Set the GPU that is used for scratch and small tensors
	 */
	public ModelParameters setMainGpu(int mainGpu) {
		parameters.put(PARAM_MAIN_GPU, String.valueOf(mainGpu));
		return this;
	}

	/**
	 * Set how split tensors should be distributed across GPUs
	 */
	public ModelParameters setTensorSplit(float[] tensorSplit) {
		if (tensorSplit.length > 0) {
			StringBuilder builder = new StringBuilder();
			builder.append("[");
			for (int i = 0; i < tensorSplit.length; i++) {
				builder.append(tensorSplit[i]);
				if (i < tensorSplit.length - 1) {
					builder.append(", ");
				}
			}
			builder.append("]");
			parameters.put(PARAM_TENSOR_SPLIT, builder.toString());
		}
		return this;
	}

	/**
	 * Set the group-attention factor (default: 1)
	 */
	public ModelParameters setGrpAttnN(int grpAttnN) {
		parameters.put(PARAM_GRP_ATTN_N, String.valueOf(grpAttnN));
		return this;
	}

	/**
	 * Set the group-attention width (default: 512.0)
	 */
	public ModelParameters setGrpAttnW(int grpAttnW) {
		parameters.put(PARAM_GRP_ATTN_W, String.valueOf(grpAttnW));
		return this;
	}

	/**
	 * Set the RoPE base frequency, used by NTK-aware scaling (default: loaded from model)
	 */
	public ModelParameters setRopeFreqBase(float ropeFreqBase) {
		parameters.put(PARAM_ROPE_FREQ_BASE, String.valueOf(ropeFreqBase));
		return this;
	}

	/**
	 * Set the RoPE frequency scaling factor, expands context by a factor of 1/N
	 */
	public ModelParameters setRopeFreqScale(float ropeFreqScale) {
		parameters.put(PARAM_ROPE_FREQ_SCALE, String.valueOf(ropeFreqScale));
		return this;
	}

	/**
	 * Set the YaRN extrapolation mix factor (default: 1.0, 0.0 = full interpolation)
	 */
	public ModelParameters setYarnExtFactor(float yarnExtFactor) {
		parameters.put(PARAM_YARN_EXT_FACTOR, String.valueOf(yarnExtFactor));
		return this;
	}

	/**
	 * Set the YaRN scale sqrt(t) or attention magnitude (default: 1.0)
	 */
	public ModelParameters setYarnAttnFactor(float yarnAttnFactor) {
		parameters.put(PARAM_YARN_ATTN_FACTOR, String.valueOf(yarnAttnFactor));
		return this;
	}

	/**
	 * Set the YaRN low correction dim or beta (default: 32.0)
	 */
	public ModelParameters setYarnBetaFast(float yarnBetaFast) {
		parameters.put(PARAM_YARN_BETA_FAST, String.valueOf(yarnBetaFast));
		return this;
	}

	/**
	 * Set the YaRN high correction dim or alpha (default: 1.0)
	 */
	public ModelParameters setYarnBetaSlow(float yarnBetaSlow) {
		parameters.put(PARAM_YARN_BETA_SLOW, String.valueOf(yarnBetaSlow));
		return this;
	}

	/**
	 * Set the YaRN original context size of model (default: 0 = model training context size)
	 */
	public ModelParameters setYarnOrigCtx(int yarnOrigCtx) {
		parameters.put(PARAM_YARN_ORIG_CTX, String.valueOf(yarnOrigCtx));
		return this;
	}

	/**
	 * Set the KV cache defragmentation threshold (default: -1.0, &lt; 0 - disabled)
	 */
	public ModelParameters setDefragmentationThreshold(float defragThold) {
		parameters.put(PARAM_DEFRAG_THOLD, String.valueOf(defragThold));
		return this;
	}

	/**
	 * Set optimization strategies that help on some NUMA systems (if available)
	 * <ul>
	 * <li><b>distribute</b>: spread execution evenly over all nodes</li>
	 * <li><b>isolate</b>: only spawn threads on CPUs on the node that execution started on</li>
	 * <li><b>numactl</b>: use the CPU map provided by numactl</li>
	 * </ul>
	 * If run without this previously, it is recommended to drop the system page cache before using this
	 * (see <a href="https://github.com/ggerganov/llama.cpp/issues/1437">#1437</a>).
	 */
	public ModelParameters setNuma(NumaStrategy numa) {
//		switch (numa) {
//			case DISTRIBUTE:
//				parameters.put(PARAM_NUMA, "\"distribute\"");
//				break;
//			case ISOLATE:
//				parameters.put(PARAM_NUMA, "\"isolate\"");
//				break;
//			case NUMA_CTL:
//				parameters.put(PARAM_NUMA, "\"numactl\"");
//				break;
//			case MIRROR:
//				parameters.put(PARAM_NUMA, "\"mirror\"");
//				break;
//		}
		parameters.put(PARAM_NUMA, String.valueOf(numa.ordinal()));
		return this;
	}

	/**
	 * Set the RoPE frequency scaling method, defaults to linear unless specified by the model
	 */
	public ModelParameters setRopeScalingType(RopeScalingType ropeScalingType) {
//		switch (ropeScalingType) {
//			case LINEAR:
//				parameters.put(PARAM_ROPE_SCALING_TYPE, "\"linear\"");
//				break;
//			case YARN:
//				parameters.put(PARAM_ROPE_SCALING_TYPE, "\"yarn\"");
//				break;
//		}
		parameters.put(PARAM_ROPE_SCALING_TYPE, String.valueOf(ropeScalingType.ordinal()));
		return this;
	}

	/**
	 * Set the pooling type for embeddings, use model default if unspecified
	 */
	public ModelParameters setPoolingType(PoolingType poolingType) {
//		switch (poolingType) {
//			case MEAN:
//				parameters.put(PARAM_POOLING_TYPE, "\"mean\"");
//				break;
//			case CLS:
//				parameters.put(PARAM_POOLING_TYPE, "\"cls\"");
//				break;
//		}
		parameters.put(PARAM_POOLING_TYPE, String.valueOf(poolingType.ordinal()));
		return this;
	}

	/**
	 * Set the model file path to load (default: models/7B/ggml-model-f16.gguf)
	 */
	public ModelParameters setModelFilePath(String model) {
		parameters.put(PARAM_MODEL, toJsonString(model));
		return this;
	}

	/**
	 * Set the draft model for speculative decoding (default: unused)
	 */
	public ModelParameters setModelDraft(String modelDraft) {
		parameters.put(PARAM_MODEL_DRAFT, toJsonString(modelDraft));
		return this;
	}

	/**
	 * Set a model alias
	 */
	public ModelParameters setModelAlias(String modelAlias) {
		parameters.put(PARAM_MODEL_ALIAS, toJsonString(modelAlias));
		return this;
	}

	/**
	 * Set a URL to download a model from (default: unused).
	 * Note, that this requires the library to be built with CURL (<code>-DLLAMA_CURL=ON</code>).
	 */
	public ModelParameters setModelUrl(String modelUrl) {
		parameters.put(PARAM_MODEL_URL, toJsonString(modelUrl));
		return this;
	}

	/**
	 * Set a Hugging Face model repository to use a model from (default: unused, see
	 * {@link #setHuggingFaceFile(String)})
	 */
	public ModelParameters setHuggingFaceRepository(String hfRepo) {
		parameters.put(PARAM_HF_REPO, toJsonString(hfRepo));
		return this;
	}

	/**
	 * Set a Hugging Face model file to use (default: unused, see {@link #setHuggingFaceRepository(String)})
	 */
	public ModelParameters setHuggingFaceFile(String hfFile) {
		parameters.put(PARAM_HF_FILE, toJsonString(hfFile));
		return this;
	}

	/**
	 * Set path to static lookup cache to use for lookup decoding (not updated by generation)
	 */
	public ModelParameters setLookupCacheStaticFilePath(String lookupCacheStatic) {
		parameters.put(PARAM_LOOKUP_CACHE_STATIC, toJsonString(lookupCacheStatic));
		return this;
	}

	/**
	 * Set path to dynamic lookup cache to use for lookup decoding (updated by generation)
	 */
	public ModelParameters setLookupCacheDynamicFilePath(String lookupCacheDynamic) {
		parameters.put(PARAM_LOOKUP_CACHE_DYNAMIC, toJsonString(lookupCacheDynamic));
		return this;
	}

	/**
	 * Set LoRA adapters to use (implies --no-mmap).
	 * The key is expected to be a file path, the values are expected to be scales.
	 */
	public ModelParameters setLoraAdapters(Map<String, Float> loraAdapters) {
		if (!loraAdapters.isEmpty()) {
			StringBuilder builder = new StringBuilder();
			builder.append("{");
			int i = 0;
			for (Map.Entry<String, Float> entry : loraAdapters.entrySet()) {
				String key = entry.getKey();
				Float value = entry.getValue();
				builder.append(toJsonString(key))
						.append(": ")
						.append(value);
				if (i++ < loraAdapters.size() - 1) {
					builder.append(", ");
				}
			}
			builder.append("}");
			parameters.put(PARAM_LORA_ADAPTER, builder.toString());
		}
		return this;
	}

	/**
	 * Whether to load model with embedding support
	 */
	public ModelParameters setEmbedding(boolean embedding) {
		parameters.put(PARAM_EMBEDDING, String.valueOf(embedding));
		return this;
	}

	/**
	 * Whether to enable continuous batching (also called "dynamic batching") (default: disabled)
	 */
	public ModelParameters setContinuousBatching(boolean contBatching) {
		parameters.put(PARAM_CONT_BATCHING, String.valueOf(contBatching));
		return this;
	}

	/**
	 * Whether to enable Flash Attention (default: disabled)
	 */
	public ModelParameters setFlashAttention(boolean flashAttention) {
		parameters.put(PARAM_FLASH_ATTENTION, String.valueOf(flashAttention));
		return this;
	}

	/**
	 * Whether to add prefix BOS to user inputs, preceding the `--in-prefix` string
	 */
	public ModelParameters setInputPrefixBos(boolean inputPrefixBos) {
		parameters.put(PARAM_INPUT_PREFIX_BOS, String.valueOf(inputPrefixBos));
		return this;
	}

	/**
	 * Whether to ignore end of stream token and continue generating (implies --logit-bias 2-inf)
	 */
	public ModelParameters setIgnoreEos(boolean ignoreEos) {
		parameters.put(PARAM_IGNORE_EOS, String.valueOf(ignoreEos));
		return this;
	}

	/**
	 * Whether to use memory-map model (faster load but may increase pageouts if not using mlock)
	 */
	public ModelParameters setUseMmap(boolean useMmap) {
		parameters.put(PARAM_USE_MMAP, String.valueOf(useMmap));
		return this;
	}

	/**
	 * Whether to force the system to keep model in RAM rather than swapping or compressing
	 */
	public ModelParameters setUseMlock(boolean useMlock) {
		parameters.put(PARAM_USE_MLOCK, String.valueOf(useMlock));
		return this;
	}

	/**
	 * Whether to disable KV offload
	 */
	public ModelParameters setNoKvOffload(boolean noKvOffload) {
		parameters.put(PARAM_NO_KV_OFFLOAD, String.valueOf(noKvOffload));
		return this;
	}

	/**
	 * Set a system prompt to use
	 */
	public ModelParameters setSystemPrompt(String systemPrompt) {
		parameters.put(PARAM_SYSTEM_PROMPT, toJsonString(systemPrompt));
		return this;
	}

	/**
	 * The chat template to use (default: empty)
	 */
	public ModelParameters setChatTemplate(String chatTemplate) {
		parameters.put(PARAM_CHAT_TEMPLATE, toJsonString(chatTemplate));
		return this;
	}

}
