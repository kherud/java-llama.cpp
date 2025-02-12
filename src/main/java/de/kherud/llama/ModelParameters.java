package de.kherud.llama;

import de.kherud.llama.args.*;

/***
 * Parameters used for initializing a {@link LlamaModel}.
 */
@SuppressWarnings("unused")
public final class ModelParameters extends CliParameters {

    /**
     * Set the number of threads to use during generation (default: -1).
     */
    public ModelParameters setThreads(int nThreads) {
        parameters.put("--threads", String.valueOf(nThreads));
        return this;
    }

    /**
     * Set the number of threads to use during batch and prompt processing (default: same as --threads).
     */
    public ModelParameters setThreadsBatch(int nThreads) {
        parameters.put("--threads-batch", String.valueOf(nThreads));
        return this;
    }

    /**
     * Set the CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: "").
     */
    public ModelParameters setCpuMask(String mask) {
        parameters.put("--cpu-mask", mask);
        return this;
    }

    /**
     * Set the range of CPUs for affinity. Complements --cpu-mask.
     */
    public ModelParameters setCpuRange(String range) {
        parameters.put("--cpu-range", range);
        return this;
    }

    /**
     * Use strict CPU placement (default: 0).
     */
    public ModelParameters setCpuStrict(int strictCpu) {
        parameters.put("--cpu-strict", String.valueOf(strictCpu));
        return this;
    }

    /**
     * Set process/thread priority: 0-normal, 1-medium, 2-high, 3-realtime (default: 0).
     */
    public ModelParameters setPriority(int priority) {
        if (priority < 0 || priority > 3) {
            throw new IllegalArgumentException("Invalid value for priority");
        }
        parameters.put("--prio", String.valueOf(priority));
        return this;
    }

    /**
     * Set the polling level to wait for work (0 - no polling, default: 0).
     */
    public ModelParameters setPoll(int poll) {
        parameters.put("--poll", String.valueOf(poll));
        return this;
    }

    /**
     * Set the CPU affinity mask for batch processing: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask).
     */
    public ModelParameters setCpuMaskBatch(String mask) {
        parameters.put("--cpu-mask-batch", mask);
        return this;
    }

    /**
     * Set the ranges of CPUs for batch affinity. Complements --cpu-mask-batch.
     */
    public ModelParameters setCpuRangeBatch(String range) {
        parameters.put("--cpu-range-batch", range);
        return this;
    }

    /**
     * Use strict CPU placement for batch processing (default: same as --cpu-strict).
     */
    public ModelParameters setCpuStrictBatch(int strictCpuBatch) {
        parameters.put("--cpu-strict-batch", String.valueOf(strictCpuBatch));
        return this;
    }

    /**
     * Set process/thread priority for batch processing: 0-normal, 1-medium, 2-high, 3-realtime (default: 0).
     */
    public ModelParameters setPriorityBatch(int priorityBatch) {
        if (priorityBatch < 0 || priorityBatch > 3) {
            throw new IllegalArgumentException("Invalid value for priority batch");
        }
        parameters.put("--prio-batch", String.valueOf(priorityBatch));
        return this;
    }

    /**
     * Set the polling level for batch processing (default: same as --poll).
     */
    public ModelParameters setPollBatch(int pollBatch) {
        parameters.put("--poll-batch", String.valueOf(pollBatch));
        return this;
    }

    /**
     * Set the size of the prompt context (default: 0, 0 = loaded from model).
     */
    public ModelParameters setCtxSize(int ctxSize) {
        parameters.put("--ctx-size", String.valueOf(ctxSize));
        return this;
    }

    /**
     * Set the number of tokens to predict (default: -1 = infinity, -2 = until context filled).
     */
    public ModelParameters setPredict(int nPredict) {
        parameters.put("--predict", String.valueOf(nPredict));
        return this;
    }

    /**
     * Set the logical maximum batch size (default: 0).
     */
    public ModelParameters setBatchSize(int batchSize) {
        parameters.put("--batch-size", String.valueOf(batchSize));
        return this;
    }

    /**
     * Set the physical maximum batch size (default: 0).
     */
    public ModelParameters setUbatchSize(int ubatchSize) {
        parameters.put("--ubatch-size", String.valueOf(ubatchSize));
        return this;
    }

    /**
     * Set the number of tokens to keep from the initial prompt (default: -1 = all).
     */
    public ModelParameters setKeep(int keep) {
        parameters.put("--keep", String.valueOf(keep));
        return this;
    }

    /**
     * Disable context shift on infinite text generation (default: enabled).
     */
    public ModelParameters disableContextShift() {
        parameters.put("--no-context-shift", null);
        return this;
    }

    /**
     * Enable Flash Attention (default: disabled).
     */
    public ModelParameters enableFlashAttn() {
        parameters.put("--flash-attn", null);
        return this;
    }

    /**
     * Disable internal libllama performance timings (default: false).
     */
    public ModelParameters disablePerf() {
        parameters.put("--no-perf", null);
        return this;
    }

    /**
     * Process escape sequences (default: true).
     */
    public ModelParameters enableEscape() {
        parameters.put("--escape", null);
        return this;
    }

    /**
     * Do not process escape sequences (default: false).
     */
    public ModelParameters disableEscape() {
        parameters.put("--no-escape", null);
        return this;
    }

    /**
     * Enable special tokens output (default: true).
     */
    public ModelParameters enableSpecial() {
        parameters.put("--special", null);
        return this;
    }

    /**
     * Skip warming up the model with an empty run (default: false).
     */
    public ModelParameters skipWarmup() {
        parameters.put("--no-warmup", null);
        return this;
    }

    /**
     * Use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this.
     * (default: disabled)
     */
    public ModelParameters setSpmInfill() {
        parameters.put("--spm-infill", null);
        return this;
    }

    /**
     * Set samplers that will be used for generation in the order, separated by ';' (default: all).
     */
    public ModelParameters setSamplers(Sampler... samplers) {
        if (samplers.length > 0) {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < samplers.length; i++) {
                Sampler sampler = samplers[i];
                builder.append(sampler.name().toLowerCase());
                if (i < samplers.length - 1) {
                    builder.append(";");
                }
            }
            parameters.put("--samplers", builder.toString());
        }
        return this;
    }

    /**
     * Set RNG seed (default: -1, use random seed).
     */
    public ModelParameters setSeed(long seed) {
        parameters.put("--seed", String.valueOf(seed));
        return this;
    }

    /**
     * Ignore end of stream token and continue generating (implies --logit-bias EOS-inf).
     */
    public ModelParameters ignoreEos() {
        parameters.put("--ignore-eos", null);
        return this;
    }

    /**
     * Set temperature for sampling (default: 0.8).
     */
    public ModelParameters setTemp(float temp) {
        parameters.put("--temp", String.valueOf(temp));
        return this;
    }

    /**
     * Set top-k sampling (default: 40, 0 = disabled).
     */
    public ModelParameters setTopK(int topK) {
        parameters.put("--top-k", String.valueOf(topK));
        return this;
    }

    /**
     * Set top-p sampling (default: 0.95, 1.0 = disabled).
     */
    public ModelParameters setTopP(float topP) {
        parameters.put("--top-p", String.valueOf(topP));
        return this;
    }

    /**
     * Set min-p sampling (default: 0.05, 0.0 = disabled).
     */
    public ModelParameters setMinP(float minP) {
        parameters.put("--min-p", String.valueOf(minP));
        return this;
    }

    /**
     * Set xtc probability (default: 0.0, 0.0 = disabled).
     */
    public ModelParameters setXtcProbability(float xtcProbability) {
        parameters.put("--xtc-probability", String.valueOf(xtcProbability));
        return this;
    }

    /**
     * Set xtc threshold (default: 0.1, 1.0 = disabled).
     */
    public ModelParameters setXtcThreshold(float xtcThreshold) {
        parameters.put("--xtc-threshold", String.valueOf(xtcThreshold));
        return this;
    }

    /**
     * Set locally typical sampling parameter p (default: 1.0, 1.0 = disabled).
     */
    public ModelParameters setTypical(float typP) {
        parameters.put("--typical", String.valueOf(typP));
        return this;
    }

    /**
     * Set last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size).
     */
    public ModelParameters setRepeatLastN(int repeatLastN) {
        if (repeatLastN < -1) {
            throw new RuntimeException("Invalid repeat-last-n value");
        }
        parameters.put("--repeat-last-n", String.valueOf(repeatLastN));
        return this;
    }

    /**
     * Set penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled).
     */
    public ModelParameters setRepeatPenalty(float repeatPenalty) {
        parameters.put("--repeat-penalty", String.valueOf(repeatPenalty));
        return this;
    }

    /**
     * Set repeat alpha presence penalty (default: 0.0, 0.0 = disabled).
     */
    public ModelParameters setPresencePenalty(float presencePenalty) {
        parameters.put("--presence-penalty", String.valueOf(presencePenalty));
        return this;
    }

    /**
     * Set repeat alpha frequency penalty (default: 0.0, 0.0 = disabled).
     */
    public ModelParameters setFrequencyPenalty(float frequencyPenalty) {
        parameters.put("--frequency-penalty", String.valueOf(frequencyPenalty));
        return this;
    }

    /**
     * Set DRY sampling multiplier (default: 0.0, 0.0 = disabled).
     */
    public ModelParameters setDryMultiplier(float dryMultiplier) {
        parameters.put("--dry-multiplier", String.valueOf(dryMultiplier));
        return this;
    }

    /**
     * Set DRY sampling base value (default: 1.75).
     */
    public ModelParameters setDryBase(float dryBase) {
        parameters.put("--dry-base", String.valueOf(dryBase));
        return this;
    }

    /**
     * Set allowed length for DRY sampling (default: 2).
     */
    public ModelParameters setDryAllowedLength(int dryAllowedLength) {
        parameters.put("--dry-allowed-length", String.valueOf(dryAllowedLength));
        return this;
    }

    /**
     * Set DRY penalty for the last n tokens (default: -1, 0 = disable, -1 = context size).
     */
    public ModelParameters setDryPenaltyLastN(int dryPenaltyLastN) {
        if (dryPenaltyLastN < -1) {
            throw new RuntimeException("Invalid dry-penalty-last-n value");
        }
        parameters.put("--dry-penalty-last-n", String.valueOf(dryPenaltyLastN));
        return this;
    }

    /**
     * Add sequence breaker for DRY sampling, clearing out default breakers (default: none).
     */
    public ModelParameters setDrySequenceBreaker(String drySequenceBreaker) {
        parameters.put("--dry-sequence-breaker", drySequenceBreaker);
        return this;
    }

    /**
     * Set dynamic temperature range (default: 0.0, 0.0 = disabled).
     */
    public ModelParameters setDynatempRange(float dynatempRange) {
        parameters.put("--dynatemp-range", String.valueOf(dynatempRange));
        return this;
    }

    /**
     * Set dynamic temperature exponent (default: 1.0).
     */
    public ModelParameters setDynatempExponent(float dynatempExponent) {
        parameters.put("--dynatemp-exp", String.valueOf(dynatempExponent));
        return this;
    }

    /**
     * Use Mirostat sampling (default: PLACEHOLDER, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).
     */
    public ModelParameters setMirostat(MiroStat mirostat) {
        parameters.put("--mirostat", String.valueOf(mirostat.ordinal()));
        return this;
    }

    /**
     * Set Mirostat learning rate, parameter eta (default: 0.1).
     */
    public ModelParameters setMirostatLR(float mirostatLR) {
        parameters.put("--mirostat-lr", String.valueOf(mirostatLR));
        return this;
    }

    /**
     * Set Mirostat target entropy, parameter tau (default: 5.0).
     */
    public ModelParameters setMirostatEnt(float mirostatEnt) {
        parameters.put("--mirostat-ent", String.valueOf(mirostatEnt));
        return this;
    }

    /**
     * Modify the likelihood of token appearing in the completion.
     */
    public ModelParameters setLogitBias(String tokenIdAndBias) {
        parameters.put("--logit-bias", tokenIdAndBias);
        return this;
    }

    /**
     * Set BNF-like grammar to constrain generations (default: empty).
     */
    public ModelParameters setGrammar(String grammar) {
        parameters.put("--grammar", grammar);
        return this;
    }

    /**
     * Specify the file to read grammar from.
     */
    public ModelParameters setGrammarFile(String fileName) {
        parameters.put("--grammar-file", fileName);
        return this;
    }

    /**
     * Specify the JSON schema to constrain generations (default: empty).
     */
    public ModelParameters setJsonSchema(String schema) {
        parameters.put("--json-schema", schema);
        return this;
    }

    /**
     * Set pooling type for embeddings (default: model default if unspecified).
     */
    public ModelParameters setPoolingType(PoolingType type) {
        parameters.put("--pooling", String.valueOf(type.getId()));
        return this;
    }

    /**
     * Set RoPE frequency scaling method (default: linear unless specified by the model).
     */
    public ModelParameters setRopeScaling(RopeScalingType type) {
        parameters.put("--rope-scaling", String.valueOf(type.getId()));
        return this;
    }

    /**
     * Set RoPE context scaling factor, expands context by a factor of N.
     */
    public ModelParameters setRopeScale(float ropeScale) {
        parameters.put("--rope-scale", String.valueOf(ropeScale));
        return this;
    }

    /**
     * Set RoPE base frequency, used by NTK-aware scaling (default: loaded from model).
     */
    public ModelParameters setRopeFreqBase(float ropeFreqBase) {
        parameters.put("--rope-freq-base", String.valueOf(ropeFreqBase));
        return this;
    }

    /**
     * Set RoPE frequency scaling factor, expands context by a factor of 1/N.
     */
    public ModelParameters setRopeFreqScale(float ropeFreqScale) {
        parameters.put("--rope-freq-scale", String.valueOf(ropeFreqScale));
        return this;
    }

    /**
     * Set YaRN: original context size of model (default: model training context size).
     */
    public ModelParameters setYarnOrigCtx(int yarnOrigCtx) {
        parameters.put("--yarn-orig-ctx", String.valueOf(yarnOrigCtx));
        return this;
    }

    /**
     * Set YaRN: extrapolation mix factor (default: 0.0 = full interpolation).
     */
    public ModelParameters setYarnExtFactor(float yarnExtFactor) {
        parameters.put("--yarn-ext-factor", String.valueOf(yarnExtFactor));
        return this;
    }

    /**
     * Set YaRN: scale sqrt(t) or attention magnitude (default: 1.0).
     */
    public ModelParameters setYarnAttnFactor(float yarnAttnFactor) {
        parameters.put("--yarn-attn-factor", String.valueOf(yarnAttnFactor));
        return this;
    }

    /**
     * Set YaRN: high correction dim or alpha (default: 1.0).
     */
    public ModelParameters setYarnBetaSlow(float yarnBetaSlow) {
        parameters.put("--yarn-beta-slow", String.valueOf(yarnBetaSlow));
        return this;
    }

    /**
     * Set YaRN: low correction dim or beta (default: 32.0).
     */
    public ModelParameters setYarnBetaFast(float yarnBetaFast) {
        parameters.put("--yarn-beta-fast", String.valueOf(yarnBetaFast));
        return this;
    }

    /**
     * Set group-attention factor (default: 1).
     */
    public ModelParameters setGrpAttnN(int grpAttnN) {
        parameters.put("--grp-attn-n", String.valueOf(grpAttnN));
        return this;
    }

    /**
     * Set group-attention width (default: 512).
     */
    public ModelParameters setGrpAttnW(int grpAttnW) {
        parameters.put("--grp-attn-w", String.valueOf(grpAttnW));
        return this;
    }

    /**
     * Enable verbose printing of the KV cache.
     */
    public ModelParameters enableDumpKvCache() {
        parameters.put("--dump-kv-cache", null);
        return this;
    }

    /**
     * Disable KV offload.
     */
    public ModelParameters disableKvOffload() {
        parameters.put("--no-kv-offload", null);
        return this;
    }

    /**
     * Set KV cache data type for K (allowed values: F16).
     */
    public ModelParameters setCacheTypeK(CacheType type) {
        parameters.put("--cache-type-k", type.name().toLowerCase());
        return this;
    }

    /**
     * Set KV cache data type for V (allowed values: F16).
     */
    public ModelParameters setCacheTypeV(CacheType type) {
        parameters.put("--cache-type-v", type.name().toLowerCase());
        return this;
    }

    /**
     * Set KV cache defragmentation threshold (default: 0.1, < 0 - disabled).
     */
    public ModelParameters setDefragThold(float defragThold) {
        parameters.put("--defrag-thold", String.valueOf(defragThold));
        return this;
    }

    /**
     * Set the number of parallel sequences to decode (default: 1).
     */
    public ModelParameters setParallel(int nParallel) {
        parameters.put("--parallel", String.valueOf(nParallel));
        return this;
    }

    /**
     * Enable continuous batching (a.k.a dynamic batching) (default: disabled).
     */
    public ModelParameters enableContBatching() {
        parameters.put("--cont-batching", null);
        return this;
    }

    /**
     * Disable continuous batching.
     */
    public ModelParameters disableContBatching() {
        parameters.put("--no-cont-batching", null);
        return this;
    }

    /**
     * Force system to keep model in RAM rather than swapping or compressing.
     */
    public ModelParameters enableMlock() {
        parameters.put("--mlock", null);
        return this;
    }

    /**
     * Do not memory-map model (slower load but may reduce pageouts if not using mlock).
     */
    public ModelParameters disableMmap() {
        parameters.put("--no-mmap", null);
        return this;
    }

    /**
     * Set NUMA optimization type for system.
     */
    public ModelParameters setNuma(NumaStrategy numaStrategy) {
        parameters.put("--numa", numaStrategy.name().toLowerCase());
        return this;
    }

    /**
     * Set comma-separated list of devices to use for offloading <dev1,dev2,..> (none = don't offload).
     */
    public ModelParameters setDevices(String devices) {
        parameters.put("--device", devices);
        return this;
    }

    /**
     * Set the number of layers to store in VRAM.
     */
    public ModelParameters setGpuLayers(int gpuLayers) {
        parameters.put("--gpu-layers", String.valueOf(gpuLayers));
        return this;
    }

    /**
     * Set how to split the model across multiple GPUs (none, layer, row).
     */
    public ModelParameters setSplitMode(GpuSplitMode splitMode) {
        parameters.put("--split-mode", splitMode.name().toLowerCase());
        return this;
    }

    /**
     * Set fraction of the model to offload to each GPU, comma-separated list of proportions N0,N1,N2,....
     */
    public ModelParameters setTensorSplit(String tensorSplit) {
        parameters.put("--tensor-split", tensorSplit);
        return this;
    }

    /**
     * Set the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row).
     */
    public ModelParameters setMainGpu(int mainGpu) {
        parameters.put("--main-gpu", String.valueOf(mainGpu));
        return this;
    }

    /**
     * Enable checking model tensor data for invalid values.
     */
    public ModelParameters enableCheckTensors() {
        parameters.put("--check-tensors", null);
        return this;
    }

    /**
     * Override model metadata by key. This option can be specified multiple times.
     */
    public ModelParameters setOverrideKv(String keyValue) {
        parameters.put("--override-kv", keyValue);
        return this;
    }

    /**
     * Add a LoRA adapter (can be repeated to use multiple adapters).
     */
    public ModelParameters addLoraAdapter(String fname) {
        parameters.put("--lora", fname);
        return this;
    }

    /**
     * Add a LoRA adapter with user-defined scaling (can be repeated to use multiple adapters).
     */
    public ModelParameters addLoraScaledAdapter(String fname, float scale) {
        parameters.put("--lora-scaled", fname + "," + scale);
        return this;
    }

    /**
     * Add a control vector (this argument can be repeated to add multiple control vectors).
     */
    public ModelParameters addControlVector(String fname) {
        parameters.put("--control-vector", fname);
        return this;
    }

    /**
     * Add a control vector with user-defined scaling (can be repeated to add multiple scaled control vectors).
     */
    public ModelParameters addControlVectorScaled(String fname, float scale) {
        parameters.put("--control-vector-scaled", fname + "," + scale);
        return this;
    }

    /**
     * Set the layer range to apply the control vector(s) to (start and end inclusive).
     */
    public ModelParameters setControlVectorLayerRange(int start, int end) {
        parameters.put("--control-vector-layer-range", start + "," + end);
        return this;
    }

    /**
     * Set the model path from which to load the base model.
     */
    public ModelParameters setModel(String model) {
        parameters.put("--model", model);
        return this;
    }

    /**
     * Set the model download URL (default: unused).
     */
    public ModelParameters setModelUrl(String modelUrl) {
        parameters.put("--model-url", modelUrl);
        return this;
    }

    /**
     * Set the Hugging Face model repository (default: unused).
     */
    public ModelParameters setHfRepo(String hfRepo) {
        parameters.put("--hf-repo", hfRepo);
        return this;
    }

    /**
     * Set the Hugging Face model file (default: unused).
     */
    public ModelParameters setHfFile(String hfFile) {
        parameters.put("--hf-file", hfFile);
        return this;
    }

    /**
     * Set the Hugging Face model repository for the vocoder model (default: unused).
     */
    public ModelParameters setHfRepoV(String hfRepoV) {
        parameters.put("--hf-repo-v", hfRepoV);
        return this;
    }

    /**
     * Set the Hugging Face model file for the vocoder model (default: unused).
     */
    public ModelParameters setHfFileV(String hfFileV) {
        parameters.put("--hf-file-v", hfFileV);
        return this;
    }

    /**
     * Set the Hugging Face access token (default: value from HF_TOKEN environment variable).
     */
    public ModelParameters setHfToken(String hfToken) {
        parameters.put("--hf-token", hfToken);
        return this;
    }

    /**
     * Enable embedding use case; use only with dedicated embedding models.
     */
    public ModelParameters enableEmbedding() {
        parameters.put("--embedding", null);
        return this;
    }

    /**
     * Enable reranking endpoint on server.
     */
    public ModelParameters enableReranking() {
        parameters.put("--reranking", null);
        return this;
    }

    /**
     * Set minimum chunk size to attempt reusing from the cache via KV shifting.
     */
    public ModelParameters setCacheReuse(int cacheReuse) {
        parameters.put("--cache-reuse", String.valueOf(cacheReuse));
        return this;
    }

    /**
     * Set the path to save the slot kv cache.
     */
    public ModelParameters setSlotSavePath(String slotSavePath) {
        parameters.put("--slot-save-path", slotSavePath);
        return this;
    }

    /**
     * Set custom jinja chat template.
     */
    public ModelParameters setChatTemplate(String chatTemplate) {
        parameters.put("--chat-template", chatTemplate);
        return this;
    }

    /**
     * Set how much the prompt of a request must match the prompt of a slot in order to use that slot.
     */
    public ModelParameters setSlotPromptSimilarity(float similarity) {
        parameters.put("--slot-prompt-similarity", String.valueOf(similarity));
        return this;
    }

    /**
     * Load LoRA adapters without applying them (apply later via POST /lora-adapters).
     */
    public ModelParameters setLoraInitWithoutApply() {
        parameters.put("--lora-init-without-apply", null);
        return this;
    }

    /**
     * Disable logging.
     */
    public ModelParameters disableLog() {
        parameters.put("--log-disable", null);
        return this;
    }

    /**
     * Set the log file path.
     */
    public ModelParameters setLogFile(String logFile) {
        parameters.put("--log-file", logFile);
        return this;
    }

    /**
     * Set verbosity level to infinity (log all messages, useful for debugging).
     */
    public ModelParameters setVerbose() {
        parameters.put("--verbose", null);
        return this;
    }

    /**
     * Set the verbosity threshold (messages with a higher verbosity will be ignored).
     */
    public ModelParameters setLogVerbosity(int verbosity) {
        parameters.put("--log-verbosity", String.valueOf(verbosity));
        return this;
    }

    /**
     * Enable prefix in log messages.
     */
    public ModelParameters enableLogPrefix() {
        parameters.put("--log-prefix", null);
        return this;
    }

    /**
     * Enable timestamps in log messages.
     */
    public ModelParameters enableLogTimestamps() {
        parameters.put("--log-timestamps", null);
        return this;
    }

    /**
     * Set the number of tokens to draft for speculative decoding.
     */
    public ModelParameters setDraftMax(int draftMax) {
        parameters.put("--draft-max", String.valueOf(draftMax));
        return this;
    }

    /**
     * Set the minimum number of draft tokens to use for speculative decoding.
     */
    public ModelParameters setDraftMin(int draftMin) {
        parameters.put("--draft-min", String.valueOf(draftMin));
        return this;
    }

    /**
     * Set the minimum speculative decoding probability for greedy decoding.
     */
    public ModelParameters setDraftPMin(float draftPMin) {
        parameters.put("--draft-p-min", String.valueOf(draftPMin));
        return this;
    }

    /**
     * Set the size of the prompt context for the draft model.
     */
    public ModelParameters setCtxSizeDraft(int ctxSizeDraft) {
        parameters.put("--ctx-size-draft", String.valueOf(ctxSizeDraft));
        return this;
    }

    /**
     * Set the comma-separated list of devices to use for offloading the draft model.
     */
    public ModelParameters setDeviceDraft(String deviceDraft) {
        parameters.put("--device-draft", deviceDraft);
        return this;
    }

    /**
     * Set the number of layers to store in VRAM for the draft model.
     */
    public ModelParameters setGpuLayersDraft(int gpuLayersDraft) {
        parameters.put("--gpu-layers-draft", String.valueOf(gpuLayersDraft));
        return this;
    }

    /**
     * Set the draft model for speculative decoding.
     */
    public ModelParameters setModelDraft(String modelDraft) {
        parameters.put("--model-draft", modelDraft);
        return this;
    }

}
