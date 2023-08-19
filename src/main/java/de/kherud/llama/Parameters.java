package de.kherud.llama;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.FloatByReference;

import de.kherud.llama.foreign.LlamaLibrary;
import de.kherud.llama.foreign.NativeSize;
import de.kherud.llama.foreign.llama_context_params;

public class Parameters {

	public final llama_context_params.ByValue ctx;
	public final int nThreads;
	public final int nPredict;   // new tokens to predict
	public final int nKeep;    // number of tokens to keep from initial prompt
	public final int nChunks;   // max number of chunks to process (-1 = unlimited)
	public final int nProbs;    // if greater than 0, output the probabilities of top nProbs tokens.
	public final Map<Integer, Float> logitBias; // logit bias for specific tokens
	public final int topK; // <= 0 to use vocab size
	public final float topP; // 1.0 = disabled
	public final float tfsZ; // 1.0 = disabled
	public final float typicalP; // 1.0 = disabled
	public final float temp; // 1.0 = disabled
	public final float repeatPenalty; // 1.0 = disabled
	public final int repeatLastN; // last n tokens to penalize (0 = disable penalty, -1 = context size)
	public final float frequencyPenalty; // 0.0 = disabled
	public final float presencePenalty; // 0.0 = disabled
	public final int mirostat; // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
	public final float mirostatTau; // target entropy
	public final float mirostatEta; // learning rate
	public final String cfgNegativePrompt;       // string to help guidance
	public final float cfgScale;   // How strong is guidance
	public final String pathPromptCache;  // path to file for saving/loading prompt eval state
	public final String inputPrefix;  // string to prefix user inputs with
	public final String inputSuffix;  // string to suffix user inputs with
	public final String grammar;  // optional BNF-like grammar to constrain sampling
	public final List<String> antiprompt; // string upon seeing which more user input is prompted
	public final String loraAdapter;  // lora adapter path
	public final String loraBase;  // base model path for the lora adapter
	public final boolean hellaswag; // compute HellaSwag score over random tasks from datafile supplied in prompt
	public final NativeSize hellaswagTasks;   // number of tasks to use when computing the HellaSwag score
	public final boolean memoryF16;  // use f16 instead of f32 for memory kv
	public final boolean randomPrompt; // do not randomize prompt if none provided
	public final boolean useColor; // use color to distinguish generations and inputs
	public final boolean interactive; // interactive mode
	public final boolean promptCacheAll; // save user input and generations to prompt cache
	public final boolean promptCacheRo; // open the prompt cache read-only and do not update it
	public final boolean interactiveFirst; // wait for user input immediately
	public final boolean multilineInput; // reverse the usage of `\`
	public final boolean simpleIo; // improves compatibility with subprocesses and limited consoles
	public final boolean inputPrefixBos; // prefix BOS to user inputs, preceding inputPrefix
	public final boolean instruct; // instruction mode (used for Alpaca models)
	public final boolean penalizeNl;  // consider newlines as a repeatable token
	public final boolean perplexity; // compute perplexity over the prompt
	public final boolean memTest; // compute maximum memory usage
	public final boolean numa; // attempt optimizations that help on some NUMA systems
	public final boolean exportCgraph; // export the computation graph
	public final boolean verbosePrompt; // print prompt tokens before generation

	private Parameters(
			llama_context_params.ByValue ctx,
			int nThreads,
			int nPredict,
			int nKeep,
			int nChunks,
			int nProbs,
			Map<Integer, Float> logitBias,
			int topK,
			float topP,
			float tfsZ,
			float typicalP,
			float temp,
			float repeatPenalty,
			int repeatLastN,
			float frequencyPenalty,
			float presencePenalty,
			int mirostat,
			float mirostatTau,
			float mirostatEta,
			String cfgNegativePrompt,
			float cfgScale,
			String pathPromptCache,
			String inputPrefix,
			String inputSuffix,
			String grammar,
			List<String> antiprompt,
			String loraAdapter,
			String loraBase,
			boolean hellaswag,
			NativeSize hellaswagTasks,
			boolean memoryF16,
			boolean randomPrompt,
			boolean useColor,
			boolean interactive,
			boolean promptCacheAll,
			boolean promptCacheRo,
			boolean interactiveFirst,
			boolean multilineInput,
			boolean simpleIo,
			boolean inputPrefixBos,
			boolean instruct,
			boolean penalizeNl,
			boolean perplexity,
			boolean memTest,
			boolean numa,
			boolean exportCgraph,
			boolean verbosePrompt
	) {
		this.ctx = ctx;
		this.nThreads = nThreads;
		this.nPredict = nPredict;
		this.nKeep = nKeep;
		this.nChunks = nChunks;
		this.nProbs = nProbs;
		this.logitBias = logitBias;
		this.topK = topK;
		this.topP = topP;
		this.tfsZ = tfsZ;
		this.typicalP = typicalP;
		this.temp = temp;
		this.repeatPenalty = repeatPenalty;
		this.repeatLastN = repeatLastN;
		this.frequencyPenalty = frequencyPenalty;
		this.presencePenalty = presencePenalty;
		this.mirostat = mirostat;
		this.mirostatTau = mirostatTau;
		this.mirostatEta = mirostatEta;
		this.cfgNegativePrompt = cfgNegativePrompt;
		this.cfgScale = cfgScale;
		this.pathPromptCache = pathPromptCache;
		this.inputPrefix = inputPrefix;
		this.inputSuffix = inputSuffix;
		this.grammar = grammar;
		this.antiprompt = antiprompt;
		this.loraAdapter = loraAdapter;
		this.loraBase = loraBase;
		this.hellaswag = hellaswag;
		this.hellaswagTasks = hellaswagTasks;
		this.memoryF16 = memoryF16;
		this.randomPrompt = randomPrompt;
		this.useColor = useColor;
		this.interactive = interactive;
		this.promptCacheAll = promptCacheAll;
		this.promptCacheRo = promptCacheRo;
		this.interactiveFirst = interactiveFirst;
		this.multilineInput = multilineInput;
		this.simpleIo = simpleIo;
		this.inputPrefixBos = inputPrefixBos;
		this.instruct = instruct;
		this.penalizeNl = penalizeNl;
		this.perplexity = perplexity;
		this.memTest = memTest;
		this.numa = numa;
		this.exportCgraph = exportCgraph;
		this.verbosePrompt = verbosePrompt;
	}

	public static class Builder {

		private final llama_context_params.ByValue ctxParams = LlamaLibrary.llama_context_default_params();

		private int nThreads = Runtime.getRuntime().availableProcessors();
		private int nPredict = -1;   // new tokens to predict
		private int nKeep = 0;    // number of tokens to keep from initial prompt
		private int nChunks = -1;   // max number of chunks to process (-1 = unlimited)
		private int nProbs = 0;    // if greater than 0, output the probabilities of top nProbs tokens.

		// sampling parameters
		private Map<Integer, Float> logitBias; // logit bias for specific tokens
		private int topK = 40;    // <= 0 to use vocab size
		private float topP = 0.95f; // 1.0 = disabled
		private float tfsZ = 1.00f; // 1.0 = disabled
		private float typicalP = 1.00f; // 1.0 = disabled
		private float temp = 0.80f; // 1.0 = disabled
		private float repeatPenalty = 1.10f; // 1.0 = disabled
		private int repeatLastN = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
		private float frequencyPenalty = 0.00f; // 0.0 = disabled
		private float presencePenalty = 0.00f; // 0.0 = disabled
		private int mirostat = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
		private float mirostatTau = 5.00f; // target entropy
		private float mirostatEta = 0.10f; // learning rate

		// Classifier-Free Guidance
		// https://arxiv.org/abs/2306.17806
		private String cfgNegativePrompt;       // string to help guidance
		private float cfgScale = 1.f;   // How strong is guidance

		private String model = "models/7B/ggml-model.bin"; // model path
		private String modelAlias = "unknown"; // model alias
		private String prompt = "";
		private String pathPromptCache = "";  // path to file for saving/loading prompt eval state
		private String inputPrefix = "";  // string to prefix user inputs with
		private String inputSuffix = "";  // string to suffix user inputs with
		private String grammar = "";  // optional BNF-like grammar to constrain sampling
		private List<String> antiprompt; // string upon seeing which more user input is prompted

		private String loraAdapter = "";  // lora adapter path
		private String loraBase = "";  // base model path for the lora adapter

		private boolean hellaswag = false; // compute HellaSwag score over random tasks from datafile supplied in prompt
		private NativeSize hellaswagTasks = new NativeSize(400);   // number of tasks to use when computing the HellaSwag score

		private boolean memoryF16 = true;  // use f16 instead of f32 for memory kv
		private boolean randomPrompt = false; // do not randomize prompt if none provided
		private boolean useColor = false; // use color to distinguish generations and inputs
		private boolean interactive = false; // interactive mode
		private boolean promptCacheAll = false; // save user input and generations to prompt cache
		private boolean promptCacheRo = false; // open the prompt cache read-only and do not update it

		private boolean interactiveFirst = false; // wait for user input immediately
		private boolean multilineInput = false; // reverse the usage of `\`
		private boolean simpleIo = false; // improves compatibility with subprocesses and limited consoles

		private boolean inputPrefixBos = false; // prefix BOS to user inputs, preceding inputPrefix
		private boolean instruct = false; // instruction mode (used for Alpaca models)
		private boolean penalizeNl = true;  // consider newlines as a repeatable token
		private boolean perplexity = false; // compute perplexity over the prompt
		private boolean memTest = false; // compute maximum memory usage
		private boolean numa = false; // attempt optimizations that help on some NUMA systems
		private boolean exportCgraph = false; // export the computation graph
		private boolean verbosePrompt = false; // print prompt tokens before generation

		public Parameters build() {
			return new Parameters(
					ctxParams,
					nThreads,
					nPredict,
					nKeep,
					nChunks,
					nProbs,
					logitBias,
					topK,
					topP,
					tfsZ,
					typicalP,
					temp,
					repeatPenalty,
					repeatLastN,
					frequencyPenalty,
					presencePenalty,
					mirostat,
					mirostatTau,
					mirostatEta,
					cfgNegativePrompt,
					cfgScale,
					pathPromptCache,
					inputPrefix,
					inputSuffix,
					grammar,
					antiprompt,
					loraAdapter,
					loraBase,
					hellaswag,
					hellaswagTasks,
					memoryF16,
					randomPrompt,
					useColor,
					interactive,
					promptCacheAll,
					promptCacheRo,
					interactiveFirst,
					multilineInput,
					simpleIo,
					inputPrefixBos,
					instruct,
					penalizeNl,
					perplexity,
					memTest,
					numa,
					exportCgraph,
					verbosePrompt
			);
		}

		public Builder setnThreads(int nThreads) {
			this.nThreads = nThreads;
			return this;
		}

		public Builder setnPredict(int nPredict) {
			this.nPredict = nPredict;
			return this;
		}

		public Builder setnKeep(int nKeep) {
			this.nKeep = nKeep;
			return this;
		}

		public Builder setnChunks(int nChunks) {
			this.nChunks = nChunks;
			return this;
		}

		public Builder setnProbs(int nProbs) {
			this.nProbs = nProbs;
			return this;
		}

		public Builder setLogitBias(Map<Integer, Float> logitBias) {
			this.logitBias = Collections.unmodifiableMap(logitBias);
			return this;
		}

		public Builder setTopK(int topK) {
			this.topK = topK;
			return this;
		}

		public Builder setTopP(float topP) {
			this.topP = topP;
			return this;
		}

		public Builder setTfsZ(float tfsZ) {
			this.tfsZ = tfsZ;
			return this;
		}

		public Builder setTypicalP(float typicalP) {
			this.typicalP = typicalP;
			return this;
		}

		public Builder setTemp(float temp) {
			this.temp = temp;
			return this;
		}

		public Builder setRepeatPenalty(float repeatPenalty) {
			this.repeatPenalty = repeatPenalty;
			return this;
		}

		public Builder setRepeatLastN(int repeatLastN) {
			this.repeatLastN = repeatLastN;
			return this;
		}

		public Builder setFrequencyPenalty(float frequencyPenalty) {
			this.frequencyPenalty = frequencyPenalty;
			return this;
		}

		public Builder setPresencePenalty(float presencePenalty) {
			this.presencePenalty = presencePenalty;
			return this;
		}

		public Builder setMirostat(int mirostat) {
			this.mirostat = mirostat;
			return this;
		}

		public Builder setMirostatTau(float mirostatTau) {
			this.mirostatTau = mirostatTau;
			return this;
		}

		public Builder setMirostatEta(float mirostatEta) {
			this.mirostatEta = mirostatEta;
			return this;
		}

		public Builder setCfgNegativePrompt(String cfgNegativePrompt) {
			this.cfgNegativePrompt = cfgNegativePrompt;
			return this;
		}

		public Builder setCfgScale(float cfgScale) {
			this.cfgScale = cfgScale;
			return this;
		}

		public Builder setModel(String model) {
			this.model = model;
			return this;
		}

		public Builder setModelAlias(String modelAlias) {
			this.modelAlias = modelAlias;
			return this;
		}

		public Builder setPrompt(String prompt) {
			this.prompt = prompt;
			return this;
		}

		public Builder setPathPromptCache(String pathPromptCache) {
			this.pathPromptCache = pathPromptCache;
			return this;
		}

		public Builder setInputPrefix(String inputPrefix) {
			this.inputPrefix = inputPrefix;
			return this;
		}

		public Builder setInputSuffix(String inputSuffix) {
			this.inputSuffix = inputSuffix;
			return this;
		}

		public Builder setGrammar(String grammar) {
			this.grammar = grammar;
			return this;
		}

		public Builder setAntiprompt(String[] antiprompt) {
			this.antiprompt = Collections.unmodifiableList(Arrays.asList(antiprompt));
			return this;
		}

		public Builder setLoraAdapter(String loraAdapter) {
			this.loraAdapter = loraAdapter;
			return this;
		}

		public Builder setLoraBase(String loraBase) {
			this.loraBase = loraBase;
			return this;
		}

		public Builder setHellaswag(boolean hellaswag) {
			this.hellaswag = hellaswag;
			return this;
		}

		public Builder setHellaswagTasks(long hellaswagTasks) {
			this.hellaswagTasks = new NativeSize(hellaswagTasks);
			return this;
		}

		public Builder setMemoryF16(boolean memoryF16) {
			this.memoryF16 = memoryF16;
			return this;
		}

		public Builder setRandomPrompt(boolean randomPrompt) {
			this.randomPrompt = randomPrompt;
			return this;
		}

		public Builder setUseColor(boolean useColor) {
			this.useColor = useColor;
			return this;
		}

		public Builder setInteractive(boolean interactive) {
			this.interactive = interactive;
			return this;
		}

		public Builder setPromptCacheAll(boolean promptCacheAll) {
			this.promptCacheAll = promptCacheAll;
			return this;
		}

		public Builder setPromptCacheRo(boolean promptCacheRo) {
			this.promptCacheRo = promptCacheRo;
			return this;
		}

		public Builder setInteractiveFirst(boolean interactiveFirst) {
			this.interactiveFirst = interactiveFirst;
			return this;
		}

		public Builder setMultilineInput(boolean multilineInput) {
			this.multilineInput = multilineInput;
			return this;
		}

		public Builder setSimpleIo(boolean simpleIo) {
			this.simpleIo = simpleIo;
			return this;
		}

		public Builder setInputPrefixBos(boolean inputPrefixBos) {
			this.inputPrefixBos = inputPrefixBos;
			return this;
		}

		public Builder setInstruct(boolean instruct) {
			this.instruct = instruct;
			return this;
		}

		public Builder setPenalizeNl(boolean penalizeNl) {
			this.penalizeNl = penalizeNl;
			return this;
		}

		public Builder setPerplexity(boolean perplexity) {
			this.perplexity = perplexity;
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

		public Builder setExportCgraph(boolean exportCgraph) {
			this.exportCgraph = exportCgraph;
			return this;
		}

		public Builder setVerbosePrompt(boolean verbosePrompt) {
			this.verbosePrompt = verbosePrompt;
			return this;
		}

		public Builder setSeed(int seed) {
			ctxParams.setSeed(seed);
			return this;
		}

		public Builder setnCtx(int n_ctx) {
			ctxParams.setN_ctx(n_ctx);
			return this;
		}

		public Builder setnBbatch(int n_batch) {
			ctxParams.setN_batch(n_batch);
			return this;
		}

		public Builder setnGqa(int n_gqa) {
			ctxParams.setN_gqa(n_gqa);
			return this;
		}

		public Builder setRmsNormEps(float rms_norm_eps) {
			ctxParams.setRms_norm_eps(rms_norm_eps);
			return this;
		}

		public Builder setnGpuLayers(int n_gpu_layers) {
			ctxParams.setN_gpu_layers(n_gpu_layers);
			return this;
		}

		public Builder setMainGpu(int main_gpu) {
			ctxParams.setMain_gpu(main_gpu);
			return this;
		}

		public Builder setTensorSplit(float[] tensor_split) {
			Memory memory = new Memory(tensor_split.length * 4L); // Allocate native memory, 4 bytes per float
			for (int i = 0; i < tensor_split.length; i++) {
				memory.setFloat(i * 4L, tensor_split[i]); // Copy the float values to native memory
			}
			FloatByReference floatByRef = new FloatByReference();
			floatByRef.setPointer(memory);
			ctxParams.setTensor_split(floatByRef);
			return this;
		}

		public Builder setRopeFreqBase(float rope_freq_base) {
			ctxParams.setRope_freq_base(rope_freq_base);
			return this;
		}

		public Builder setRopeFreqScale(float rope_freq_scale) {
			ctxParams.setRope_freq_scale(rope_freq_scale);
			return this;
		}

		public Builder setProgressCallback(LlamaLibrary.llama_progress_callback progress_callback) {
			ctxParams.setProgress_callback(progress_callback);
			return this;
		}

		public Builder setProgressCallbackUserData(Pointer progress_callback_user_data) {
			ctxParams.setProgress_callback_user_data(progress_callback_user_data);
			return this;
		}

		/**
		 * use fp16 for KV cache
		 */
		public Builder setF16Kv(byte f16_kv) {
			ctxParams.setF16_kv(f16_kv);
			return this;
		}

		/**
		 * the llama_eval() call computes all logits, not just the last one
		 */
		public Builder setLogitsAll(byte logits_all) {
			ctxParams.setLogits_all(logits_all);
			return this;
		}

		/**
		 * only load the vocabulary, no weights
		 */
		public Builder setVocabOnly(byte vocab_only) {
			ctxParams.setVocab_only(vocab_only);
			return this;
		}

		/**
		 * use mmap if possible
		 */
		public Builder setUseMmap(byte use_mmap) {
			ctxParams.setUse_mmap(use_mmap);
			return this;
		}

		/**
		 * force system to keep model in RAM
		 */
		public Builder setUseMLock(byte use_mlock) {
			ctxParams.setUse_mlock(use_mlock);
			return this;
		}

		/**
		 * embedding mode only
		 */
		public Builder setEmbedding(byte embedding) {
			ctxParams.setEmbedding(embedding);
			return this;
		}
	}
}
