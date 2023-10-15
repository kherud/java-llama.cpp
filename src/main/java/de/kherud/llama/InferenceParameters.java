package de.kherud.llama;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.Map;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Parameters used throughout inference of a {@link LlamaModel}, e.g., {@link LlamaModel#generate(String)} and
 * {@link LlamaModel#complete(String)}.
 */
public final class InferenceParameters {

	public final int nPredict;   // new tokens to predict
	public final int nKeep;    // number of tokens to keep from initial prompt
	public final int nProbs;    // if greater than 0, output the probabilities of top nProbs tokens.
	@Nullable
	public final Map<Integer, Float> logitBias; // logit bias for specific tokens
	public final int topK; // <= 0 to use vocab size
	public final float topP; // 1.0 = disabled
	public final float tfsZ; // 1.0 = disabled
	public final float typicalP; // 1.0 = disabled
	public final float temperature; // 1.0 = disabled
	public final float repeatPenalty; // 1.0 = disabled
	public final int repeatLastN; // last n tokens to penalize (0 = disable penalty, -1 = context size)
	public final float frequencyPenalty; // 0.0 = disabled
	public final float presencePenalty; // 0.0 = disabled
	public final boolean penalizeNL; // 0.0 = disabled
	public final boolean ignoreEos;
	public final int mirostat; // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
	public final float mirostatTau; // target entropy
	public final float mirostatEta; // learning rate
	public final boolean beamSearch;
	public final int nBeams;
	@Nullable
	public final String grammar;  // optional BNF-like grammar to constrain sampling
	@Nullable
	public final String[] antiprompt; // string upon seeing which more user input is prompted
	public final int seed;

	/**
	 * Private constructor to build immutable parameters object. Called via {@link Builder}.
	 */
	private InferenceParameters(
			int nPredict,
			int nKeep,
			int nProbs,
			@Nullable Map<Integer, Float> logitBias,
			int topK,
			float topP,
			float tfsZ,
			float typicalP,
			float temperature,
			float repeatPenalty,
			int repeatLastN,
			float frequencyPenalty,
			float presencePenalty,
			boolean penalizeNL,
			boolean ignoreEos,
			MiroStat mirostat,
			float mirostatTau,
			float mirostatEta,
			boolean beamSearch,
			int nBeams,
			@Nullable String grammar,
			@Nullable String[] antiprompt,
			int seed
	) {
		this.nPredict = nPredict;
		this.nKeep = nKeep;
		this.nProbs = nProbs;
		this.logitBias = logitBias;
		this.topK = topK;
		this.topP = topP;
		this.tfsZ = tfsZ;
		this.typicalP = typicalP;
		this.temperature = temperature;
		this.repeatPenalty = repeatPenalty;
		this.repeatLastN = repeatLastN;
		this.frequencyPenalty = frequencyPenalty;
		this.presencePenalty = presencePenalty;
		this.penalizeNL = penalizeNL;
		this.ignoreEos = ignoreEos;
		this.mirostat = mirostat.level;
		this.mirostatTau = mirostatTau;
		this.mirostatEta = mirostatEta;
		this.beamSearch = beamSearch;
		this.nBeams = nBeams;
		this.grammar = grammar;
		this.antiprompt = antiprompt;
		this.seed = seed;
	}

	/**
	 * The builder class used for creating new {@link InferenceParameters} of a {@link LlamaModel}.
	 */
	public static class Builder {

		private int nPredict = -1;   // new tokens to predict
		private int nKeep = 0;    // number of tokens to keep from initial prompt
		private int nProbs = 0;    // if greater than 0, output the probabilities of top nProbs tokens.

		// sampling parameters
		private Map<Integer, Float> logitBias = null; // logit bias for specific tokens
		private int topK = 40;    // <= 0 to use vocab size
		private float topP = 0.95f; // 1.0 = disabled
		private float tfsZ = 1.00f; // 1.0 = disabled
		private float typicalP = 1.00f; // 1.0 = disabled
		private float temperature = 0.80f; // 1.0 = disabled
		private float repeatPenalty = 1.10f; // 1.0 = disabled
		private int repeatLastN = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
		private float frequencyPenalty = 0.00f; // 0.0 = disabled
		private float presencePenalty = 0.00f; // 0.0 = disabled
		private boolean penalizeNl = false;  // consider newlines as a repeatable token
		private boolean ignoreEos = false;
		private MiroStat mirostat = MiroStat.Disabled;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
		private float mirostatTau = 5.00f; // target entropy
		private float mirostatEta = 0.10f; // learning rate
		private boolean beamSearch = false;
		private int nBeams = 2;

		private String grammar = null;  // optional BNF-like grammar to constrain sampling
		private String[] antiPrompt = null; // string upon seeing which more user input is prompted

		private int seed = 42;

		/**
		 * Constructs the immutable {@link InferenceParameters} objects with the configured options.
		 * Note, that all options not configured have sensible defaults.
		 *
		 * @return an immutable parameters object
		 */
		public InferenceParameters build() {
			return new InferenceParameters(
					nPredict,
					nKeep,
					nProbs,
					logitBias,
					topK,
					topP,
					tfsZ,
					typicalP,
					temperature,
					repeatPenalty,
					repeatLastN,
					frequencyPenalty,
					presencePenalty,
					penalizeNl,
					ignoreEos,
					mirostat,
					mirostatTau,
					mirostatEta,
					beamSearch,
					nBeams,
					grammar,
					antiPrompt,
					seed
			);
		}

		public Builder setNPredict(int nPredict) {
			this.nPredict = nPredict;
			return this;
		}

		public Builder setNKeep(int nKeep) {
			this.nKeep = nKeep;
			return this;
		}

		public Builder setNProbs(int nProbs) {
			this.nProbs = nProbs;
			return this;
		}

		public Builder setLogitBias(@NotNull Map<Integer, Float> logitBias) {
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

		public Builder setTemperature(float temperature) {
			this.temperature = temperature;
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

		public Builder setPenalizeNl(boolean penalizeNl) {
			this.penalizeNl = penalizeNl;
			return this;
		}

		public Builder setIgnoreEos(boolean ignoreEos) {
			this.ignoreEos = ignoreEos;
			return this;
		}

		public Builder setMirostat(MiroStat mode) {
			this.mirostat = mode;
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

		public Builder setBeamSearch(boolean beamSearch) {
			this.beamSearch = beamSearch;
			return this;
		}

		public Builder setNBeams(int nBeams) {
			this.nBeams = nBeams;
			return this;
		}

		// default charset usage for Java backwards compatibility
		@SuppressWarnings("ImplicitDefaultCharsetUsage")
		public Builder setGrammar(@NotNull File file) throws IOException {
			StringBuilder grammarBuilder = new StringBuilder();
			try (BufferedReader br = new BufferedReader(new FileReader(file))) {
				String currentLine;
				while ((currentLine = br.readLine()) != null) {
					grammarBuilder.append(currentLine).append("\n");
				}
			}
			return setGrammar(grammarBuilder.toString());
		}

		public Builder setGrammar(@Nullable String grammar) {
			this.grammar = grammar;
			return this;
		}

		public Builder setAntiPrompt(@NotNull String... antiPrompt) {
			this.antiPrompt = antiPrompt;
			return this;
		}

		public Builder setSeed(int seed) {
			this.seed = seed;
			return this;
		}
	}

	public enum MiroStat {

		Disabled(0),
		V1(1),
		V2(2);

		private final int level;

		MiroStat(int level) {
			this.level = level;
		}
	}
}
