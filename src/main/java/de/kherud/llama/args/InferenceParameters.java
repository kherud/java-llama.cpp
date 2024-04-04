package de.kherud.llama.args;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.annotation.Native;
import java.util.Collections;
import java.util.Map;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import de.kherud.llama.LlamaModel;

/**
 * Parameters used throughout inference of a {@link LlamaModel}, e.g., {@link LlamaModel#generate(String)} and
 * {@link LlamaModel#complete(String)}.
 */
public final class InferenceParameters {

	@Native
	private int nPredict = -1; // new tokens to predict
	@Native
	private boolean cachePrompt = false;
	// number of tokens to keep from initial prompt
	@Native
	private int nKeep = 0;
	@Native
	private int nDiscard = 0;
	private int minKeep = 0;
	// if greater than 0, output the probabilities of top nProbs tokens.
	@Native
	private int nProbs = 0;
	// logit bias for specific tokens
	@Nullable
	@Native
	private Map<Integer, Float> logitBias = null;
	// <= 0 to use vocab size
	@Native
	private int topK = 40;
	// 1.0 = disabled
	@Native
	private float topP = 0.95f;
	@Native
	private float minP = 0.05f;
	// 1.0 = disabled
	@Native
	private float tfsZ = 1.00f;
	// 1.0 = disabled
	@Native
	private float typicalP = 1.00f;
	// 1.0 = disabled
	@Native
	private float temperature = 0.80f;
	private float dynamicTemperatureRange = 0.00f;
	private float dynamicTemperatureExponent = 1.00f;
	// 1.0 = disabled
	@Native
	private float repeatPenalty = 1.10f;
	// last n tokens to penalize (0 = disable penalty, -1 = context size)
	@Native
	private int repeatLastN = 64;
	// 0.0 = disabled
	@Native
	private float frequencyPenalty = 0.00f;
	// 0.0 = disabled
	@Native
	private float presencePenalty = 0.00f;
	// 0.0 = disabled
	@Native
	private boolean penalizeNl = false;
	@Native
	private boolean ignoreEos = false;
	// 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
	@Native
	private MiroStat mirostat = MiroStat.DISABLED;
	// target entropy
	@Native
	private float mirostatTau = 5.00f;
	// learning rate
	@Native
	private float mirostatEta = 0.10f;
	@Native
	private boolean beamSearch = false;
	@Native
	private int nBeams = 2;
	// optional BNF-like grammar to constrain sampling
	@Nullable
	@Native
	private String grammar = null;
	// strings upon seeing which more user input is prompted
	@Nullable
	@Native
	private String[] stopStrings = null;
	@Nullable
	@Native
	private String[] promptTokenPenalties = null;
	@Native
	private Sampler[] samplers = null;
	@Native
	private int seed = 42;

	/**
	 * Set the amount of new tokens to predict
	 */
	public InferenceParameters setNPredict(int nPredict) {
		this.nPredict = nPredict;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setCachePrompt(boolean cachePrompt) {
		this.cachePrompt = cachePrompt;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setNKeep(int nKeep) {
		this.nKeep = nKeep;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setNDiscard(int nDiscard) {
		this.nDiscard = nDiscard;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setMinKeep(int minKeep) {
		this.minKeep = minKeep;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setNProbs(int nProbs) {
		this.nProbs = nProbs;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setLogitBias(@NotNull Map<Integer, Float> logitBias) {
		this.logitBias = Collections.unmodifiableMap(logitBias);
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setTopK(int topK) {
		this.topK = topK;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setTopP(float topP) {
		this.topP = topP;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setMinP(float minP) {
		this.minP = minP;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setTfsZ(float tfsZ) {
		this.tfsZ = tfsZ;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setTypicalP(float typicalP) {
		this.typicalP = typicalP;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setTemperature(float temperature) {
		this.temperature = temperature;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setDynamicTemperatureRange(float dynamicTemperatureRange) {
		this.dynamicTemperatureRange = dynamicTemperatureRange;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setDynamicTemperatureExponent(float dynamicTemperatureExponent) {
		this.dynamicTemperatureExponent = dynamicTemperatureExponent;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setRepeatPenalty(float repeatPenalty) {
		this.repeatPenalty = repeatPenalty;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setRepeatLastN(int repeatLastN) {
		this.repeatLastN = repeatLastN;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setFrequencyPenalty(float frequencyPenalty) {
		this.frequencyPenalty = frequencyPenalty;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setPresencePenalty(float presencePenalty) {
		this.presencePenalty = presencePenalty;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setPenalizeNl(boolean penalizeNl) {
		this.penalizeNl = penalizeNl;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setIgnoreEos(boolean ignoreEos) {
		this.ignoreEos = ignoreEos;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setMirostat(MiroStat mirostat) {
		this.mirostat = mirostat;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setMirostatTau(float mirostatTau) {
		this.mirostatTau = mirostatTau;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setMirostatEta(float mirostatEta) {
		this.mirostatEta = mirostatEta;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setBeamSearch(boolean beamSearch) {
		this.beamSearch = beamSearch;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setNBeams(int nBeams) {
		this.nBeams = nBeams;
		return this;
	}

	// default charset usage for Java backwards compatibility
	@SuppressWarnings("ImplicitDefaultCharsetUsage")
	public InferenceParameters setGrammar(@NotNull File file) throws IOException {
		StringBuilder grammarBuilder = new StringBuilder();
		try (BufferedReader br = new BufferedReader(new FileReader(file))) {
			String currentLine;
			while ((currentLine = br.readLine()) != null) {
				grammarBuilder.append(currentLine).append("\n");
			}
		}
		return setGrammar(grammarBuilder.toString());
	}

	/**
	 *
	 */
	public InferenceParameters setGrammar(@Nullable String grammar) {
		this.grammar = grammar;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setStopStrings(@NotNull String... stopStrings) {
		this.stopStrings = stopStrings;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setPromptTokenPenalties(@NotNull String... promptTokenPenalties) {
		this.promptTokenPenalties = promptTokenPenalties;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setSamplers(@NotNull Sampler... samplers) {
		this.samplers = samplers;
		return this;
	}

	/**
	 *
	 */
	public InferenceParameters setSeed(int seed) {
		this.seed = seed;
		return this;
	}

	public int getNPredict() {
		return nPredict;
	}

	public boolean isCachePrompt() {
		return cachePrompt;
	}

	public int getNKeep() {
		return nKeep;
	}

	public int getMinKeep() {
		return minKeep;
	}

	public int getNDiscard() {
		return nDiscard;
	}

	public int getNProbs() {
		return nProbs;
	}

	public @Nullable Map<Integer, Float> getLogitBias() {
		return logitBias;
	}

	public int getTopK() {
		return topK;
	}

	public float getTopP() {
		return topP;
	}

	public float getMinP() {
		return minP;
	}

	public float getTfsZ() {
		return tfsZ;
	}

	public float getTypicalP() {
		return typicalP;
	}

	public float getTemperature() {
		return temperature;
	}

	public float getDynamicTemperatureRange() {
		return dynamicTemperatureRange;
	}

	public float getDynamicTemperatureExponent() {
		return dynamicTemperatureExponent;
	}

	public float getRepeatPenalty() {
		return repeatPenalty;
	}

	public int getRepeatLastN() {
		return repeatLastN;
	}

	public float getFrequencyPenalty() {
		return frequencyPenalty;
	}

	public float getPresencePenalty() {
		return presencePenalty;
	}

	public boolean isPenalizeNl() {
		return penalizeNl;
	}

	public boolean isIgnoreEos() {
		return ignoreEos;
	}

	public MiroStat getMirostat() {
		return mirostat;
	}

	public float getMirostatTau() {
		return mirostatTau;
	}

	public float getMirostatEta() {
		return mirostatEta;
	}

	public boolean isBeamSearch() {
		return beamSearch;
	}

	public int getNBeams() {
		return nBeams;
	}

	public @Nullable String getGrammar() {
		return grammar;
	}

	public @Nullable String[] getStopStrings() {
		return stopStrings;
	}

	public @Nullable String[] getPromptTokenPenalties() {
		return promptTokenPenalties;
	}

	public @Nullable Sampler[] getSamplers() {
		return samplers;
	}

	public int getSeed() {
		return seed;
	}

}
