package de.kherud.llama;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import de.kherud.llama.args.MiroStat;
import de.kherud.llama.args.Sampler;

/**
 * Parameters used throughout inference of a {@link LlamaModel}, e.g., {@link LlamaModel#generate(InferenceParameters)}
 * and
 * {@link LlamaModel#complete(InferenceParameters)}.
 */
@SuppressWarnings("unused")
public final class InferenceParameters extends JsonParameters {

	private static final String PARAM_PROMPT = "prompt";
	private static final String PARAM_INPUT_PREFIX = "input_prefix";
	private static final String PARAM_INPUT_SUFFIX = "input_suffix";
	private static final String PARAM_CACHE_PROMPT = "cache_prompt";
	private static final String PARAM_N_PREDICT = "n_predict";
	private static final String PARAM_TOP_K = "top_k";
	private static final String PARAM_TOP_P = "top_p";
	private static final String PARAM_MIN_P = "min_p";
	private static final String PARAM_TFS_Z = "tfs_z";
	private static final String PARAM_TYPICAL_P = "typical_p";
	private static final String PARAM_TEMPERATURE = "temperature";
	private static final String PARAM_DYNATEMP_RANGE = "dynatemp_range";
	private static final String PARAM_DYNATEMP_EXPONENT = "dynatemp_exponent";
	private static final String PARAM_REPEAT_LAST_N = "repeat_last_n";
	private static final String PARAM_REPEAT_PENALTY = "repeat_penalty";
	private static final String PARAM_FREQUENCY_PENALTY = "frequency_penalty";
	private static final String PARAM_PRESENCE_PENALTY = "presence_penalty";
	private static final String PARAM_MIROSTAT = "mirostat";
	private static final String PARAM_MIROSTAT_TAU = "mirostat_tau";
	private static final String PARAM_MIROSTAT_ETA = "mirostat_eta";
	private static final String PARAM_PENALIZE_NL = "penalize_nl";
	private static final String PARAM_N_KEEP = "n_keep";
	private static final String PARAM_SEED = "seed";
	private static final String PARAM_N_PROBS = "n_probs";
	private static final String PARAM_MIN_KEEP = "min_keep";
	private static final String PARAM_GRAMMAR = "grammar";
	private static final String PARAM_PENALTY_PROMPT = "penalty_prompt";
	private static final String PARAM_IGNORE_EOS = "ignore_eos";
	private static final String PARAM_LOGIT_BIAS = "logit_bias";
	private static final String PARAM_STOP = "stop";
	private static final String PARAM_SAMPLERS = "samplers";
	private static final String PARAM_STREAM = "stream";
	private static final String PARAM_USE_CHAT_TEMPLATE = "use_chat_template";
	private static final String PARAM_USE_JINJA = "use_jinja";
	private static final String PARAM_MESSAGES = "messages";
	private static final String PARAM_TOOLS = "tools";
	private static final String PARAM_TOOL_CHOICE = "tool_choice";
	private static final String PARAM_PARALLEL_TOOL_CALLS = "parallel_tool_calls";
	private static final String PARAM_POST_SAMPLING_PROBS = "post_sampling_probs";
	private static final String PARAM_CHAT_FORMAT ="chat_format";
	private static final String PARAM_CHAT_TEMPLATE ="chat_template";
	private static final String PARAM_QUERY = "query";
	private static final String PARAM_DOCUMENTS = "documents";

	/**
	 * Set the prompt to start generation with (default: empty)
	 */
	public InferenceParameters setPrompt(String prompt) {
		parameters.put(PARAM_PROMPT, toJsonString(prompt));
		return this;
	}

	/**
	 * Set a prefix for infilling (default: empty)
	 */
	public InferenceParameters setInputPrefix(String inputPrefix) {
		parameters.put(PARAM_INPUT_PREFIX, toJsonString(inputPrefix));
		return this;
	}

	/**
	 * Set a suffix for infilling (default: empty)
	 */
	public InferenceParameters setInputSuffix(String inputSuffix) {
		parameters.put(PARAM_INPUT_SUFFIX, toJsonString(inputSuffix));
		return this;
	}

	/**
	 * Whether to remember the prompt to avoid reprocessing it
	 */
	public InferenceParameters setCachePrompt(boolean cachePrompt) {
		parameters.put(PARAM_CACHE_PROMPT, String.valueOf(cachePrompt));
		return this;
	}

	/**
	 * Set the number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
	 */
	public InferenceParameters setNPredict(int nPredict) {
		parameters.put(PARAM_N_PREDICT, String.valueOf(nPredict));
		return this;
	}

	/**
	 * Set top-k sampling (default: 40, 0 = disabled)
	 */
	public InferenceParameters setTopK(int topK) {
		parameters.put(PARAM_TOP_K, String.valueOf(topK));
		return this;
	}

	/**
	 * Set top-p sampling (default: 0.9, 1.0 = disabled)
	 */
	public InferenceParameters setTopP(float topP) {
		parameters.put(PARAM_TOP_P, String.valueOf(topP));
		return this;
	}

	/**
	 * Set min-p sampling (default: 0.1, 0.0 = disabled)
	 */
	public InferenceParameters setMinP(float minP) {
		parameters.put(PARAM_MIN_P, String.valueOf(minP));
		return this;
	}

	/**
	 * Set tail free sampling, parameter z (default: 1.0, 1.0 = disabled)
	 */
	public InferenceParameters setTfsZ(float tfsZ) {
		parameters.put(PARAM_TFS_Z, String.valueOf(tfsZ));
		return this;
	}

	/**
	 * Set locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)
	 */
	public InferenceParameters setTypicalP(float typicalP) {
		parameters.put(PARAM_TYPICAL_P, String.valueOf(typicalP));
		return this;
	}

	/**
	 * Set the temperature (default: 0.8)
	 */
	public InferenceParameters setTemperature(float temperature) {
		parameters.put(PARAM_TEMPERATURE, String.valueOf(temperature));
		return this;
	}

	/**
	 * Set the dynamic temperature range (default: 0.0, 0.0 = disabled)
	 */
	public InferenceParameters setDynamicTemperatureRange(float dynatempRange) {
		parameters.put(PARAM_DYNATEMP_RANGE, String.valueOf(dynatempRange));
		return this;
	}

	/**
	 * Set the dynamic temperature exponent (default: 1.0)
	 */
	public InferenceParameters setDynamicTemperatureExponent(float dynatempExponent) {
		parameters.put(PARAM_DYNATEMP_EXPONENT, String.valueOf(dynatempExponent));
		return this;
	}

	/**
	 * Set the last n tokens to consider for penalties (default: 64, 0 = disabled, -1 = ctx_size)
	 */
	public InferenceParameters setRepeatLastN(int repeatLastN) {
		parameters.put(PARAM_REPEAT_LAST_N, String.valueOf(repeatLastN));
		return this;
	}

	/**
	 * Set the penalty of repeated sequences of tokens (default: 1.0, 1.0 = disabled)
	 */
	public InferenceParameters setRepeatPenalty(float repeatPenalty) {
		parameters.put(PARAM_REPEAT_PENALTY, String.valueOf(repeatPenalty));
		return this;
	}

	/**
	 * Set the repetition alpha frequency penalty (default: 0.0, 0.0 = disabled)
	 */
	public InferenceParameters setFrequencyPenalty(float frequencyPenalty) {
		parameters.put(PARAM_FREQUENCY_PENALTY, String.valueOf(frequencyPenalty));
		return this;
	}

	/**
	 * Set the repetition alpha presence penalty (default: 0.0, 0.0 = disabled)
	 */
	public InferenceParameters setPresencePenalty(float presencePenalty) {
		parameters.put(PARAM_PRESENCE_PENALTY, String.valueOf(presencePenalty));
		return this;
	}

	/**
	 * Set MiroStat sampling strategies.
	 */
	public InferenceParameters setMiroStat(MiroStat mirostat) {
		parameters.put(PARAM_MIROSTAT, String.valueOf(mirostat.ordinal()));
		return this;
	}

	/**
	 * Set the MiroStat target entropy, parameter tau (default: 5.0)
	 */
	public InferenceParameters setMiroStatTau(float mirostatTau) {
		parameters.put(PARAM_MIROSTAT_TAU, String.valueOf(mirostatTau));
		return this;
	}

	/**
	 * Set the MiroStat learning rate, parameter eta (default: 0.1)
	 */
	public InferenceParameters setMiroStatEta(float mirostatEta) {
		parameters.put(PARAM_MIROSTAT_ETA, String.valueOf(mirostatEta));
		return this;
	}

	/**
	 * Whether to penalize newline tokens
	 */
	public InferenceParameters setPenalizeNl(boolean penalizeNl) {
		parameters.put(PARAM_PENALIZE_NL, String.valueOf(penalizeNl));
		return this;
	}

	/**
	 * Set the number of tokens to keep from the initial prompt (default: 0, -1 = all)
	 */
	public InferenceParameters setNKeep(int nKeep) {
		parameters.put(PARAM_N_KEEP, String.valueOf(nKeep));
		return this;
	}

	/**
	 * Set the RNG seed (default: -1, use random seed for &lt; 0)
	 */
	public InferenceParameters setSeed(int seed) {
		parameters.put(PARAM_SEED, String.valueOf(seed));
		return this;
	}

	/**
	 * Set the amount top tokens probabilities to output if greater than 0.
	 */
	public InferenceParameters setNProbs(int nProbs) {
		parameters.put(PARAM_N_PROBS, String.valueOf(nProbs));
		return this;
	}

	/**
	 * Set the amount of tokens the samplers should return at least (0 = disabled)
	 */
	public InferenceParameters setMinKeep(int minKeep) {
		parameters.put(PARAM_MIN_KEEP, String.valueOf(minKeep));
		return this;
	}

	/**
	 * Set BNF-like grammar to constrain generations (see samples in grammars/ dir)
	 */
	public InferenceParameters setGrammar(String grammar) {
		parameters.put(PARAM_GRAMMAR, toJsonString(grammar));
		return this;
	}

	/**
	 * Override which part of the prompt is penalized for repetition.
	 * E.g. if original prompt is "Alice: Hello!" and penaltyPrompt is "Hello!", only the latter will be penalized if
	 * repeated. See <a href="https://github.com/ggerganov/llama.cpp/pull/3727">pull request 3727</a> for more details.
	 */
	public InferenceParameters setPenaltyPrompt(String penaltyPrompt) {
		parameters.put(PARAM_PENALTY_PROMPT, toJsonString(penaltyPrompt));
		return this;
	}

	/**
	 * Override which tokens to penalize for repetition.
	 * E.g. if original prompt is "Alice: Hello!" and penaltyPrompt corresponds to the token ids of "Hello!", only the
	 * latter will be penalized if repeated.
	 * See <a href="https://github.com/ggerganov/llama.cpp/pull/3727">pull request 3727</a> for more details.
	 */
	public InferenceParameters setPenaltyPrompt(int[] tokens) {
		if (tokens.length > 0) {
			StringBuilder builder = new StringBuilder();
			builder.append("[");
			for (int i = 0; i < tokens.length; i++) {
				builder.append(tokens[i]);
				if (i < tokens.length - 1) {
					builder.append(", ");
				}
			}
			builder.append("]");
			parameters.put(PARAM_PENALTY_PROMPT, builder.toString());
		}
		return this;
	}

	/**
	 * Set whether to ignore end of stream token and continue generating (implies --logit-bias 2-inf)
	 */
	public InferenceParameters setIgnoreEos(boolean ignoreEos) {
		parameters.put(PARAM_IGNORE_EOS, String.valueOf(ignoreEos));
		return this;
	}

	/**
	 * Modify the likelihood of tokens appearing in the completion by their id. E.g., <code>Map.of(15043, 1f)</code>
	 * to increase the  likelihood of token ' Hello', or a negative value to decrease it.
	 * Note, this method overrides any previous calls to
	 * <ul>
	 *     <li>{@link #setTokenBias(Map)}</li>
	 *     <li>{@link #disableTokens(Collection)}</li>
	 *     <li>{@link #disableTokenIds(Collection)}}</li>
	 * </ul>
	 */
	public InferenceParameters setTokenIdBias(Map<Integer, Float> logitBias) {
		if (!logitBias.isEmpty()) {
			StringBuilder builder = new StringBuilder();
			builder.append("[");
			int i = 0;
			for (Map.Entry<Integer, Float> entry : logitBias.entrySet()) {
				Integer key = entry.getKey();
				Float value = entry.getValue();
				builder.append("[")
						.append(key)
						.append(", ")
						.append(value)
						.append("]");
				if (i++ < logitBias.size() - 1) {
					builder.append(", ");
				}
			}
			builder.append("]");
			parameters.put(PARAM_LOGIT_BIAS, builder.toString());
		}
		return this;
	}

	/**
	 * Set tokens to disable, this corresponds to {@link #setTokenIdBias(Map)} with a value of
	 * {@link Float#NEGATIVE_INFINITY}.
	 * Note, this method overrides any previous calls to
	 * <ul>
	 *     <li>{@link #setTokenIdBias(Map)}</li>
	 *     <li>{@link #setTokenBias(Map)}</li>
	 *     <li>{@link #disableTokens(Collection)}</li>
	 * </ul>
	 */
	public InferenceParameters disableTokenIds(Collection<Integer> tokenIds) {
		if (!tokenIds.isEmpty()) {
			StringBuilder builder = new StringBuilder();
			builder.append("[");
			int i = 0;
			for (Integer token : tokenIds) {
				builder.append("[")
						.append(token)
						.append(", ")
						.append(false)
						.append("]");
				if (i++ < tokenIds.size() - 1) {
					builder.append(", ");
				}
			}
			builder.append("]");
			parameters.put(PARAM_LOGIT_BIAS, builder.toString());
		}
		return this;
	}

	/**
	 * Modify the likelihood of tokens appearing in the completion by their id. E.g., <code>Map.of(" Hello", 1f)</code>
	 * to increase the  likelihood of token id 15043, or a negative value to decrease it.
	 * Note, this method overrides any previous calls to
	 * <ul>
	 *     <li>{@link #setTokenIdBias(Map)}</li>
	 *     <li>{@link #disableTokens(Collection)}</li>
	 *     <li>{@link #disableTokenIds(Collection)}}</li>
	 * </ul>
	 */
	public InferenceParameters setTokenBias(Map<String, Float> logitBias) {
		if (!logitBias.isEmpty()) {
			StringBuilder builder = new StringBuilder();
			builder.append("[");
			int i = 0;
			for (Map.Entry<String, Float> entry : logitBias.entrySet()) {
				String key = entry.getKey();
				Float value = entry.getValue();
				builder.append("[")
						.append(toJsonString(key))
						.append(", ")
						.append(value)
						.append("]");
				if (i++ < logitBias.size() - 1) {
					builder.append(", ");
				}
			}
			builder.append("]");
			parameters.put(PARAM_LOGIT_BIAS, builder.toString());
		}
		return this;
	}

	/**
	 * Set tokens to disable, this corresponds to {@link #setTokenBias(Map)} with a value of
	 * {@link Float#NEGATIVE_INFINITY}.
	 * Note, this method overrides any previous calls to
	 * <ul>
	 *     <li>{@link #setTokenBias(Map)}</li>
	 *     <li>{@link #setTokenIdBias(Map)}</li>
	 *     <li>{@link #disableTokenIds(Collection)}</li>
	 * </ul>
	 */
	public InferenceParameters disableTokens(Collection<String> tokens) {
		if (!tokens.isEmpty()) {
			StringBuilder builder = new StringBuilder();
			builder.append("[");
			int i = 0;
			for (String token : tokens) {
				builder.append("[")
						.append(toJsonString(token))
						.append(", ")
						.append(false)
						.append("]");
				if (i++ < tokens.size() - 1) {
					builder.append(", ");
				}
			}
			builder.append("]");
			parameters.put(PARAM_LOGIT_BIAS, builder.toString());
		}
		return this;
	}

	/**
	 * Set strings upon seeing which token generation is stopped
	 */
	public InferenceParameters setStopStrings(String... stopStrings) {
		if (stopStrings.length > 0) {
			StringBuilder builder = new StringBuilder();
			builder.append("[");
			for (int i = 0; i < stopStrings.length; i++) {
				builder.append(toJsonString(stopStrings[i]));
				if (i < stopStrings.length - 1) {
					builder.append(", ");
				}
			}
			builder.append("]");
			parameters.put(PARAM_STOP, builder.toString());
		}
		return this;
	}

	/**
	 * Set which samplers to use for token generation in the given order
	 */
	public InferenceParameters setSamplers(Sampler... samplers) {
		if (samplers.length > 0) {
			StringBuilder builder = new StringBuilder();
			builder.append("[");
			for (int i = 0; i < samplers.length; i++) {
				switch (samplers[i]) {
					case TOP_K:
						builder.append("\"top_k\"");
						break;
					case TOP_P:
						builder.append("\"top_p\"");
						break;
					case MIN_P:
						builder.append("\"min_p\"");
						break;
					case TEMPERATURE:
						builder.append("\"temperature\"");
						break;
				}
				if (i < samplers.length - 1) {
					builder.append(", ");
				}
			}
			builder.append("]");
			parameters.put(PARAM_SAMPLERS, builder.toString());
		}
		return this;
	}

	/**
	 * Set whether generate should apply a chat template (default: false)
	 */
	public InferenceParameters setUseChatTemplate(boolean useChatTemplate) {
		parameters.put(PARAM_USE_JINJA, String.valueOf(useChatTemplate));
		return this;
	}
	
	/**
     * Set the messages for chat-based inference.
     * - Allows **only one** system message.
     * - Allows **one or more** user/assistant messages.
     */
    public InferenceParameters setMessages(String systemMessage, List<Pair<String, String>> messages) {
		StringBuilder messagesBuilder = new StringBuilder();
		messagesBuilder.append("[");

        // Add system message (if provided)
        if (systemMessage != null && !systemMessage.isEmpty()) {
			messagesBuilder.append("{\"role\": \"system\", \"content\": ")
					.append(toJsonString(systemMessage))
					.append("}");
			if (!messages.isEmpty()) {
				messagesBuilder.append(", ");
			}
        }

        // Add user/assistant messages
        for (int i = 0; i < messages.size(); i++) {
            Pair<String, String> message = messages.get(i);
            String role = message.getKey();
            String content = message.getValue();

            if (!role.equals("user") && !role.equals("assistant")) {
                throw new IllegalArgumentException("Invalid role: " + role + ". Role must be 'user' or 'assistant'.");
            }

			messagesBuilder.append("{\"role\":")
					.append(toJsonString(role))
					.append(", \"content\": ")
					.append(toJsonString(content))
					.append("}");

			if (i < messages.size() - 1) {
				messagesBuilder.append(", ");
			}
        }

		messagesBuilder.append("]");

        // Convert ArrayNode to a JSON string and store it in parameters
        parameters.put(PARAM_MESSAGES, messagesBuilder.toString());
        return this;
    }
    
    

	InferenceParameters setStream(boolean stream) {
		parameters.put(PARAM_STREAM, String.valueOf(stream));
		return this;
	}
	
	/**
	 * Set Tools
	 */
	public InferenceParameters setTools(String... tools) {
		StringBuilder toolBuilder = new StringBuilder();
		
		for (String tool:tools) {
			if (toolBuilder.length() > 0) {
				toolBuilder.append(",");
			}
			toolBuilder.append(tool);
			
		}
		
 		parameters.put(PARAM_TOOLS, "[" + toolBuilder.toString() +"]");
 		parameters.put(PARAM_TOOL_CHOICE, toJsonString("required"));
// 		parameters.put(PARAM_PARALLEL_TOOL_CALLS,String.valueOf(false));
		return this;
	}
	
	public InferenceParameters setPostSamplingProbs(boolean postSamplingProbs) {
		parameters.put(PARAM_POST_SAMPLING_PROBS, String.valueOf(postSamplingProbs));
		return this;
	}

	public InferenceParameters setChatTemplate(String chatTemplate) {
		parameters.put(PARAM_CHAT_TEMPLATE, toJsonString(chatTemplate));
		return this;
	}

	public InferenceParameters setQuery(String query) {
		parameters.put(PARAM_QUERY, toJsonString(query));
		return this;
		
	}

	public InferenceParameters setDocuments(String[] documents) {
		
		if (documents.length > 0) {
			StringBuilder builder = new StringBuilder();
			builder.append("[");
			for (int i = 0; i < documents.length; i++) {
				builder.append(toJsonString(documents[i]));
				if (i < documents.length - 1) {
					builder.append(", ");
				}
			}
			builder.append("]");
			parameters.put(PARAM_DOCUMENTS, builder.toString());
		}
		
		return this;
	}

}
