package de.kherud.llama;

import com.sun.jna.Pointer;
import de.kherud.llama.foreign.LlamaLibrary;
import de.kherud.llama.foreign.NativeSize;
import de.kherud.llama.foreign.llama_timings;
import de.kherud.llama.foreign.llama_token_data;
import de.kherud.llama.foreign.llama_token_data_array;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.function.BiConsumer;

/**
 * This class is a wrapper around the llama.cpp functionality.
 * Upon being created, it natively allocates memory for the model context.
 * Thus, this class is an {@link AutoCloseable}, in order to de-allocate the memory when it is no longer being needed.
 * <p>
 * This class is NOT stateless, it internally manages the state of a <i>single</i> conversation.
 * If this state isn't needed or wanted, call {@link #reset()}.
 * <p>
 * The main functionality of this class is:
 * <ul>
 *     <li>Streaming answers (and probabilities) via {@link #generate(String)}</li>
 *     <li>Creating whole responses to prompts via {@link #complete(String)}</li>
 *     <li>Creating embeddings via {@link #getEmbedding(String)} (make sure to configure {@link Parameters.Builder#setEmbedding(boolean)}</li>
 *     <li>Accessing the tokenizer via {@link #encode(String)} and {@link #decode(int[])}</li>
 * </ul>
 */
public class LlamaModel implements AutoCloseable {

    static {
        LlamaLibrary.llama_backend_init((byte) 0);
    }
    private static BiConsumer<LogLevel, String> logCallback;

    private final Parameters params;
    private final LlamaLibrary.llama_model model;
    private final LlamaLibrary.llama_context ctx;

    // cache some things for performance
    private final Pointer logitsPointer; // pointer to retrieve the logits of the llm
    private final SliceableIntBuffer contextBuffer; // used to hold all tokens of a conversation
    private final SliceableIntBuffer tokenBuffer; // used for tokenization
    private SliceableByteBuffer tokenPieceBuffer; // used to decode tokens to string (might be resized if out of capacity)
    private final llama_token_data.ByReference[] candidateData; // candidates used for sampling
    private final llama_token_data_array candidates; // array holding the candidates
    private final int nVocab;
    private final int tokenBos;
    private final int tokenEos;
    private final int tokenNl;
    private int nPast = 0; // how many evaluated tokens are currently in the context buffer
    private int nContext = 0; // how many tokens currently are in the context buffer
    private int nBuffered = 0;

    /**
     * Load a <b>gguf</b> llama.cpp model from a given file path with default {@link Parameters}.
     *
     * @param filePath a file path pointing to the model
     */
    public LlamaModel(String filePath) {
        this(filePath, new Parameters.Builder().build());
    }

    /**
     * Load a <b>gguf</b> llama.cpp model from a given file path with custom {@link Parameters}.
     *
     * @param filePath a file path pointing to the model
     * @param params the set of previously configured options
     */
    public LlamaModel(String filePath, Parameters params) {
        // load the model
        this.params = params;
        model = LlamaLibrary.llama_load_model_from_file(filePath, params.ctx);
        if (model == null) {
            throw new RuntimeException("error: unable to load model");
        }
        ctx = LlamaLibrary.llama_new_context_with_model(model, params.ctx);

        // load lora adapter if configured
        if (params.loraAdapter != null
                && params.loraBase != null
                && LlamaLibrary.llama_model_apply_lora_from_file(model, params.loraAdapter, params.loraBase, params.nThreads) != 0) {
            throw new RuntimeException("error: unable to apply lora");
        }

        // setup some cached variables used throughout lifecycle
        contextBuffer = new SliceableIntBuffer(IntBuffer.allocate(params.ctx.n_ctx));
        tokenBuffer = new SliceableIntBuffer(IntBuffer.allocate(params.ctx.n_ctx));
        tokenPieceBuffer = new SliceableByteBuffer(ByteBuffer.allocate(64));
        nVocab = getVocabularySize();

        logitsPointer = LlamaLibrary.llama_get_logits(ctx).getPointer();
        candidateData = (llama_token_data.ByReference[]) new llama_token_data.ByReference().toArray(nVocab);
        candidates = new llama_token_data_array();

        tokenBos = LlamaLibrary.llama_token_bos(ctx);
        tokenEos = LlamaLibrary.llama_token_eos(ctx);
        tokenNl = LlamaLibrary.llama_token_nl(ctx);
    }

    /**
     * Generate and stream outputs. Unless {@link #reset()} is called, the previous conversation is used as context.
     * Note, that the prompt isn't preprocessed in any way, nothing like "User: ", "###Instruction", etc. is added.
     *
     * @param prompt the LLM prompt
     * @return iterable LLM outputs
     */
    public Iterable<Output> generate(String prompt) {
        return new Iterable<>() {
            @NotNull
            @Override
            public Iterator<Output> iterator() {
                return new LlamaIterator(prompt);
            }
        };
    }

    /**
     * Generate and return a whole answer. Unless {@link #reset()} is called, the previous conversation is used as context.
     * Note, that the prompt isn't preprocessed in any way, nothing like "User: ", "###Instruction", etc. is added.
     *
     * @param prompt the LLM prompt
     * @return an LLM response
     */
    public String complete(String prompt) {
        StringBuilder builder = new StringBuilder();
        Iterator<Output> iterator = new LlamaIterator(prompt);
        while (iterator.hasNext()) {
            Output output = iterator.next();
            builder.append(output);
        }
        return builder.toString();
    }

    /**
     * Get the embedding of a string. Unless {@link #reset()} is called, the previous conversation is used as context.
     * Note, that the prompt isn't preprocessed in any way, nothing like "User: ", "###Instruction", etc. is added.
     *
     * @param prompt the string to embed
     * @return an embedding float array the size of {@link #getEmbeddingSize()}
     * @throws IllegalStateException if embedding mode was not activated (see {@link Parameters.Builder#setEmbedding(boolean)})
     */
    public float[] getEmbedding(String prompt) {
        if (params.ctx.embedding == 0) {
            throw new IllegalStateException("embedding mode not activated (see parameters)");
        }
        SliceableIntBuffer tokens = tokenize(prompt, false);
        addContext(tokens);
        evaluate();
        return LlamaLibrary.llama_get_embeddings(ctx).getPointer().getFloatArray(0, getEmbeddingSize());
    }

    /**
     * Resets the state of the LLM, its timings, and discards any previous conversation.
     */
    public void reset() {
        LlamaLibrary.llama_reset_timings(ctx);
        this.contextBuffer.clear();
        this.nContext = 0;
        this.nPast = 0;
    }

    /**
     * Tokenize a prompt given the native tokenizer
     *
     * @param prompt the prompt to tokenize
     * @return an array of integers each representing a token id (see {@link #getVocabularySize()})
     */
    public int[] encode(String prompt) {
        SliceableIntBuffer buffer = tokenize(prompt, false);
        int[] tokens = new int[buffer.capacity()];
        System.arraycopy(buffer.delegate.array(), 0, tokens, 0, buffer.capacity());
        return tokens;
    }

    /**
     * Convert an array of token ids to its string representation
     *
     * @param tokens an array of tokens conforming to {@link #getVocabularySize()}
     * @return the token ids decoded to a string
     */
    public String decode(int[] tokens) {
        StringBuilder builder = new StringBuilder();
        for (int token : tokens) {
            String decoded = decodeToken(token);
            builder.append(decoded);
        }
        return builder.toString();
    }

    /**
     * Sets a callback for both Java and C++ log messages.
     * Can be set to {@code null} to disable logging.
     * Note, that in that case C++ output will appear in stdout/err, however.
     * To completely silence output provide a callback like:
     * <pre>
     * LlamaModel.setLogger((level, message) -> {});
     * </pre>
     *
     * @param logCallback a method to call for log messages
     */
    public static void setLogger(@Nullable BiConsumer<LogLevel, String> logCallback) {
        LlamaModel.logCallback = logCallback;
        // We currently do not allow to pass any user data to `llama_log_set`, since the JVM might move
        // the object around in the memory, thus invalidating any pointers.
        // Maybe this could be circumvented by allowing something like <T extends Structure> to be passed
        // as user data. However, the use cases for this are minuscule I think.
        if (logCallback == null) {
            LlamaLibrary.llama_log_set(null, null);
        } else {
            LlamaLibrary.llama_log_set((code, text, user_data) -> {
                LogLevel level = LogLevel.fromCode(code);
                logCallback.accept(level, text);
            }, null);
        }
    }

    /**
     * Returns the context size of the loaded model, e.g., how many tokens the LLM can process at most in one request.
     * E.g., for llama 1 this size is 2048.
     *
     * @return the context size of the loaded model
     */
    public int getContextSize() {
        return LlamaLibrary.llama_n_ctx(ctx);
    }

    /**
     * Returns the hidden dimensionality of the loaded model, which corresponds to the size of {@link #getEmbedding(String)}.
     * This size typically depends on the amount of parameters in the model, e.g., for llama 2 13b this size is 5120.
     *
     * @return the amount of embedding dimensions
     */
    public int getEmbeddingSize() {
        return LlamaLibrary.llama_n_embd(ctx);
    }

    /**
     * Returns the total amount of tokens in the vocabulary.
     *
     * @return the vocabulary size
     */
    public int getVocabularySize() {
        return LlamaLibrary.llama_n_vocab(ctx);
    }

    /**
     * Returns the tokenization method / vocabulary type used by this model.
     *
     * @return the vocabulary type
     */
    public VocabularyType getVocabularyType() {
        int code = LlamaLibrary.llama_vocab_type(ctx);
        return VocabularyType.fromCode(code);
    }

    /**
     * Returns the memory size of this model in bytes
     *
     * @return amount of bytes allocated
     */
    public long getMemorySize() {
        return LlamaLibrary.llama_model_size(model);
    }

    /**
     * Returns the amount of parameters this model has
     *
     * @return amount of parameters
     */
    public long getAmountParameters() {
        return LlamaLibrary.llama_model_n_params(model);
    }

    /**
     * Get performance information about this model
     *
     * @return the inference timings
     */
    public llama_timings getTimings() {
        return LlamaLibrary.llama_get_timings(ctx);
    }

    /**
     * Deallocates the native memory used by this model
     */
    @Override
    public void close() {
        LlamaLibrary.llama_free_model(model);
        LlamaLibrary.llama_free(ctx);
    }

    @Override
    public String toString() {
        byte[] buffer = new byte[512];
        int size = LlamaLibrary.llama_model_desc(model, buffer, new NativeSize(buffer.length));
        return new String(buffer, 0, Math.min(size, buffer.length), StandardCharsets.UTF_8);
    }


    /**
     * Internally tokenizes a prompt and returns its tokens without any padding.
     * At most {@link #getContextSize()} tokens can be tokenized. Makes use of {@link #tokenBuffer}.
     *
     * @param prompt the prompt to tokenize
     * @return an IntBuffer containing the tokenized prompt without any padding
     * @throws RuntimeException if tokenization fails
     */
    private SliceableIntBuffer tokenize(String prompt, boolean addBos) {
        int nTokens = LlamaLibrary.llama_tokenize(ctx, prompt, prompt.length(), tokenBuffer.delegate, params.ctx.n_ctx, addBos ? (byte) 1 : 0);
        if (nTokens < 0) {
            throw new RuntimeException("tokenization failed due to unknown reasons");
        }
        return tokenBuffer.slice(0, nTokens);
    }

    private void evaluate() {
        while (nPast < nContext) {
            int nEval = nContext - nPast;
            if (nEval > params.ctx.n_batch) {
                nEval = params.ctx.n_batch;
            }
            if (LlamaLibrary.llama_eval(ctx, contextBuffer.slice(nPast, nEval).delegate, nEval, nPast, params.nThreads) != 0) {
                String msg = String.format("evaluation failed (%d to evaluate, %d past, %d threads)", nEval, nPast, params.nThreads);
                log(LogLevel.ERROR, msg);
                throw new RuntimeException("token evaluation failed");
            }
            nPast += nEval;
        }
    }

    private Output sample() {
        float[] logits = logitsPointer.getFloatArray(0, nVocab);
        float nlLogit = logits[tokenNl];
        params.logitBias.forEach((i, bias) -> logits[i] += bias);
        // I'm not sure why anything but `setLogit` has to be called here for `candidateData` and `candidates` again.
        // Otherwise, the results are garbage, however. Maybe the JVM/JIT is moving stuff around.
        for (int i = 0; i < nVocab; i++) {
            candidateData[i].setId(i);
            candidateData[i].setLogit(logits[i]);
//			candidateData[i].setP(0f);
        }
        candidates.setData(candidateData[0]);
        candidates.setSize(new NativeSize(nVocab));
        candidates.setSorted((byte) 0);

        samplePenalty();

        if (!params.penalizeNl) {
            candidateData[tokenNl].setLogit(nlLogit);
        }

        if (params.grammar != null) {
            LlamaLibrary.llama_sample_grammar(ctx, candidates, params.grammar.foreign);
        }

        int token;
        if (params.temperature == 0) {
            token = sampleGreedy();
        } else if (params.mirostat == Parameters.MiroStat.V1) {
            token = sampleMirostatV1();
        } else if (params.mirostat == Parameters.MiroStat.V2) {
            token = sampleMirostatV2();
        } else {
            token = sampleTopK();
        }

        if (params.grammar != null) {
            LlamaLibrary.llama_grammar_accept_token(ctx, params.grammar.foreign, token);
        }

        return new Output(token, candidateData[token].p);
    }

    private void samplePenalty() {
        int repeat_last_n = params.repeatLastN < 0 ? params.ctx.n_ctx : params.repeatLastN;
        int last_n_repeat = Math.min(Math.min(nContext, repeat_last_n), params.ctx.n_ctx);
        NativeSize nTokens = new NativeSize(last_n_repeat);
        SliceableIntBuffer lastTokens = tokenBuffer.slice(nContext - last_n_repeat, last_n_repeat);
        LlamaLibrary.llama_sample_repetition_penalty(
                ctx,
                candidates,
                lastTokens.delegate,
                nTokens,
                params.repeatPenalty
        );
        LlamaLibrary.llama_sample_frequency_and_presence_penalties(
                ctx,
                candidates,
                lastTokens.delegate,
                nTokens,
                params.frequencyPenalty,
                params.presencePenalty
        );
    }

    private int sampleGreedy() {
        int token = LlamaLibrary.llama_sample_token_greedy(ctx, candidates);
        if (params.nProbs > 0) {
            LlamaLibrary.llama_sample_softmax(ctx, candidates);
        }
        return token;
    }

    private int sampleMirostatV1() {
        LlamaLibrary.llama_sample_temperature(ctx, candidates, params.temperature);
        return LlamaLibrary.llama_sample_token_mirostat(
                ctx,
                candidates,
                params.mirostatTau,
                params.mirostatEta,
                params.mirostatM,
                params.mirostatMu
        );
    }

    private int sampleMirostatV2() {
        LlamaLibrary.llama_sample_temperature(ctx, candidates, params.temperature);
        return LlamaLibrary.llama_sample_token_mirostat_v2(
                ctx,
                candidates,
                params.mirostatTau,
                params.mirostatEta,
                params.mirostatMu
        );
    }

    private int sampleTopK() {
        NativeSize minKeep = new NativeSize(Math.max(1, params.nProbs));
        LlamaLibrary.llama_sample_top_k(
                ctx,
                candidates,
                params.topK <= 0 ? nVocab : params.topK,
                minKeep
        );
        LlamaLibrary.llama_sample_tail_free(
                ctx,
                candidates,
                params.tfsZ,
                minKeep
        );
        LlamaLibrary.llama_sample_typical(
                ctx,
                candidates,
                params.typicalP,
                minKeep
        );
        LlamaLibrary.llama_sample_top_p(ctx,
                candidates,
                params.topP,
                minKeep
        );
        LlamaLibrary.llama_sample_temperature(
                ctx,
                candidates,
                params.temperature
        );
        return LlamaLibrary.llama_sample_token(ctx, candidates);
    }

    private void addContext(SliceableIntBuffer tokens) {
        truncateContext(tokens.capacity());
        System.arraycopy(tokens.delegate.array(), 0, contextBuffer.delegate.array(), nContext, tokens.capacity());
        nContext += tokens.capacity();
    }

    private void truncateContext(int nAdd) {
        if (nContext + nAdd > params.ctx.n_ctx) {
            int nCtxKeep = params.ctx.n_ctx / 2 - nAdd;
            String msg = "truncating context from " + nContext + " to " + nCtxKeep + " tokens (+" + nAdd + " to add)";
            log(LogLevel.INFO, msg);
            System.arraycopy(contextBuffer.delegate.array(), nContext - nCtxKeep, contextBuffer.delegate.array(), 0, nCtxKeep);
            nPast = 0;
            nContext = nCtxKeep;
        }
    }

    private String decodeToken(int token) {
        int bufferSize = tokenPieceBuffer.capacity() - nBuffered;
        SliceableByteBuffer slice = tokenPieceBuffer.slice(nBuffered, bufferSize);
        int pieceSize = LlamaLibrary.llama_token_to_piece(ctx, token, slice.delegate, bufferSize);

        // while the buffer is too small for the decoded tokens, resize the buffer and retry de-tokenization
        while (pieceSize < 0 || pieceSize > bufferSize) {
            // create the buffer double the size
            ByteBuffer newBuffer = ByteBuffer.allocate(tokenPieceBuffer.capacity() * 2);
            // copy the old content
            for (int i = 0; i < nBuffered; i++) {
                 newBuffer.put(i, tokenPieceBuffer.get(i));
            }
            tokenPieceBuffer = new SliceableByteBuffer(newBuffer);
            bufferSize = tokenPieceBuffer.capacity() - nBuffered;
            slice = tokenPieceBuffer.slice(nBuffered, bufferSize);
            pieceSize = LlamaLibrary.llama_token_to_piece(ctx, token, slice.delegate, bufferSize);
        }

        // after successful de-tokenization, check if the piece can be directly returned or is an utf-8 codepoint a
        // needs to be buffered
        if ((tokenPieceBuffer.get(nBuffered) & 0x80) == 0) {
            int nTotal = nBuffered + Math.min(pieceSize, tokenPieceBuffer.capacity());
            nBuffered = 0;
            return new String(tokenPieceBuffer.delegate.array(), 0, nTotal, StandardCharsets.UTF_8);
        } else {
            nBuffered += pieceSize;
            return "";
        }
    }

    private void log(LogLevel level, String message) {
        if (logCallback != null) {
            logCallback.accept(level, message);
        }
    }

    private class LlamaIterator implements Iterator<Output> {

        private final StringBuilder builder = new StringBuilder();
        private boolean hasNext = true;
        private int nRemain = params.nPredict;

        public LlamaIterator(String prompt) {
            setup(prompt);
        }

        @Override
        public boolean hasNext() {
            return hasNext;
        }

        @Override
        public Output next() {
            if (!hasNext) {
                throw new NoSuchElementException();
            }
            evaluate();
            Output result = sample();
            builder.append(result.text);
            truncateContext(1);
            contextBuffer.put(nContext, result.token);
            nContext++;
            nRemain--;
            checkHasNext();
            return result;
        }

        private void checkHasNext() {
            hasNext = (params.nPredict < 0 || nRemain > 0) // if there are infinite or more tokens to predict
                    && contextBuffer.get(nContext - 1) != tokenEos // if the last token was not a stop token
                    && !isAntiPrompt(); // if the last tokens don't form an anti-prompt
        }

        private void setup(String prompt) {
            // add a space in front of the first character to match OG llama tokenizer behavior (taken from main.cpp)
            if (nContext == 0 && !prompt.startsWith(" ")) {
                prompt = " " + prompt;
            }
            SliceableIntBuffer tokens = tokenize(prompt, true);
            addContext(tokens);
        }

        private boolean isAntiPrompt() {
            for (String antiPrompt : params.antiprompt) {
                if (builder.lastIndexOf(antiPrompt) > 0) {
                    return true;
                }
            }
            return false;
        }
    }

    /**
     * An output generated by the LLM. Provides access to its probability.
     */
    public final class Output {
        public final int token;
        public final float probability;
        public final String text;


        private Output(int token, float probability) {
            this.token = token;
            this.probability = probability;
            // directly convert it to text to do it only once
            this.text = decodeToken(token);
        }

        @Override
        public String toString() {
            return text;
        }

        /**
         * Returns the type of this token.
         *
         * @return the token type
         */
        public TokenType getType() {
            int code = LlamaLibrary.llama_token_get_type(ctx, token);
            return TokenType.fromCode(code);
        }

        /**
         * Returns the score of this token
         *
         * @return the float score
         */
        public float getScore() {
            return LlamaLibrary.llama_token_get_score(ctx, token);
        }
    }
}
