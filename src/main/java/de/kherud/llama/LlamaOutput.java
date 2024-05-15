package de.kherud.llama;

import org.jetbrains.annotations.NotNull;

import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * An output of the LLM providing access to the generated text and the associated probabilities. You have to configure
 * {@link InferenceParameters#setNProbs(int)} in order for probabilities to be returned.
 */
public final class LlamaOutput {

    /**
     * The last bit of generated text that is representable as text (i.e., cannot be individual utf-8 multibyte code
     * points).
     */
    @NotNull
    public final String text;

    /**
     * Note, that you have to configure {@link InferenceParameters#setNProbs(int)} in order for probabilities to be returned.
     */
    @NotNull
    public final Map<String, Float> probabilities;

    final boolean stop;

    LlamaOutput(byte[] generated, @NotNull Map<String, Float> probabilities, boolean stop) {
        this.text = new String(generated, StandardCharsets.UTF_8);
        this.probabilities = probabilities;
        this.stop = stop;
    }

    @Override
    public String toString() {
        return text;
    }
}
