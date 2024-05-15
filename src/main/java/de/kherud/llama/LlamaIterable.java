package de.kherud.llama;

import org.jetbrains.annotations.NotNull;

/**
 * An iterable used by {@link LlamaModel#generate(InferenceParameters)} that specifically returns a {@link LlamaIterator}.
 */
@FunctionalInterface
public interface LlamaIterable extends Iterable<LlamaOutput> {

    @NotNull
    @Override
    LlamaIterator iterator();

}
