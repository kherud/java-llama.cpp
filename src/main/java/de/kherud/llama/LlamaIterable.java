package de.kherud.llama;

/**
 * An iterable used by {@link LlamaModel#generate(InferenceParameters)} that specifically returns a {@link LlamaIterator}.
 */
@FunctionalInterface
public interface LlamaIterable extends Iterable<LlamaOutput> {

    @Override
    LlamaIterator iterator();

}
