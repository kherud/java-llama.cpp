package de.kherud.llama;

import java.lang.annotation.Native;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * This iterator is used by {@link LlamaModel#generate(InferenceParameters)}. In addition to implementing {@link Iterator},
 * it allows to cancel ongoing inference (see {@link #cancel()}).
 */
public final class LlamaIterator implements Iterator<LlamaOutput> {

    private final LlamaModel model;
    private final int taskId;

    @Native
    @SuppressWarnings("FieldMayBeFinal")
    private boolean hasNext = true;

    LlamaIterator(LlamaModel model, InferenceParameters parameters) {
        this.model = model;
        parameters.setStream(true);
        taskId = model.requestCompletion(parameters.toString());
    }

    @Override
    public boolean hasNext() {
        return hasNext;
    }

    @Override
    public LlamaOutput next() {
        if (!hasNext) {
            throw new NoSuchElementException();
        }
        LlamaOutput output = model.receiveCompletion(taskId);
        hasNext = !output.stop;
        return output;
    }

    /**
     * Cancel the ongoing generation process.
     */
    public void cancel() {
        model.cancelCompletion(taskId);
        hasNext = false;
    }
}
