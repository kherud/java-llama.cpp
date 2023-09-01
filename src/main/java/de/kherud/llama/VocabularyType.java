package de.kherud.llama;

import de.kherud.llama.foreign.LlamaLibrary;

/**
 * This enum represents the native vocabulary types of llama.cpp.
 */
public enum VocabularyType {

	SENTENCE_PIECE(LlamaLibrary.llama_vocab_type.LLAMA_VOCAB_TYPE_SPM),
	BYTE_PAIR(LlamaLibrary.llama_vocab_type.LLAMA_VOCAB_TYPE_BPE);

	private final int code;

	VocabularyType(int code) {
		this.code = code;
	}

	/**
	 * Returns a Java enum option given a native vocabulary type.
	 * For unknown native codes {@link #BYTE_PAIR} is returned instead of throwing.
	 *
	 * @param code the native vocabulary type
	 * @return the Java vocabulary type representation
	 */
	static VocabularyType fromCode(int code) {
		if (code == LlamaLibrary.llama_vocab_type.LLAMA_VOCAB_TYPE_SPM) {
			return SENTENCE_PIECE;
		} else {
			return BYTE_PAIR;
		}
	}

}
