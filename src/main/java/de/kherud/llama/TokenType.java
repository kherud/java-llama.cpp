package de.kherud.llama;

import de.kherud.llama.foreign.LlamaLibrary;

/**
 * This enum represents the native token types of llama.cpp.
 */
public enum TokenType {

	UNDEFINED(LlamaLibrary.llama_token_type.LLAMA_TOKEN_TYPE_UNDEFINED),
	NORMAL(LlamaLibrary.llama_token_type.LLAMA_TOKEN_TYPE_NORMAL),
	UNKNOWN(LlamaLibrary.llama_token_type.LLAMA_TOKEN_TYPE_UNKNOWN),
	CONTROL(LlamaLibrary.llama_token_type.LLAMA_TOKEN_TYPE_CONTROL),
	USER_DEFINED(LlamaLibrary.llama_token_type.LLAMA_TOKEN_TYPE_USER_DEFINED),
	UNUSED(LlamaLibrary.llama_token_type.LLAMA_TOKEN_TYPE_UNUSED),
	BYTE(LlamaLibrary.llama_token_type.LLAMA_TOKEN_TYPE_BYTE);

	private final int code;

	TokenType(int code) {
		this.code = code;
	}

	/**
	 * Returns a Java enum option given a native token type.
	 * For unknown native codes {@link #UNKNOWN} is returned instead of throwing.
	 *
	 * @param code the native token type
	 * @return the Java token type representation
	 */
	static TokenType fromCode(int code) {
		switch (code) {
			case LlamaLibrary.llama_token_type.LLAMA_TOKEN_TYPE_UNDEFINED:
				return UNDEFINED;
			case LlamaLibrary.llama_token_type.LLAMA_TOKEN_TYPE_NORMAL:
				return NORMAL;
			case LlamaLibrary.llama_token_type.LLAMA_TOKEN_TYPE_CONTROL:
				return CONTROL;
			case LlamaLibrary.llama_token_type.LLAMA_TOKEN_TYPE_USER_DEFINED:
				return USER_DEFINED;
			case LlamaLibrary.llama_token_type.LLAMA_TOKEN_TYPE_UNUSED:
				return UNUSED;
			case LlamaLibrary.llama_token_type.LLAMA_TOKEN_TYPE_BYTE:
				return BYTE;
			case LlamaLibrary.llama_token_type.LLAMA_TOKEN_TYPE_UNKNOWN:
			default:
				return UNKNOWN;
		}
	}

}
