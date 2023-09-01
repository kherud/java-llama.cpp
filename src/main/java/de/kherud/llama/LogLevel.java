package de.kherud.llama;

import de.kherud.llama.foreign.LlamaLibrary;

/**
 * This enum represents the native log levels of llama.cpp.
 */
public enum LogLevel {

	DEBUG(-1),
	INFO(LlamaLibrary.llama_log_level.LLAMA_LOG_LEVEL_INFO),
	WARN(LlamaLibrary.llama_log_level.LLAMA_LOG_LEVEL_WARN),
	ERROR(LlamaLibrary.llama_log_level.LLAMA_LOG_LEVEL_ERROR);

	private final int code;

	LogLevel(int code) {
		this.code = code;
	}

	/**
	 * Returns the native log level code of this option
	 *
	 * @return the native code
	 */
	int getCode() {
		return code;
	}

	/**
	 * Returns a Java enum option given a native log level code.
	 * For unknown native codes the level {@link #DEBUG} is returned instead of throwing.
	 *
	 * @param code the native log level code
	 * @return the Java level representation
	 */
	static LogLevel fromCode(int code) {
		switch (code) {
			case LlamaLibrary.llama_log_level.LLAMA_LOG_LEVEL_INFO:
				return INFO;
			case LlamaLibrary.llama_log_level.LLAMA_LOG_LEVEL_WARN:
				return WARN;
			case LlamaLibrary.llama_log_level.LLAMA_LOG_LEVEL_ERROR:
				return ERROR;
			default:
				return DEBUG;
		}
	}

}
