package de.kherud.llama;

/**
 * This enum represents the native log levels of llama.cpp.
 */
public enum LogLevel {

	DEBUG(-1),
	INFO(4),
	WARN(3),
	ERROR(2);

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

}
