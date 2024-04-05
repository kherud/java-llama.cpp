package de.kherud.llama.args;

import java.util.HashMap;
import java.util.Map;

/**
 * The Java library re-uses most of the llama.cpp server code, which mostly works with JSONs. Thus, the complexity and
 * maintainability is much lower if we work with JSONs. This class provides a simple abstraction to easily create
 * JSON object strings by filling a <code>Map&lt;String, String&gt;</code> with key value pairs.
 */
abstract class JsonParameters {

	// We save parameters directly as a String map here, to re-use as much as possible of the (json-based) C++ code.
	// The JNI code for a proper Java-typed data object is comparatively too complex and hard to maintain.
	final Map<String, String> parameters = new HashMap<>();

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("{\n");
		int i = 0;
		for (Map.Entry<String, String> entry : parameters.entrySet()) {
			String key = entry.getKey();
			String value = entry.getValue();
			builder.append("\t\"")
					.append(key)
					.append("\": ")
					.append(value);
			if (i++ < parameters.size() - 1) {
				builder.append(",");
			}
			builder.append("\n");
		}
		builder.append("}");
		return builder.toString();
	}

	String toJsonString(String text) {
		if (text == null) return null;
		StringBuilder builder = new StringBuilder((text.length()) + 2);
		builder.append('"');
		for (int i = 0; i < text.length(); i++) {
			char c = text.charAt(i);
			if (c == '"' || c == '\\') {
				builder.append('\\');
			}
			builder.append(c);
		}
		builder.append('"');
		return builder.toString();
	}
}
