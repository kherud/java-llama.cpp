package de.kherud.llama;

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

	// taken from org.json.JSONObject#quote(String, Writer)
	String toJsonString(String text) {
		if (text == null) return null;
		StringBuilder builder = new StringBuilder((text.length()) + 2);

		char b;
		char c = 0;
		String hhhh;
		int i;
		int len = text.length();

		builder.append('"');
		for (i = 0; i < len; i += 1) {
			b = c;
			c = text.charAt(i);
			switch (c) {
				case '\\':
				case '"':
					builder.append('\\');
					builder.append(c);
					break;
				case '/':
					if (b == '<') {
						builder.append('\\');
					}
					builder.append(c);
					break;
				case '\b':
					builder.append("\\b");
					break;
				case '\t':
					builder.append("\\t");
					break;
				case '\n':
					builder.append("\\n");
					break;
				case '\f':
					builder.append("\\f");
					break;
				case '\r':
					builder.append("\\r");
					break;
				default:
					if (c < ' ' || (c >= '\u0080' && c < '\u00a0') || (c >= '\u2000' && c < '\u2100')) {
						builder.append("\\u");
						hhhh = Integer.toHexString(c);
						builder.append("0000", 0, 4 - hhhh.length());
						builder.append(hhhh);
					} else {
						builder.append(c);
					}
			}
		}
		builder.append('"');
		return builder.toString();
	}
}
