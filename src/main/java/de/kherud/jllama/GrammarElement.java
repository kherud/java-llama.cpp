package de.kherud.jllama;

import java.util.Arrays;
import java.util.List;

import com.sun.jna.Structure;

public class GrammarElement extends Structure {

	public int type;
	public int value;

	public enum Type {
		// Define the values of the enum here, for example:
		TYPE1,
		TYPE2;
		// ... and so on for all enum values
	}

	@Override
	protected List<String> getFieldOrder() {
		return Arrays.asList(
				"type",
				"value"
		);
	}

	public static class ByReference extends GrammarElement implements Structure.ByReference { }
}
