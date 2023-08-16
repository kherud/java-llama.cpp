package de.kherud.jllama;

import java.util.Arrays;
import java.util.List;

import com.sun.jna.Structure;

public class TokenData extends Structure {

	public int id; // token id
	public float logit; // log-odds of the token
	public float p; // probability of the token

	@Override
	protected List<String> getFieldOrder() {
		return Arrays.asList(
				"id",
				"logit",
				"p"
		);
	}

	public static class ByReference extends TokenData implements Structure.ByReference { }
}
