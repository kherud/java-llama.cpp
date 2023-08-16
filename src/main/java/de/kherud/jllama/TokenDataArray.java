package de.kherud.jllama;

import java.util.Arrays;
import java.util.List;

import com.sun.jna.Structure;

public class TokenDataArray extends Structure {

	public TokenData.ByReference data;
	public long size;
	public boolean sorted;

	@Override
	protected List<String> getFieldOrder() {
		return Arrays.asList(
				"data",
				"size",
				"sorted"
		);
	}

	public static class ByReference extends TokenDataArray implements Structure.ByReference { }
}
