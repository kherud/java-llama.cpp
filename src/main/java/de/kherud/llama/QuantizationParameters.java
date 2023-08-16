package de.kherud.llama;

import com.sun.jna.Structure;

public class QuantizationParameters extends Structure {

	public int nthread;
	public int ftype;
	public boolean allow_requantize;
	public boolean quantize_output_tensor;

	public static class ByReference extends QuantizationParameters implements Structure.ByReference { }

}
