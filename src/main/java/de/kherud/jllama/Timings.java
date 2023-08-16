package de.kherud.jllama;

import java.util.Arrays;
import java.util.List;

import com.sun.jna.Structure;

public class Timings extends Structure {

	public double t_start_ms;
	public double t_end_ms;
	public double t_load_ms;
	public double t_sample_ms;
	public double t_p_eval_ms;
	public double t_eval_ms;

	public int n_sample;
	public int n_p_eval;
	public int n_eval;

	@Override
	protected List<String> getFieldOrder() {
		return Arrays.asList(
				"t_start_ms",
				"t_end_ms",
				"t_load_ms",
				"t_sample_ms",
				"t_p_eval_ms",
				"t_eval_ms",
				"n_sample",
				"n_p_eval",
				"n_eval"
		);
	}

	public static class ByReference extends Timings implements Structure.ByReference { }
}
