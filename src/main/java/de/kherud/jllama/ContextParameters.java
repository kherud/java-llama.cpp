package de.kherud.jllama;

import java.util.Arrays;
import java.util.List;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.ptr.FloatByReference;

public class ContextParameters extends Structure {

	public int seed;
	public int n_ctx;
	public int n_batch;
	public int n_gqa;
	public float rms_norm_eps;
	public int n_gpu_layers;
	public int main_gpu;

	public FloatByReference tensor_split;

	public float rope_freq_base;
	public float rope_freq_scale;

	public ProgressCallback progress_callback;
	public Pointer progress_callback_user_data;

	public boolean low_vram;
	public boolean mul_mat_q;
	public boolean f16_kv;
	public boolean logits_all;
	public boolean vocab_only;
	public boolean use_mmap;
	public boolean use_mlock;
	public boolean embedding;

	@Override
	protected List<String> getFieldOrder() {
		return Arrays.asList(
				"seed",
				"n_ctx",
				"n_batch",
				"n_gqa",
				"rms_norm_eps",
				"n_gpu_layers",
				"main_gpu",
				"tensor_split",
				"rope_freq_base",
				"rope_freq_scale",
				"progress_callback",
				"progress_callback_user_data",
				"low_vram",
				"mul_mat_q",
				"f16_kv",
				"logits_all",
				"vocab_only",
				"use_mmap",
				"use_mlock",
				"embedding"
		);
	}

	public static class ByReference extends ContextParameters implements Structure.ByReference { }


}
