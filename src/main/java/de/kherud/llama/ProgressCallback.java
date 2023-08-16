package de.kherud.llama;

import com.sun.jna.Callback;

public interface ProgressCallback extends Callback {
	void callback(float progress);

}
