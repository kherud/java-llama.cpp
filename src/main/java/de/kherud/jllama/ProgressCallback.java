package de.kherud.jllama;

import com.sun.jna.Callback;

public interface ProgressCallback extends Callback {
	void callback(float progress);

}
