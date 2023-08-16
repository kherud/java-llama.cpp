package de.kherud.llama;

import com.sun.jna.Callback;
import com.sun.jna.Pointer;

public interface LogCallback extends Callback {

	void callback(int level, String text, Pointer user_data);

}
