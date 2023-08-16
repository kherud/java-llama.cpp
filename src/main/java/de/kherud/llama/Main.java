package de.kherud.llama;

public class Main {

	public static void main(String... args) {
		System.out.println(LlamaCpp.INSTANCE.llama_print_system_info());
	}


}
