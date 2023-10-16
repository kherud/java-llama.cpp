package examples;

import java.util.HashMap;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;

public class GrammarExample {

	public static void main(String... args) {
		String grammar = "root  ::= (expr \"=\" term \"\\n\")+\n" +
				"expr  ::= term ([-+*/] term)*\n" +
				"term  ::= [0-9]";
		InferenceParameters params = new InferenceParameters().setGrammar(grammar);

		String filePath = "/run/media/konstantin/Seagate/models/llama2/llama-2-13b-chat/gguf-model-q4_0.bin";
		try (LlamaModel model = new LlamaModel(filePath)) {
			for (LlamaModel.Output output : model.generate("", params)) {
				System.out.print(output);
			}
		}
	}

}
