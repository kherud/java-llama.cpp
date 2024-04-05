package examples;

import de.kherud.llama.ModelParameters;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;

public class GrammarExample {

	public static void main(String... args) {
		String grammar = "root  ::= (expr \"=\" term \"\\n\")+\n" +
				"expr  ::= term ([-+*/] term)*\n" +
				"term  ::= [0-9]";
		ModelParameters modelParams = new ModelParameters()
				.setModelFilePath("models/mistral-7b-instruct-v0.2.Q2_K.gguf")
				.setModelUrl("https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q2_K.gguf");
		InferenceParameters inferParams = new InferenceParameters("")
				.setGrammar(grammar);
		try (LlamaModel model = new LlamaModel(modelParams)) {
			for (LlamaModel.Output output : model.generate(inferParams)) {
				System.out.print(output);
			}
		}
	}

}
