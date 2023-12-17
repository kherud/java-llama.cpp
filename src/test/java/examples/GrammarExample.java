package examples;

import de.kherud.llama.ModelResolver;
import java.util.HashMap;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;

public class GrammarExample {

	public static void main(String... args) {
		String grammar = "root  ::= (expr \"=\" term \"\\n\")+\n" +
				"expr  ::= term ([-+*/] term)*\n" +
				"term  ::= [0-9]";
		InferenceParameters params = new InferenceParameters().setGrammar(grammar);
		String modelName = System.getProperty("model.name");
		String modelPath = ModelResolver.getPathToModel(modelName);
		try (LlamaModel model = new LlamaModel(modelPath)) {
			for (LlamaModel.Output output : model.generate("", params)) {
				System.out.print(output);
			}
		}
	}

}
