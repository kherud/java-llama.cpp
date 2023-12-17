package examples;

import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;
import de.kherud.llama.ModelResolver;

public class InfillExample {

	public static void main(String... args) {
		LlamaModel.setLogger((level, message) -> System.out.print(message));
		ModelParameters modelParams = new ModelParameters()
				.setNGpuLayers(43);

		String prefix = "def remove_non_ascii(s: str) -> str:\n    \"\"\" ";
		String suffix = "\n    return result\n";
		String modelName = System.getProperty("model.name");
		String modelPath = ModelResolver.getPathToModel(modelName);
		try (LlamaModel model = new LlamaModel(modelPath, modelParams)) {
			System.out.print(prefix);
			for (LlamaModel.Output output : model.generate(prefix, suffix)) {
				System.out.print(output);
			}
			System.out.print(suffix);
		}
	}
}
