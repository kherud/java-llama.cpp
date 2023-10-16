package examples;

import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;

public class InfillExample {

	public static void main(String... args) {
		LlamaModel.setLogger((level, message) -> System.out.print(message));
		ModelParameters modelParams = new ModelParameters.Builder()
				.setNGpuLayers(43)
				.build();

		String prefix = "def remove_non_ascii(s: str) -> str:\n    \"\"\" ";
		String suffix = "\n    return result\n";

		String modelPath = "/run/media/konstantin/Seagate/models/codellama-13b.q5_k_m.gguf";
		try (LlamaModel model = new LlamaModel(modelPath, modelParams)) {
			System.out.print(prefix);
			for (LlamaModel.Output output : model.generate(prefix, suffix)) {
				System.out.print(output);
			}
			System.out.print(suffix);
		}
	}
}
