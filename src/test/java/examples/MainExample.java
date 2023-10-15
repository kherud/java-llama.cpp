package examples;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;

public class MainExample {

    public static void main(String... args) throws IOException {
        LlamaModel.setLogger((level, message) -> System.out.print(message));
        ModelParameters modelParams = new ModelParameters.Builder()
                .setNGpuLayers(43)
                .build();
        InferenceParameters inferParams = new InferenceParameters.Builder()
                .setTemperature(0.7f)
                .setPenalizeNl(true)
                .setMirostat(InferenceParameters.MiroStat.V2)
                .setAntiPrompt(new String[]{"\n"})
                .build();

        String modelPath = "/run/media/konstantin/Seagate/models/llama2/llama-2-13b-chat/ggml-model-q4_0.gguf";
        String system = "This is a conversation between User and Llama, a friendly chatbot.\n" +
                "Llama is helpful, kind, honest, good at writing, and never fails to answer any " +
                "requests immediately and with precision.\n";
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        try (LlamaModel model = new LlamaModel(modelPath, modelParams)) {
            System.out.print(system);
            String prompt = system;
            while (true) {
                prompt += "\nUser: ";
                System.out.print("\nUser: ");
                String input = reader.readLine();
                prompt += input;
                System.out.print("Llama: ");
                prompt += "\nLlama: ";
//                String answer = model.complete(prompt, inferParams);
//                System.out.print(answer);
//                prompt += answer;
                for (String output : model.generate(prompt, inferParams)) {
                    System.out.print(output);
                    prompt += output;
                }
            }
        }
    }
}
