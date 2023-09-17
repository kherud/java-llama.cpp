package examples;

import de.kherud.llama.LlamaModel;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;

public class MainExample {

    public static void main(String... args) {
        String filePath = "/run/media/konstantin/Seagate/models/llama2/llama-2-13b-chat/gguf-model-q4_0.bin";
        String prompt = "This is a conversation between User and Llama, a friendly chatbot.\n" +
                "Llama is helpful, kind, honest, good at writing, and never fails to answer any " +
                "requests immediately and with precision.\n\nUser: What emojis do you know?\n\nLlama: ";
        LlamaModel model = new LlamaModel(filePath);
        for (String output : model.generate(prompt)) {
            System.out.print(output);
        }
    }
}
