package examples;

import de.kherud.llama.ChatMessage;
import de.kherud.llama.ChatRequest;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class ChatExample {

    public static void main(String... args) throws Exception {
        ModelParameters modelParams = new ModelParameters()
                .setModel("models/codellama-7b.Q2_K.gguf")
                .setGpuLayers(43);
        try (LlamaModel model = new LlamaModel(modelParams)) {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
            List<ChatMessage> messages = new ArrayList<>();
            messages.add(new ChatMessage(ChatMessage.Role.SYSTEM, "You are a helpful assistant."));
            while (true) {
                System.out.print("User: ");
                String input = reader.readLine();
                messages.add(new ChatMessage(ChatMessage.Role.USER, input));
                ChatRequest request = new ChatRequest(messages, false);
                ChatMessage response = (ChatMessage) model.chat(request);
                System.out.println("Assistant: " + response.getContent());
                messages.add(response);
            }
        }
    }
}
