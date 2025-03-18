package de.kherud.llama;

import java.util.ArrayList;
import java.util.List;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.Assert;


public class LlamaChatModelTest {
	
	private static LlamaModel model;
	
	@BeforeClass
	public static void setup() {
//		LlamaModel.setLogger(LogFormat.TEXT, (level, msg) -> System.out.println(level + ": " + msg));
		model = new LlamaModel(
				new ModelParameters()
						.setCtxSize(128)
						.setModel("models/codellama-7b.Q2_K.gguf")
						//.setModelUrl("https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.Q2_K.gguf")
						.setGpuLayers(43)
						.enableEmbedding().enableLogTimestamps().enableLogPrefix()
		);
	}

	@AfterClass
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
	}
	
	@Test
	public void testChat() {
		List<Pair<String, String>> userMessages = new ArrayList<>();
        userMessages.add(new Pair<>("user", "What is the best book for machine learning?"));
        
        InferenceParameters params = new InferenceParameters("A book recommendation system.")
				.setMessages("Book", userMessages)
				.setTemperature(0.0f)
				.setStopStrings("\"\"\"")
				.setNPredict(10)
				.setSeed(42);
        
        String assistantResponse = model.completeChat(params);
        
        Assert.assertNotNull(assistantResponse);
        
        Assert.assertEquals(params.get("prompt"), "\"<|im_start|>system\\nBook<|im_end|>\\n<|im_start|>user\\nWhat is the best book for machine learning?<|im_end|>\\n<|im_start|>assistant\\n\"");
        
        userMessages.add(new Pair<>("assistant", assistantResponse));
        userMessages.add(new Pair<>("user", "that is great book for machine learning?, what about linear algebra"));
        
        params = new InferenceParameters("A book recommendation system.")
				.setMessages("Book", userMessages)
				.setTemperature(0.0f)
				.setStopStrings("\"\"\"")
				.setNPredict(10)
				.setSeed(42);
        
        
        assistantResponse = model.completeChat(params);
        Assert.assertNotNull(assistantResponse);
        
        Assert.assertEquals(params.get("prompt"), "\"<|im_start|>system\\nBook<|im_end|>\\n<|im_start|>user\\nWhat is the best book for machine learning?<|im_end|>\\n<|im_start|>assistant\\nWhat is the best book for machine learning?<<|im_end|>\\n<|im_start|>user\\nthat is great book for machine learning?, what about linear algebra<|im_end|>\\n<|im_start|>assistant\\n\"");
        
        
	}

}
