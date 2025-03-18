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
		model = new LlamaModel(
					new ModelParameters()
						.setCtxSize(128)
						.setModel("models/codellama-7b.Q2_K.gguf")
						.setGpuLayers(43)
						.enableLogTimestamps()
						.enableLogPrefix()
				);
	}

	@AfterClass
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testMultiTurnChat() {
	    List<Pair<String, String>> userMessages = new ArrayList<>();
	    userMessages.add(new Pair<>("user", "Recommend a good ML book."));
	    
	    InferenceParameters params = new InferenceParameters("")
	            .setMessages("You are a Book Recommendation System", userMessages)
	            .setTemperature(0.7f)
	            .setNPredict(50);

	    String response1 = model.completeChat(params);
	    Assert.assertNotNull(response1);
	    
	    userMessages.add(new Pair<>("assistant", response1));
	    userMessages.add(new Pair<>("user", "How does it compare to 'Hands-on ML'?"));

	    params.setMessages("Book", userMessages);
	    String response2 = model.completeChat(params);

	    Assert.assertNotNull(response2);
	    Assert.assertNotEquals(response1, response2);
	}
	
	@Test
	public void testEmptyInput() {
	    List<Pair<String, String>> userMessages = new ArrayList<>();
	    userMessages.add(new Pair<>("user", ""));

	    InferenceParameters params = new InferenceParameters("A book recommendation system.")
	            .setMessages("Book", userMessages)
	            .setTemperature(0.5f)
	            .setNPredict(20);

	    String response = model.completeChat(params);
	    Assert.assertNotNull(response);
	    Assert.assertFalse(response.isEmpty());
	}
	
	@Test
	public void testStopString() {
	    List<Pair<String, String>> userMessages = new ArrayList<>();
	    userMessages.add(new Pair<>("user", "Tell me about AI ethics."));

	    InferenceParameters params = new InferenceParameters("A book recommendation system.")
	            .setMessages("AI", userMessages)
	            .setStopStrings("\"\"\"") // Ensures stopping at proper place
	            .setTemperature(0.7f)
	            .setNPredict(50);

	    String response = model.completeChat(params);
	    Assert.assertNotNull(response);
	    Assert.assertFalse(response.contains("\"\"\""));
	}
	
	@Test
	public void testFixedSeed() {
	    List<Pair<String, String>> userMessages = new ArrayList<>();
	    userMessages.add(new Pair<>("user", "What is reinforcement learning?"));

	    InferenceParameters params = new InferenceParameters("AI Chatbot.")
	            .setMessages("AI", userMessages)
	            .setTemperature(0f)
	            .setSeed(42) // Fixed seed for reproducibility
	            .setNPredict(50);

	    String response1 = model.completeChat(params);
	    String response2 = model.completeChat(params);

	    Assert.assertEquals(response1, response2); // Responses should be identical
	}
	
	@Test
	public void testNonEnglishInput() {
	    List<Pair<String, String>> userMessages = new ArrayList<>();
	    userMessages.add(new Pair<>("user", "Quel est le meilleur livre sur l'apprentissage automatique ?"));

	    InferenceParameters params = new InferenceParameters("A book recommendation system.")
	            .setMessages("Book", userMessages)
	            .setTemperature(0.7f)
	            .setNPredict(50);

	    String response = model.completeChat(params);
	    Assert.assertNotNull(response);
	    Assert.assertTrue(response.length() > 5); // Ensure some response is generated
	}

}
