package de.kherud.llama;

import java.util.ArrayList;
import java.util.List;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;

public class LlamaChatModelTest {

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {
		model = new LlamaModel(new ModelParameters().setCtxSize(128).setModel("models/codellama-7b.Q2_K.gguf")
				.setGpuLayers(43).enableLogTimestamps().enableLogPrefix());
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
				.setMessages("You are a Book Recommendation System", userMessages).setTemperature(0.7f).setNPredict(50);

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
				.setMessages("Book", userMessages).setTemperature(0.5f).setNPredict(20);

		String response = model.completeChat(params);
		Assert.assertNotNull(response);
		Assert.assertFalse(response.isEmpty());
	}

	@Test
	public void testStopString() {
		List<Pair<String, String>> userMessages = new ArrayList<>();
		userMessages.add(new Pair<>("user", "Tell me about AI ethics."));

		InferenceParameters params = new InferenceParameters("A book recommendation system.")
				.setMessages("AI", userMessages).setStopStrings("\"\"\"") // Ensures stopping at proper place
				.setTemperature(0.7f).setNPredict(50);

		String response = model.completeChat(params);
		Assert.assertNotNull(response);
		Assert.assertFalse(response.contains("\"\"\""));
	}

	@Ignore
	public void testFixedSeed() {
		List<Pair<String, String>> userMessages = new ArrayList<>();
		userMessages.add(new Pair<>("user", "What is reinforcement learning?"));

		InferenceParameters params = new InferenceParameters("AI Chatbot.").setMessages("AI", userMessages)
				.setTemperature(0f).setSeed(42) // Fixed seed for reproducibility
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
				.setMessages("Book", userMessages).setTemperature(0.7f).setNPredict(50);

		String response = model.completeChat(params);
		Assert.assertNotNull(response);
		Assert.assertTrue(response.length() > 5); // Ensure some response is generated
	}

	@Test
	public void testCompletions() {
		InferenceParameters params = new InferenceParameters("Tell me a joke?").setTemperature(0.7f).setNPredict(50)
				.setNProbs(1).setPostSamplingProbs(true).setStopStrings("\"\"\"");

		// Call handleCompletions with streaming = false and task type = completion
		String response = model.handleCompletions(params.toString(), false, 0);

	    
	    // Parse the response JSON
	    JsonNode responseNode = JsonUtils.INSTANCE.jsonToNode(response);
	    
	    // Verify basic response structure
	    Assert.assertNotNull("Response should not be null", response);
	    Assert.assertEquals("Completion type should be 'completion'", "completion", responseNode.get("type").asText());
	    Assert.assertEquals("Streaming should be false", false, responseNode.get("streaming").asBoolean());
	    Assert.assertTrue("Should have a completion_id", responseNode.has("completion_id"));
	    
	    // Verify result content
	    JsonNode result = responseNode.get("result");
	    Assert.assertNotNull("Result should not be null", result);
	    Assert.assertTrue("Content should not be null", result.has("content"));
	    Assert.assertFalse("Content should not be empty", result.get("content").asText().isEmpty());
	    
	    System.out.println("Completion result: " + result.get("content").asText());
	}

	@Test
	public void testStreamingCompletions() {
		InferenceParameters params = new InferenceParameters("Tell me a joke?").setTemperature(0.7f).setNPredict(50)
				.setNProbs(1).setPostSamplingProbs(true).setStopStrings("\"\"\"");

		String response = model.handleCompletions(params.toString(), true, 0);

		JsonNode node = JsonUtils.INSTANCE.jsonToNode(response);

		ArrayNode taskIdsNode = (ArrayNode) node.get("task_ids");
		Assert.assertTrue("Should have at least one task ID", taskIdsNode.size() > 0);

		int taskId = taskIdsNode.get(0).asInt();
		System.out.println("Using task ID: " + taskId + " for streaming");

		// For collecting results
		StringBuilder fullContent = new StringBuilder();
		List<JsonNode> tokenInfoList = new ArrayList<>();
		boolean isFinal = false;
		int chunkCount = 0;

		// Get streaming chunks until completion
		while (!isFinal && chunkCount < 51) { // Limit to prevent infinite loop in test
			String chunkResponse = model.getNextStreamResult(taskId);
			JsonNode chunkNode = JsonUtils.INSTANCE.jsonToNode(chunkResponse);

			// Verify chunk structure
			Assert.assertEquals("Type should be 'stream_chunk'", "stream_chunk", chunkNode.get("type").asText());
			Assert.assertEquals("Task ID should match", taskId, chunkNode.get("task_id").asInt());

			JsonNode result = chunkNode.get("result");
			Assert.assertNotNull("Result should not be null", result);

			// Extract and accumulate content
			if (result.has("content")) {
				String chunkContent = result.get("content").asText();
				fullContent.append(chunkContent);

				System.out.println("\nChunk #" + chunkCount + ": \"" + chunkContent + "\"");

				// Check for token probabilities
				if (result.has("completion_probabilities")) {
					ArrayNode probs = (ArrayNode) result.get("completion_probabilities");
					if (probs.size() > 0) {
						tokenInfoList.add(result);

						// Log top token options for this chunk
						JsonNode firstToken = probs.get(0);
						ArrayNode topProbs = (ArrayNode) firstToken.get("top_probs");
						System.out.println("  Token alternatives:");
						for (JsonNode prob : topProbs) {
							String token = prob.get("token").asText();
							double probability = prob.get("prob").asDouble();
							System.out.printf("    \"%s\" (%.4f)%n", token, probability);
						}
					}
				}
			}

			isFinal = chunkNode.get("is_final").asBoolean();
			chunkCount++;
		}

		// Verify results
		Assert.assertTrue("Should have received at least one chunk", chunkCount > 0);
		Assert.assertTrue("Final chunk should have been received", isFinal);
		Assert.assertFalse("Accumulated content should not be empty", fullContent.toString().isEmpty());

		System.out.println("\nFinal content from streaming: \"" + fullContent + "\"");
		System.out.println("Received " + chunkCount + " chunks in total");

	}

}
