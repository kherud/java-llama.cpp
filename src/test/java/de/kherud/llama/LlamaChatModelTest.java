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
		model = new LlamaModel(new ModelParameters().setCtxSize(128).setModel("models/Llama-3.2-3B-Instruct-Q8_0.gguf")
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

		// Call handleCompletions with streaming = false and task type = chat
		String response1 = model.handleCompletions(params.toString(), false, 0);

		// Parse the response JSON
		JsonNode responseNode1 = JsonUtils.INSTANCE.jsonToNode(response1);

		// Verify response structure
		Assert.assertNotNull("Response should not be null", response1);
		Assert.assertEquals("Completion type should be 'completion'", "completion", responseNode1.get("type").asText());
		Assert.assertTrue("Should have a completion_id", responseNode1.has("completion_id"));

		// Extract content from result
		JsonNode result1 = responseNode1.get("result");
		Assert.assertNotNull("Result should not be null", result1);
		JsonNode choicesNode1 = result1.get("choices");
		JsonNode messageNode1 = choicesNode1.get(0).get("message");
		JsonNode contentNode1 = messageNode1.get("content");
		String content1 = contentNode1.asText();
		Assert.assertFalse("Content should not be empty", content1.isEmpty());

		// Get the completion_id from the first response
		String completionId1 = responseNode1.get("completion_id").asText();

		// Continue the conversation with a more specific follow-up
		userMessages.add(new Pair<>("assistant", content1));
		userMessages.add(new Pair<>("user",
				"Can you compare that book specifically with 'Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow'?"));

		params.setMessages("Book", userMessages);
		String response2 = model.handleCompletions(params.toString(), false, 0);

		// Parse the second response
		JsonNode responseNode2 = JsonUtils.INSTANCE.jsonToNode(response2);
		JsonNode result2 = responseNode2.get("result");
		JsonNode choicesNode2 = result2.get("choices");
		JsonNode messageNode2 = choicesNode2.get(0).get("message");
		JsonNode contentNode2 = messageNode2.get("content");
		String content2 = contentNode2.asText();
		String completionId2 = responseNode2.get("completion_id").asText();

		// Better assertions
		Assert.assertNotNull("Second response should not be null", content2);

		// Check that completion IDs are different (indicating separate completions)
		Assert.assertNotEquals("Completion IDs should be different", completionId1, completionId2);

		// Check that the second response contains specific text related to the
		// follow-up question
		Assert.assertTrue("Response should mention 'Hands-on Machine Learning'",
				content2.contains("Hands-on Machine Learning") || content2.contains("Hands-on ML")
						|| content2.contains("Scikit-Learn") || content2.contains("Keras")
						|| content2.contains("TensorFlow"));

		// Check that the model is actually responding to the comparison request
		Assert.assertTrue("Response should contain comparison language",
				content2.contains("compare") || content2.contains("comparison") || content2.contains("differ")
						|| content2.contains("similar") || content2.contains("unlike") || content2.contains("whereas")
						|| content2.contains("while"));
	}

	@Test
	public void testEmptyInput() {
		List<Pair<String, String>> userMessages = new ArrayList<>();
		userMessages.add(new Pair<>("user", ""));

		InferenceParameters params = new InferenceParameters("A book recommendation system.")
				.setMessages("Book", userMessages).setTemperature(0.5f).setNPredict(20);

		// Call handleCompletions
		String response = model.handleCompletions(params.toString(), false, 0);

		// Parse the response JSON
		JsonNode responseNode = JsonUtils.INSTANCE.jsonToNode(response);
		JsonNode result = responseNode.get("result");
		JsonNode choicesNode = result.get("choices");
		JsonNode messageNode = choicesNode.get(0).get("message");
		JsonNode contentNode = messageNode.get("content");
		String content = contentNode.asText();

		Assert.assertNotNull("Response should not be null", content);
		Assert.assertFalse("Content should not be empty", content.isEmpty());
	}

	@Test
	public void testStopString() {
		List<Pair<String, String>> userMessages = new ArrayList<>();
		userMessages.add(new Pair<>("user", "Tell me about AI ethics."));

		InferenceParameters params = new InferenceParameters("A book recommendation system.")
				.setMessages("AI", userMessages).setStopStrings("\"\"\"") // Ensures stopping at proper place
				.setTemperature(0.7f).setNPredict(50);

		// Call handleCompletions
		String response = model.handleCompletions(params.toString(), false, 0);

		// Parse the response JSON
		JsonNode responseNode = JsonUtils.INSTANCE.jsonToNode(response);
		JsonNode result = responseNode.get("result");
		JsonNode choicesNode = result.get("choices");
		JsonNode messageNode = choicesNode.get(0).get("message");
		JsonNode contentNode = messageNode.get("content");
		String content = contentNode.asText();

		Assert.assertNotNull("Response should not be null", content);
		Assert.assertFalse("Content should contain stop string", content.contains("\"\"\""));
	}

	@Test
	public void testFixedSeed() {
		List<Pair<String, String>> userMessages = new ArrayList<>();
		userMessages.add(new Pair<>("user", "What is reinforcement learning?"));

		InferenceParameters params = new InferenceParameters("AI Chatbot.").setMessages("AI", userMessages)
				.setTemperature(0f).setSeed(42) // Fixed seed for reproducibility
				.setNPredict(50);

		// Call handleCompletions for the first response
		String response1 = model.handleCompletions(params.toString(), false, 0);

		// Parse the first response JSON
		JsonNode responseNode1 = JsonUtils.INSTANCE.jsonToNode(response1);
		JsonNode result1 = responseNode1.get("result");
		JsonNode choicesNode1 = result1.get("choices");
		JsonNode messageNode1 = choicesNode1.get(0).get("message");
		JsonNode contentNode1 = messageNode1.get("content");
		String content1 = contentNode1.asText();

		// Call handleCompletions again with the same parameters
		String response2 = model.handleCompletions(params.toString(), false, 0);

		// Parse the second response JSON
		JsonNode responseNode2 = JsonUtils.INSTANCE.jsonToNode(response2);
		JsonNode result2 = responseNode2.get("result");
		JsonNode choicesNode2 = result2.get("choices");
		JsonNode messageNode2 = choicesNode2.get(0).get("message");
		JsonNode contentNode2 = messageNode2.get("content");
		String content2 = contentNode2.asText();

		Assert.assertEquals("Responses with same seed should be identical", content1, content2);
	}

	@Test
	public void testNonEnglishInput() {
		List<Pair<String, String>> userMessages = new ArrayList<>();
		userMessages.add(new Pair<>("user", "Quel est le meilleur livre sur l'apprentissage automatique ?"));

		InferenceParameters params = new InferenceParameters("A book recommendation system.")
				.setMessages("Book", userMessages).setTemperature(0.7f).setNPredict(50);

		// Call handleCompletions
		String response = model.handleCompletions(params.toString(), false, 0);

		// Parse the response JSON
		JsonNode responseNode = JsonUtils.INSTANCE.jsonToNode(response);
		JsonNode result = responseNode.get("result");
		JsonNode choicesNode = result.get("choices");
		JsonNode messageNode = choicesNode.get(0).get("message");
		JsonNode contentNode = messageNode.get("content");
		String content = contentNode.asText();

		Assert.assertNotNull("Response should not be null", content);
		Assert.assertTrue("Content should have sufficient length", content.length() > 5);
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
