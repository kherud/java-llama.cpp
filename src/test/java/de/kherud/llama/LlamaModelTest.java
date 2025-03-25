package de.kherud.llama;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.regex.Pattern;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import de.kherud.llama.args.LogFormat;

public class LlamaModelTest {

	private static final String prefix = "def remove_non_ascii(s: str) -> str:\n    \"\"\" ";
	private static final String suffix = "\n    return result\n";
	private static final int nPredict = 10;

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {

		model = new LlamaModel(new ModelParameters()
				.setModel("models/Phi-4-mini-instruct-Q2_K.gguf")
				.setGpuLayers(43)
				.enableJinja());
	}

	@AfterClass
	public static void tearDown() throws Exception {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testGenerateAnswer() {
	    System.out.println("***** Running the test: testGenerateAnswer");
	    
	    // Create a map for logit bias
	    Map<Integer, Float> logitBias = new HashMap<>();
	    logitBias.put(2, 2.0f);
	    
	    // Create parameters using the InferenceParameters builder
	    InferenceParameters params = new InferenceParameters()
	        .setPrompt(prefix)
	        .setTemperature(0.95f)
	        .setStopStrings("\"\"\"")
	        .setNPredict(nPredict)
	        .setTokenIdBias(logitBias)
	        .setStream(true); // Set streaming to true
	    
	    // Get the JSON string from the parameters
	    String requestJson = params.toString();
	    
	    // Call handleCompletions with streaming enabled
	    String streamInitResponse = model.handleCompletions(requestJson, true);
	    
	    try {
	        // Parse the stream initialization response
	        
	        JsonNode responseObj = JsonUtils.INSTANCE.jsonToNode(streamInitResponse);
	        JsonNode taskIdsArray = responseObj.get("task_ids");
	        
	        // We should have at least one task ID
	        Assert.assertTrue(taskIdsArray.size() > 0);
	        int taskId = taskIdsArray.get(0).asInt();
	        
	        // Stream until we get all tokens or reach the end
	        int generated = 0;
	        boolean isComplete = false;
	        
	        while (!isComplete && generated < nPredict) {
	            // Get the next chunk of streaming results
	            String chunkResponse = model.getNextStreamResult(taskId);
	            JsonNode chunkObj = JsonUtils.INSTANCE.jsonToNode(chunkResponse);
	            
	            // Check if this is the final chunk
	            isComplete = chunkObj.get("is_final").asBoolean();
	            
	            // Extract and process the content
	            JsonNode resultObj = chunkObj.get("result");
	            if (resultObj.has("content")) {
	                String content = resultObj.get("content").asText();
	                if (!content.isEmpty()) {
	                    generated++;
	                }
	            }
	        }
	        
	        // Make sure we generated something within expected limits
	        Assert.assertTrue(generated > 0 && generated <= nPredict + 1);
	        
	        // Release the task to clean up resources
	        model.releaseTask(taskId);
	        
	    } catch (Exception e) {
	        Assert.fail("Failed during streaming test: " + e.getMessage());
	    }
	}
	
	@Ignore
	public void testGenerateInfill() {
	    System.out.println("***** Running the test: testGenerateInfill");
	    
	    // Create a map for logit bias
	    Map<Integer, Float> logitBias = new HashMap<>();
	    logitBias.put(2, 2.0f);
	    
	    // Create parameters using the InferenceParameters builder
	    InferenceParameters params = new InferenceParameters()
	        .setPrompt("")
	        .setInputPrefix(prefix)
	        .setInputSuffix(suffix)
	        .setTemperature(0.95f)
	        .setStopStrings("\"\"\"")
	        .setNPredict(nPredict)
	        .setTokenIdBias(logitBias)
	        .setSeed(42)
	        .setStream(true); // Set streaming to true
	    
	    // Get the JSON string from the parameters
	    String requestJson = params.toString();
	    
	    // Call handleInfill with streaming enabled
	    String streamInitResponse = model.handleInfill(requestJson, true);
	    
	    try {

	        JsonNode responseObj = JsonUtils.INSTANCE.jsonToNode(streamInitResponse);
	        JsonNode taskIdsArray = responseObj.get("task_ids");
	        
	        // We should have at least one task ID
	        Assert.assertTrue(taskIdsArray.size() > 0);
	        int taskId = taskIdsArray.get(0).asInt();
	        
	        // Stream until we get all tokens or reach the end
	        int generated = 0;
	        boolean isComplete = false;
	        
	        while (!isComplete && generated < nPredict) {
	            // Get the next chunk of streaming results
	            String chunkResponse = model.getNextStreamResult(taskId);
	            JsonNode chunkObj = JsonUtils.INSTANCE.jsonToNode(chunkResponse);
	            
	            // Check if this is the final chunk
	            isComplete = chunkObj.get("is_final").asBoolean();
	            
	            // Extract and process the content
	            JsonNode resultObj = chunkObj.get("result");
	            if (resultObj.has("content")) {
	                String content = resultObj.get("content").asText();
	                if (!content.isEmpty()) {
	                    // Process the generated content if needed
	                    System.out.println("Generated infill chunk: " + content);
	                    generated++;
	                }
	            }
	        }
	        
	        // Make sure we generated something within expected limits
	        Assert.assertTrue(generated > 0 && generated <= nPredict + 1);
	        
	        // Release the task to clean up resources
	        model.releaseTask(taskId);
	        
	    } catch (Exception e) {
	        Assert.fail("Failed during infill test: " + e.getMessage());
	    }
	}

	@Test
	public void testGenerateGrammar() {
		System.out.println("***** Running the test:  testGenerateGrammar");
		InferenceParameters params = new InferenceParameters().setPrompt(prefix)
				.setGrammar("root ::= (\"a\" | \"b\")+")
				.setNPredict(nPredict);
		List<Pair<String, String>> userMessages = new ArrayList<>();
		userMessages.add(new Pair<>("user", "Does not matter what I say, does it?"));
		
		String output = model.handleCompletions(params.toString(), false);
		JsonNode jsonNode = JsonUtils.INSTANCE.jsonToNode(output);
		JsonNode resultNode = jsonNode.get("result");
		String content = resultNode.get("content").asText();
		Assert.assertTrue(content.matches("[ab]+"));
		int generated = model.encode(content).length;
		
		Assert.assertTrue("generated should be between 0 and  11 but is " + generated, generated > 0 && generated <= nPredict + 1);
	}

	@Test
	public void testCompleteAnswer() {
		System.out.println("***** Running the test:  testGenerateGrammar");
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters().setPrompt(prefix)
				.setTemperature(0.95f)
				.setStopStrings("\"\"\"")
				.setNPredict(nPredict)
				.setTokenIdBias(logitBias)
				.setSeed(42);

		String output = model.handleCompletions(params.toString(),false);
		Assert.assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteInfillCustom() {
		System.out.println("***** Running the test:  testCompleteInfillCustom");
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters().setPrompt("code ")
				.setInputPrefix(prefix)
				.setInputSuffix(suffix)
				.setTemperature(0.95f)
				.setStopStrings("\"\"\"")
				.setNPredict(nPredict)
				.setTokenIdBias(logitBias)
				.setSeed(42);

		String output = model.handleCompletions(params.toString(),false);
		Assert.assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteGrammar() {
		System.out.println("***** Running the test:  testCompleteGrammar");
		InferenceParameters params = new InferenceParameters().setPrompt("code")
				.setGrammar("root ::= (\"a\" | \"b\")+")
				.setTemperature(0.6f)
				.setTopP(0.95f)
				.setNPredict(nPredict);
		String output = model.handleCompletions(params.toString(),false);
		JsonNode resultNode = JsonUtils.INSTANCE.jsonToNode(output).get("result");
		String content = resultNode.get("content").asText();
		Assert.assertTrue(content + " doesn't match [ab]+", content.matches("[ab]+"));
		int generated = model.encode(content).length;
		Assert.assertTrue("generated count is: " + generated,  generated > 0 && generated <= nPredict + 1);
		
	}

	@Test
	public void testCancelGenerating() {
	    System.out.println("***** Running the test: testCancelGenerating");
	    
	    // Create parameters using the InferenceParameters builder
	    InferenceParameters params = new InferenceParameters()
	        .setPrompt(prefix)
	        .setNPredict(nPredict)
	        .setStream(true);
	    
	    // Get the JSON string from the parameters
	    String requestJson = params.toString();
	    
	    // Call handleCompletions with streaming enabled
	    String streamInitResponse = model.handleCompletions(requestJson, true);
	    
	    try {
	        // Parse the stream initialization response
	        ObjectMapper mapper = new ObjectMapper();
	        JsonNode responseObj = mapper.readTree(streamInitResponse);
	        JsonNode taskIdsArray = responseObj.get("task_ids");
	        
	        // We should have at least one task ID
	        Assert.assertTrue(taskIdsArray.size() > 0);
	        int taskId = taskIdsArray.get(0).asInt();
	        
	        // Stream until we get 5 tokens then cancel
	        int generated = 0;
	        boolean isComplete = false;
	        
	        while (!isComplete && generated < nPredict) {
	            // Get the next chunk of streaming results
	            String chunkResponse = model.getNextStreamResult(taskId);
	            JsonNode chunkObj = mapper.readTree(chunkResponse);
	            
	            // Check if this is the final chunk
	            isComplete = chunkObj.get("is_final").asBoolean();
	            
	            // Extract and process the content
	            JsonNode resultObj = chunkObj.get("result");
	            if (resultObj.has("content")) {
	                String content = resultObj.get("content").asText();
	                if (!content.isEmpty()) {
	                    // Process the generated content if needed
	                    System.out.println("Generated chunk: " + content);
	                    generated++;
	                    
	                    // Cancel after 5 tokens are generated
	                    if (generated == 5) {
	                        model.cancelCompletion(taskId);
	                        break;
	                    }
	                }
	            }
	        }
	        
	        // Ensure exactly 5 tokens were generated before cancellation
	        Assert.assertEquals(5, generated);
	        
	        // Release the task to clean up resources (though it was already cancelled)
	        model.releaseTask(taskId);
	        
	    } catch (Exception e) {
	        Assert.fail("Failed during cancellation test: " + e.getMessage());
	    }
	}

	
	
	@Test
	public void testTokenization() {
		System.out.println("***** Running the test:  testTokenization");

		String prompt = "Hello, world!";
		String resultJson = model.handleTokenize(prompt, false, false);
		JsonNode root = JsonUtils.INSTANCE.jsonToNode(resultJson);

		JsonNode tokensNode = root.get("tokens");

		int[] tokens = new int[tokensNode.size()];
		for (int i = 0; i < tokensNode.size(); i++) {
		    tokens[i] = tokensNode.get(i).asInt();
		}
		
		Assert.assertEquals(4, tokens.length);
		
		String detokenized = JsonUtils.INSTANCE.jsonToNode(model.handleDetokenize(tokens)).get("content").asText();
		
		Assert.assertEquals(prompt, detokenized);
	}

	@Test
	public void testLogText() {
		List<LogMessage> messages = new ArrayList<>();
		LlamaModel.setLogger(LogFormat.TEXT, (level, msg) -> messages.add(new LogMessage(level, msg)));

		InferenceParameters params = new InferenceParameters().setPrompt(prefix)
				.setNPredict(nPredict)
				.setSeed(42);
		model.handleCompletions(params.toString(), false);

		Assert.assertFalse(messages.isEmpty());

		Pattern jsonPattern = Pattern.compile("^\\s*[\\[{].*[}\\]]\\s*$");
		for (LogMessage message : messages) {
			Assert.assertNotNull(message.level);
			Assert.assertFalse(jsonPattern.matcher(message.text).matches());
		}
	}

	@Test
	public void testLogJSON() {
		List<LogMessage> messages = new ArrayList<>();
		LlamaModel.setLogger(LogFormat.JSON, (level, msg) -> messages.add(new LogMessage(level, msg)));

		InferenceParameters params = new InferenceParameters().setPrompt(prefix)
				.setNPredict(nPredict)
				.setSeed(42);
		model.handleCompletions(params.toString(), false);

		Assert.assertFalse(messages.isEmpty());

		Pattern jsonPattern = Pattern.compile("^\\s*[\\[{].*[}\\]]\\s*$");
		for (LogMessage message : messages) {
			Assert.assertNotNull(message.level);
			System.out.println("messageText: " + message.text);
			Assert.assertTrue(jsonPattern.matcher(message.text).matches());
		}
	}

	@Test
	public void testLogStdout() {
		System.out.println("***** Running the test:  testLogStdout");
		
		// Unfortunately, `printf` can't be easily re-directed to Java. This test only works manually, thus.
		InferenceParameters params = new InferenceParameters().setPrompt(prefix)
				.setNPredict(nPredict)
				.setSeed(42);

		System.out.println("########## Log Text ##########");
		LlamaModel.setLogger(LogFormat.TEXT, null);
		model.handleCompletions(params.toString(), false);

		System.out.println("########## Log JSON ##########");
		LlamaModel.setLogger(LogFormat.JSON, null);
		model.handleCompletions(params.toString(), false);

		System.out.println("########## Log None ##########");
		LlamaModel.setLogger(LogFormat.TEXT, (level, msg) -> {});
		model.handleCompletions(params.toString(), false);

		System.out.println("##############################");
	}

	private String completeAndReadStdOut() {
		PrintStream stdOut = System.out;
		ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
		@SuppressWarnings("ImplicitDefaultCharsetUsage") PrintStream printStream = new PrintStream(outputStream);
		System.setOut(printStream);

		try {
			InferenceParameters params = new InferenceParameters().setPrompt(prefix)
					.setNPredict(nPredict)
					.setSeed(42);
			model.handleCompletions(params.toString(), false);
		} finally {
			System.out.flush();
			System.setOut(stdOut);
			printStream.close();
		}

		return outputStream.toString();
	}

	private List<String> splitLines(String text) {
		List<String> lines = new ArrayList<>();

		Scanner scanner = new Scanner(text);
		while (scanner.hasNextLine()) {
			String line = scanner.nextLine();
			lines.add(line);
		}
		scanner.close();

		return lines;
	}

	private static final class LogMessage {
		private final LogLevel level;
		private final String text;

		private LogMessage(LogLevel level, String text) {
			this.level = level;
			this.text = text;
		}
	}
	
	@Test
	public void testJsonSchemaToGrammar() {
		
		System.out.println("***** Running the test:  testJsonSchemaToGrammar");
		String schema = "{\n" +
                "    \"properties\": {\n" +
                "        \"a\": {\"type\": \"string\"},\n" +
                "        \"b\": {\"type\": \"string\"},\n" +
                "        \"c\": {\"type\": \"string\"}\n" +
                "    },\n" +
                "    \"additionalProperties\": false\n" +
                "}";
		
		String expectedGrammar = "a-kv ::= \"\\\"a\\\"\" space \":\" space string\n" +
                "a-rest ::= ( \",\" space b-kv )? b-rest\n" +
                "b-kv ::= \"\\\"b\\\"\" space \":\" space string\n" +
                "b-rest ::= ( \",\" space c-kv )?\n" +
                "c-kv ::= \"\\\"c\\\"\" space \":\" space string\n" +
                "char ::= [^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})\n" +
                "root ::= \"{\" space  (a-kv a-rest | b-kv b-rest | c-kv )? \"}\" space\n" +
                "space ::= | \" \" | \"\\n\"{1,2} [ \\t]{0,20}\n" +
                "string ::= \"\\\"\" char* \"\\\"\" space\n";
		
		byte[] actualGrammarBytes = LlamaModel.jsonSchemaToGrammarBytes(schema);
		String actualGrammar =  new String(actualGrammarBytes, StandardCharsets.UTF_8);
		Assert.assertEquals(expectedGrammar, actualGrammar);
	}
	
	@Test
	public void testTemplate() {
		System.out.println("***** Running the test:  testTemplate");
		List<Pair<String, String>> userMessages = new ArrayList<>();
        userMessages.add(new Pair<>("user", "What is the best book?"));
        userMessages.add(new Pair<>("assistant", "It depends on your interests. Do you like fiction or non-fiction?"));
        
		InferenceParameters params = new InferenceParameters()
				.setMessages("Book", userMessages)
				.setTemperature(0.95f)
				.setStopStrings("\"\"\"")
				.setNPredict(nPredict)
				.setSeed(42);
		
		Assert.assertEquals(model.applyTemplate(params.toString()), "{\n"
				+ "  \"prompt\": \"<|system|>Book<|end|><|user|>What is the best book?<|end|><|assistant|>It depends on your interests. Do you like fiction or non-fiction?<|end|><|assistant|>\"\n"
				+ "}");
	}
}
