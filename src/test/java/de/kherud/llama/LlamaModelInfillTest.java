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
import org.junit.Test;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import de.kherud.llama.args.LogFormat;

public class LlamaModelInfillTest {

	private static final String prefix = "def remove_non_ascii(s: str) -> str:\n    \"\"\" ";
	private static final String suffix = "\n    return result\n";
	private static final int nPredict = 10;

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {

		model = new LlamaModel(new ModelParameters()
				.setModel("models/stories260K-infill.gguf")
				.setCtxSize(4096)
				.enableJinja());
	}

	@AfterClass
	public static void tearDown() throws Exception {
		if (model != null) {
			model.close();
		}
	}

	
	
	@Test
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
	    System.out.println("***** Running the test: testGenerateGrammar");
	    
	    InferenceParameters params = new InferenceParameters()
	        .setPrompt(prefix)
	        .setGrammar("root ::= (\"a\" | \"b\")+")
	        .setNPredict(nPredict);
	    
	    // Try up to 3 times to handle potential transient issues
	    String output = null;
	    int attempts = 0;
	    while (attempts < 3) {
	        try {
	            output = model.handleCompletions(params.toString(), false);
	            break; // Success, exit loop
	        } catch (Exception e) {
	            attempts++;
	            System.err.println("Grammar generation attempt " + attempts + " failed: " + e.getMessage());
	            if (attempts >= 3) {
	                throw e; // Re-throw after max attempts
	            }
	            // Wait briefly before retrying
	            try {
	                Thread.sleep(500);
	            } catch (InterruptedException ie) {
	                Thread.currentThread().interrupt();
	            }
	        }
	    }
	    
	    JsonNode jsonNode = JsonUtils.INSTANCE.jsonToNode(output);
	    JsonNode resultNode = jsonNode.get("result");
	    String content = resultNode.get("content").asText();
	    Assert.assertTrue(content.matches("[ab]+"));
	    int generated = model.encode(content).length;

	    Assert.assertTrue("generated should be between 0 and 11 but is " + generated, 
	                      generated > 0 && generated <= nPredict + 1);
	}

	@Test
	public void testCompleteInfillCustom() {
		System.out.println("***** Running the test:  testCompleteInfillCustom");
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters().setPrompt(" ")
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
}
