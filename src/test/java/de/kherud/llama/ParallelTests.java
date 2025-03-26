package de.kherud.llama;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Ignore;

import com.fasterxml.jackson.databind.JsonNode;

public class ParallelTests {

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {
		model = new LlamaModel(new ModelParameters()
				.setModel("models/Phi-4-mini-instruct-Q2_K.gguf")
				.setGpuLayers(43)
				.enableLogTimestamps()
				.enableLogPrefix()
				.enableJinja()
				.slotSavePath("models"));
				;
	}

	@AfterClass
	public static void tearDown() throws Exception {
		if (model != null) {
			model.close();
		}
	}
	
	@Ignore
	public void testParallelInference() {
	    System.out.println("***** Running the test: testParallelInference");
	    
	    // 1. Configure parallel inference with specific parameters
	    String config = "{\"slot_prompt_similarity\": 0.8, \"batch_mode\": true, \"defer_when_full\": true}";
	    boolean configSuccess = model.configureParallelInference(config);
	    Assert.assertTrue("Failed to configure parallel inference", configSuccess);
	    
	    // 2. Create multiple inference tasks with different prompts
	    List<String> prompts = Arrays.asList(
	        "The quick brown fox",
	        "Once upon a time",
	        "In a galaxy far far away",
	        "Four score and seven years ago"
	    );
	    
	    // 3. Execute tasks concurrently and measure response times
	    List<Callable<Long>> tasks = new ArrayList<>();
	    List<Future<Long>> futures = new ArrayList<>();
	    ExecutorService executor = Executors.newFixedThreadPool(prompts.size());
	    
	    for (String prompt : prompts) {
	        tasks.add(() -> {
	            long startTime = System.currentTimeMillis();
	            
	            InferenceParameters params = new InferenceParameters()
	                .setPrompt(prompt)
	                .setNPredict(10);
	            
	            // Run completion and wait for result
	            String result = model.handleCompletions(params.toString(), false);
	            
	            // Calculate execution time
	            return System.currentTimeMillis() - startTime;
	        });
	    }
	    
	    try {
	        // Submit all tasks
	        futures = executor.invokeAll(tasks);
	        
	        // Collect execution times
	        List<Long> executionTimes = new ArrayList<>();
	        for (Future<Long> future : futures) {
	            executionTimes.add(future.get());
	        }
	        
	        // 4. Verify parallel execution happened
	        // Calculate total and average execution time
	        long totalTime = executionTimes.stream().mapToLong(Long::longValue).sum();
	        long avgTime = totalTime / executionTimes.size();
	        
	        System.out.println("Individual execution times: " + executionTimes);
	        System.out.println("Total execution time: " + totalTime + "ms");
	        System.out.println("Average execution time: " + avgTime + "ms");
	        
	        // 5. Validate the results - if parallel inference is working correctly:
	        // - Total time should be less than sum of individual times if run sequentially
	        // - Individual times should be reasonable given the prompt length
	        
	        // Here we're assuming that if parallel inference is working correctly,
	        // the total time should be significantly less than 4x the average time
	        // This is a heuristic and might need adjustment based on your hardware
	        Assert.assertTrue("Parallel inference doesn't appear to be working efficiently",
	                          totalTime < (avgTime * executionTimes.size() * 0.8));
	        
	    } catch (InterruptedException | ExecutionException e) {
	        Assert.fail("Error during parallel execution: " + e.getMessage());
	    } finally {
	        executor.shutdown();
	    }
	    
	    // 6. Test slot reuse with similar prompts
	    String similarPrompt1 = "The quick brown fox jumps over the lazy dog";
	    String similarPrompt2 = "The quick brown fox jumps over the fence";
	    
	    try {
	        // First run with one prompt
	        InferenceParameters params1 = new InferenceParameters()
	            .setPrompt(similarPrompt1)
	            .setNPredict(5);
	        
	        String result1 = model.handleCompletions(params1.toString(), false);
	        
	        // Then quickly run with a similar prompt - should reuse the slot
	        InferenceParameters params2 = new InferenceParameters()
	            .setPrompt(similarPrompt2)
	            .setNPredict(5);
	        
	        String result2 = model.handleCompletions(params2.toString(), false);
	        
	        // Both operations should succeed
	        JsonNode jsonNode1 = JsonUtils.INSTANCE.jsonToNode(result1);
	        JsonNode jsonNode2 = JsonUtils.INSTANCE.jsonToNode(result2);
	        
	        Assert.assertTrue(jsonNode1.has("result"));
	        Assert.assertTrue(jsonNode2.has("result"));
	        
	        // We can't directly verify slot reuse from the API, but we can check
	        // that both operations completed successfully
	        System.out.println("Successfully processed similar prompts, likely with slot reuse");
	        
	    } catch (Exception e) {
	        Assert.fail("Error during slot reuse test: " + e.getMessage());
	    }
	}
}
