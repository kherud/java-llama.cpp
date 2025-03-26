package de.kherud.llama;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import com.fasterxml.jackson.databind.JsonNode;

public class KVCacheTests {

	private static LlamaModel model;
	private final String prefix = "test for KVCache";

	@BeforeClass
	public static void setup() {
		model = new LlamaModel(new ModelParameters()
				.setModel("models/Phi-4-mini-instruct-Q2_K.gguf")
				.setGpuLayers(43)
				.enableLogTimestamps()
				.enableLogPrefix()
				.enableJinja()
				.setCtxSize(4096)
				.slotSavePath("models"));
				;
	}

	@AfterClass
	public static void tearDown() throws Exception {
		if (model != null) {
			model.close();
		}
	}
	
	/**
	 * Test getting KV cache information
	 */
	@Test
	public void testKVCacheInfo() {
	    System.out.println("***** Running the test: testKVCacheInfo");
	    
	    // First generate some text to populate the KV cache
	    InferenceParameters params = new InferenceParameters()
	        .setPrompt(prefix)
	        .setNPredict(5);
	    
	    model.handleCompletions(params.toString(), false);
	    
	    // Now get KV cache info for slot 0
	    String infoResult = model.handleKVCacheAction(LlamaModel.KVCACHE_ACTION_INFO, 0, null);
	    
	    // Parse the result
	    JsonNode infoNode = JsonUtils.INSTANCE.jsonToNode(infoResult);
	    
	    // Verify the result contains expected fields
	    Assert.assertEquals("info", infoNode.get("action").asText());
	    Assert.assertEquals(0, infoNode.get("slot_id").asInt());
	    Assert.assertTrue(infoNode.has("kv_cache_tokens"));
	    Assert.assertTrue(infoNode.has("kv_cache_used_cells"));
	    Assert.assertTrue(infoNode.get("success").asBoolean());
	    
	    // Verify KV cache has tokens (since we generated text)
	    Assert.assertTrue(infoNode.get("kv_cache_tokens").asInt() > 0);
	}

	/**
	 * Test clearing KV cache
	 */
	@Test
	public void testKVCacheClear() {
	    System.out.println("***** Running the test: testKVCacheClear");
	    
	    // First generate some text to populate the KV cache
	    InferenceParameters params = new InferenceParameters()
	        .setPrompt(prefix)
	        .setNPredict(5);
	    
	    model.handleCompletions(params.toString(), false);
	    
	    // Get initial KV cache info
	    String initialInfo = model.handleKVCacheAction(LlamaModel.KVCACHE_ACTION_INFO, 0, null);
	    JsonNode initialNode = JsonUtils.INSTANCE.jsonToNode(initialInfo);
	    int initialTokens = initialNode.get("kv_cache_tokens").asInt();
	    
	    // Verify we have tokens in the cache
	    Assert.assertTrue(initialTokens > 0);
	    
	    // Now clear the KV cache
	    String clearResult = model.handleKVCacheAction(LlamaModel.KVCACHE_ACTION_CLEAR, 0, null);
	    JsonNode clearNode = JsonUtils.INSTANCE.jsonToNode(clearResult);
	    
	    // Verify the clear operation was successful
	    Assert.assertEquals("clear", clearNode.get("action").asText());
	    Assert.assertEquals(0, clearNode.get("slot_id").asInt());
	    Assert.assertTrue(clearNode.get("success").asBoolean());
	    
	    // Get KV cache info after clearing
	    String afterInfo = model.handleKVCacheAction(LlamaModel.KVCACHE_ACTION_INFO, 0, null);
	    JsonNode afterNode = JsonUtils.INSTANCE.jsonToNode(afterInfo);
	    
	    // Verify KV cache has been cleared (should have 0 tokens or fewer tokens than before)
	    int afterTokens = afterNode.get("kv_cache_tokens").asInt();
	    Assert.assertTrue(afterTokens < initialTokens);
	}

	/**
	 * Test saving and loading KV cache
	 */
	@Test
	public void testKVCacheSaveLoad() {
	    System.out.println("***** Running the test: testKVCacheSaveLoad");
	    
	 
	    // First generate some text to populate the KV cache
	    InferenceParameters params = new InferenceParameters()
	        .setPrompt("This is a unique prompt to test KV cache persistence")
	        .setNPredict(5);
	    
	    String firstResult = model.handleCompletions(params.toString(), false);
	    JsonNode firstNode = JsonUtils.INSTANCE.jsonToNode(firstResult);
	    String firstContent = firstNode.get("result").get("content").asText();
	    
	    // Save the KV cache state
	    String filename = "test_kvcache_" + System.currentTimeMillis() + ".bin";
	    String saveResult = model.handleKVCacheAction(LlamaModel.KVCACHE_ACTION_SAVE, 0, filename);
	    JsonNode saveNode = JsonUtils.INSTANCE.jsonToNode(saveResult);
	    
	    // Verify save was successful
	    Assert.assertTrue(saveNode.get("success").asBoolean());
	    
	    // Clear the KV cache
	    model.handleKVCacheAction(LlamaModel.KVCACHE_ACTION_CLEAR, 0, null);
	    
	    // Generate new text with a different prompt to change the KV cache
	    InferenceParameters diffParams = new InferenceParameters()
	        .setPrompt("A completely different prompt")
	        .setNPredict(5);
	    
	    model.handleCompletions(diffParams.toString(), false);
	    
	    // Now restore the saved KV cache
	    String loadResult = model.handleKVCacheAction(LlamaModel.KVCACHE_ACTION_LOAD, 0, filename);
	    JsonNode loadNode = JsonUtils.INSTANCE.jsonToNode(loadResult);
	    
	    // Verify load was successful
	    Assert.assertTrue(loadNode.get("success").asBoolean());
	    
	    // Generate text with the same prompt as before
	    // With the restored KV cache, it should continue from where it left off
	    String secondResult = model.handleCompletions(params.toString(), false);
	    JsonNode secondNode = JsonUtils.INSTANCE.jsonToNode(secondResult);
	    String secondContent = secondNode.get("result").get("content").asText();
	    
	    // The second result should not be identical to the first result
	    // as we're continuing from the previous context
	    Assert.assertNotEquals(firstContent, secondContent);
	    
	    // Cleanup: try to delete the test file
	    try {
	        new java.io.File(filename).delete();
	    } catch (Exception e) {
	        System.err.println("Could not delete test file: " + e.getMessage());
	    }
	}
}
