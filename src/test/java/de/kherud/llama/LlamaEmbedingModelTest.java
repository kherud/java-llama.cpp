package de.kherud.llama;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

public class LlamaEmbedingModelTest {

	private static LlamaModel model;
	
	
	@BeforeClass
	public static void setup() {

		model = new LlamaModel(new ModelParameters()
				.setModel("models/ggml-model-f16.gguf")
				.setCtxSize(512)
				.setBatchSize(128)
				.setUbatchSize(128)
				.setDefragThold(0.1f)
				.setParallel(2)
				.enableEmbedding());
	}

	@AfterClass
	public static void tearDown() throws Exception {
		if (model != null) {
			model.close();
		}
	}
	
	@Test
	public void testEmbedding() {

		model.handleKVCacheAction(LlamaModel.KVCACHE_ACTION_CLEAR, 0, null);
	    // Create the request in JSON format
	    String request = "{\"content\": \"AI Assistant\"}";
	    
	    // Call the handleEmbeddings method
	    String response = model.handleEmbeddings(request, true);
	    
	    // Parse the JSON response
	    try {
	        // You'll need a JSON parser - this example uses Jackson
	        ObjectMapper mapper = new ObjectMapper();
	        JsonNode rootNode = mapper.readTree(response);
	        
	        // For non-OAI format, the embedding is in the first result's "embedding" field
	        JsonNode embeddingNode = rootNode.get(0).get("embedding").get(0);
	        
	        // Convert embedding from JSON array to float array
	        float[] embedding = new float[embeddingNode.size()];
	        for (int i = 0; i < embedding.length; i++) {
	            embedding[i] = (float) embeddingNode.get(i).asDouble();
	        }
	        
	        // Verify the embedding dimensions
	        Assert.assertEquals(384, embedding.length);
	    } catch (Exception e) {
	        Assert.fail("Failed to parse embedding response: " + e.getMessage());
	    }
	}
}
