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
				.setModel("models/EXAONE-Deep-2.4B-Q4_K_M.gguf")
				.setGpuLayers(43)
				.enableLogTimestamps()
				.enableLogPrefix()
				.enableJinja()
				.enableEmbedding()
				.setChatTemplate("{% for message in messages %}{% if "
						+ "loop.first and message['role'] != 'system' %}"
						+ "{{ '[|system|][|endofturn|]\\n' }}{% endif %}"
						+ "{% set content = message['content'] %}"
						+ "{% if '</thought>' in content %}{% "
						+ "set content = content.split('</thought>')"
						+ "[-1].lstrip('\\\\n') %}{% endif %}"
						+ "{{ '[|' + message['role'] + '|]' + content }}"
						+ "{% if not message['role'] == 'user' %}"
						+ "{{ '[|endofturn|]' }}{% endif %}{% if not loop.last %}"
						+ "{{ '\\n' }}{% endif %}{% endfor %}"
						+ "{% if add_generation_prompt %}"
						+ "{{ '\\n[|assistant|]<thought>\\n' }}"
						+ "{% endif %}"));
	}

	@AfterClass
	public static void tearDown() throws Exception {
		if (model != null) {
			model.close();
		}
	}
	
	@Test
	public void testEmbedding() {
	    // Create the request in JSON format
	    String request = "{\"content\": \"You are an AI Assistant\"}";
	    
	    // Call the handleEmbeddings method
	    String response = model.handleEmbeddings(request, false);
	    
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
	        Assert.assertEquals(2560, embedding.length);
	    } catch (Exception e) {
	        Assert.fail("Failed to parse embedding response: " + e.getMessage());
	    }
	}
}
