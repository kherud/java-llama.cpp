package de.kherud.llama;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

public class JsonUtils {
	private final ObjectMapper mapper = new ObjectMapper();
	public static final JsonUtils INSTANCE = new JsonUtils();
	
	private JsonUtils() {
		
	}
	
	public String nodeToJson(JsonNode node) {
	    try {
	        return mapper.writeValueAsString(node);
	    } catch (Exception e) {
	        throw new RuntimeException("Failed to convert JsonNode to JSON string", e);
	    }
	}

	public JsonNode jsonToNode(String json) {
	    try {
	        return mapper.readTree(json);
	    } catch (Exception e) {
	        throw new RuntimeException("Failed to parse JSON: " + json, e);
	    }
	}

}
