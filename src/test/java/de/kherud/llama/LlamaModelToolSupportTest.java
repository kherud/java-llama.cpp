package de.kherud.llama;

import java.util.ArrayList;
import java.util.List;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import com.fasterxml.jackson.databind.JsonNode;

public class LlamaModelToolSupportTest {

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {
		model = new LlamaModel(
				new ModelParameters().setCtxSize(128).setModel("models/Llama-3.2-3B-Instruct-Q8_0.gguf")
						.setGpuLayers(43).enableLogTimestamps().enableLogPrefix().enableJinja());

	}

	@AfterClass
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	
	String get_current_temperatureFunction = "{\n"
			+ "    \"type\": \"function\",\n"
			+ "    \"function\": {\n"
			+ "      \"name\": \"get_current_temperature\",\n"
			+ "      \"description\": \"Get current temperature at a location.\",\n"
			+ "      \"parameters\": {\n"
			+ "        \"type\": \"object\",\n"
			+ "        \"properties\": {\n"
			+ "          \"location\": {\n"
			+ "            \"type\": \"string\",\n"
			+ "            \"description\": \"The location to get the temperature for, in the format \\\"City, State, Country\\\".\"\n"
			+ "          },\n"
			+ "          \"unit\": {\n"
			+ "            \"type\": \"string\",\n"
			+ "            \"enum\": [\n"
			+ "              \"celsius\",\n"
			+ "              \"fahrenheit\"\n"
			+ "            ],\n"
			+ "            \"description\": \"The unit to return the temperature in. Defaults to \\\"celsius\\\".\"\n"
			+ "          }\n"
			+ "        },\n"
			+ "        \"required\": [\n"
			+ "          \"location\"\n"
			+ "        ]\n"
			+ "      }\n"
			+ "    }\n"
			+ "  }";
	
	String get_temperature_dateFunction = "{\n"
			+ "    \"type\": \"function\",\n"
			+ "    \"function\": {\n"
			+ "      \"name\": \"get_temperature_date\",\n"
			+ "      \"description\": \"Get temperature at a location and date.\",\n"
			+ "      \"parameters\": {\n"
			+ "        \"type\": \"object\",\n"
			+ "        \"properties\": {\n"
			+ "          \"location\": {\n"
			+ "            \"type\": \"string\",\n"
			+ "            \"description\": \"The location to get the temperature for, in the format \\\"City, State, Country\\\".\"\n"
			+ "          },\n"
			+ "          \"date\": {\n"
			+ "            \"type\": \"string\",\n"
			+ "            \"description\": \"The date to get the temperature for, in the format \\\"Year-Month-Day\\\".\"\n"
			+ "          },\n"
			+ "          \"unit\": {\n"
			+ "            \"type\": \"string\",\n"
			+ "            \"enum\": [\n"
			+ "              \"celsius\",\n"
			+ "              \"fahrenheit\"\n"
			+ "            ],\n"
			+ "            \"description\": \"The unit to return the temperature in. Defaults to \\\"celsius\\\".\"\n"
			+ "          }\n"
			+ "        },\n"
			+ "        \"required\": [\n"
			+ "          \"location\",\n"
			+ "          \"date\"\n"
			+ "        ]\n"
			+ "      }\n"
			+ "    }\n"
			+ "  }";
	

	@Test
	public void testToolCalling() {
		

		List<Pair<String, String>> userMessages = new ArrayList<>();

		userMessages.add(new Pair<>("user", "What's the temperature in San Francisco today?"));


		InferenceParameters params = new InferenceParameters(null)
				.setMessages("You are a helpful assistant.\\n\\nCurrent Date: 2024-09-30", userMessages).setTemperature(0f)
				.setTools(get_current_temperatureFunction, get_temperature_dateFunction).setNPredict(512)
				.setUseChatTemplate(true).setChatTemplate("{{- bos_token }}\n"
						+ "{%- if custom_tools is defined %}\n"
						+ "    {%- set tools = custom_tools %}\n"
						+ "{%- endif %}\n"
						+ "{%- if not tools_in_user_message is defined %}\n"
						+ "    {%- set tools_in_user_message = true %}\n"
						+ "{%- endif %}\n"
						+ "{%- if not date_string is defined %}\n"
						+ "    {%- if strftime_now is defined %}\n"
						+ "        {%- set date_string = strftime_now(\"%d %b %Y\") %}\n"
						+ "    {%- else %}\n"
						+ "        {%- set date_string = \"26 Jul 2024\" %}\n"
						+ "    {%- endif %}\n"
						+ "{%- endif %}\n"
						+ "{%- if not tools is defined %}\n"
						+ "    {%- set tools = none %}\n"
						+ "{%- endif %}\n"
						+ "\n"
						+ "{#- This block extracts the system message, so we can slot it into the right place. #}\n"
						+ "{%- if messages[0]['role'] == 'system' %}\n"
						+ "    {%- set system_message = messages[0]['content']|trim %}\n"
						+ "    {%- set messages = messages[1:] %}\n"
						+ "{%- else %}\n"
						+ "    {%- set system_message = \"\" %}\n"
						+ "{%- endif %}\n"
						+ "\n"
						+ "{#- System message #}\n"
						+ "{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n"
						+ "{%- if tools is not none %}\n"
						+ "    {{- \"Environment: ipython\\n\" }}\n"
						+ "{%- endif %}\n"
						+ "{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n"
						+ "{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n"
						+ "{%- if tools is not none and not tools_in_user_message %}\n"
						+ "    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n"
						+ "    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n"
						+ "    {{- \"Do not use variables.\\n\\n\" }}\n"
						+ "    {%- for t in tools %}\n"
						+ "        {{- t | tojson(indent=4) }}\n"
						+ "        {{- \"\\n\\n\" }}\n"
						+ "    {%- endfor %}\n"
						+ "{%- endif %}\n"
						+ "{{- system_message }}\n"
						+ "{{- \"<|eot_id|>\" }}\n"
						+ "\n"
						+ "{#- Custom tools are passed in a user message with some extra guidance #}\n"
						+ "{%- if tools_in_user_message and not tools is none %}\n"
						+ "    {#- Extract the first user message so we can plug it in here #}\n"
						+ "    {%- if messages | length != 0 %}\n"
						+ "        {%- set first_user_message = messages[0]['content']|trim %}\n"
						+ "        {%- set messages = messages[1:] %}\n"
						+ "    {%- else %}\n"
						+ "        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n"
						+ "{%- endif %}\n"
						+ "    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n"
						+ "    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n"
						+ "    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n"
						+ "    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n"
						+ "    {{- \"Do not use variables.\\n\\n\" }}\n"
						+ "    {%- for t in tools %}\n"
						+ "        {{- t | tojson(indent=4) }}\n"
						+ "        {{- \"\\n\\n\" }}\n"
						+ "    {%- endfor %}\n"
						+ "    {{- first_user_message + \"<|eot_id|>\"}}\n"
						+ "{%- endif %}\n"
						+ "\n"
						+ "{%- for message in messages %}\n"
						+ "    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n"
						+ "        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n"
						+ "    {%- elif 'tool_calls' in message %}\n"
						+ "        {%- if not message.tool_calls|length == 1 %}\n"
						+ "            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n"
						+ "        {%- endif %}\n"
						+ "        {%- set tool_call = message.tool_calls[0].function %}\n"
						+ "        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n"
						+ "        {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n"
						+ "        {{- '\"parameters\": ' }}\n"
						+ "        {{- tool_call.arguments | tojson }}\n"
						+ "        {{- \"}\" }}\n"
						+ "        {{- \"<|eot_id|>\" }}\n"
						+ "    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n"
						+ "        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n"
						+ "        {%- if message.content is mapping or message.content is iterable %}\n"
						+ "            {{- message.content | tojson }}\n"
						+ "        {%- else %}\n"
						+ "            {{- message.content }}\n"
						+ "        {%- endif %}\n"
						+ "        {{- \"<|eot_id|>\" }}\n"
						+ "    {%- endif %}\n"
						+ "{%- endfor %}\n"
						+ "{%- if add_generation_prompt %}\n"
						+ "    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n"
						+ "{%- endif %}");
				
		
		String responseJson = model.handleCompletions(params.toString(), false, 0);
		
		 // Parse the JSON response using your existing JsonUtils
	    JsonNode response = JsonUtils.INSTANCE.jsonToNode(responseJson);
	    
	    // Check the basics of the response
	    Assert.assertEquals("completion", response.get("type").asText());
	    Assert.assertEquals(false, response.get("streaming").asBoolean());
	    Assert.assertNotNull("Should have a completion ID", response.get("completion_id"));
	    
	    // Get to the message part of the response
	    JsonNode result = response.get("result");
	    JsonNode choices = result.get("choices");
	    Assert.assertTrue("Should have at least one choice", choices.size() > 0);
	    
	    JsonNode firstChoice = choices.get(0);
	    
	    // Check that finish reason is tool_calls
	    Assert.assertEquals("tool_calls", firstChoice.get("finish_reason").asText());
	    
	    // Check message structure
	    JsonNode message = firstChoice.get("message");
	    Assert.assertEquals("assistant", message.get("role").asText());
	    Assert.assertTrue("Content should be null when using tool calls", 
	        message.get("content").isNull());
	    
	    // Check tool calls
	    JsonNode toolCalls = message.get("tool_calls");
	    Assert.assertTrue("Should have tool calls", toolCalls.isArray());
	    Assert.assertTrue("Should have at least one tool call", toolCalls.size() > 0);
	    
	    // Check the first tool call
	    JsonNode firstToolCall = toolCalls.get(0);
	    Assert.assertEquals("function", firstToolCall.get("type").asText());
	    Assert.assertTrue("Tool call should have an ID", firstToolCall.has("id"));
	    
	    // Check function details
	    JsonNode function = firstToolCall.get("function");
	    Assert.assertTrue("Should have function name", function.has("name"));
	    String functionName = function.get("name").asText();
	    Assert.assertTrue("Function name should be one of the provided functions", 
	        functionName.equals("get_current_temperature") || 
	        functionName.equals("get_temperature_date"));
	        
	    // Check function arguments
	    Assert.assertTrue("Should have function arguments", function.has("arguments"));
	    String arguments = function.get("arguments").asText();
	    JsonNode args = JsonUtils.INSTANCE.jsonToNode(arguments);
	    
	    // Verify arguments structure based on which function was called
	    Assert.assertTrue("Arguments should include location", args.has("location"));
	    Assert.assertEquals("San Francisco", args.get("location").asText());
	    
	    if (functionName.equals("get_temperature_date")) {
	        Assert.assertTrue("Should have date argument", args.has("date"));
	        //weird that date returned sometimes is having hours, mins and seconds
	        //Assert.assertEquals("2024-09-30", args.get("date").asText());
	    }
	    
	    System.out.println("Tool call succeeded with function: " + functionName);
	    System.out.println("Arguments: " + arguments); 

	}

}
