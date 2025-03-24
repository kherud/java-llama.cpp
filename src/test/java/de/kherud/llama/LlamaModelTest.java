package de.kherud.llama;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
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

import de.kherud.llama.args.LogFormat;

public class LlamaModelTest {

	private static final String prefix = "def remove_non_ascii(s: str) -> str:\n    \"\"\" ";
	private static final String suffix = "\n    return result\n";
	private static final int nPredict = 10;

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {

		model = new LlamaModel(new ModelParameters()
				.setModel("models/EXAONE-Deep-2.4B-Q4_K_M.gguf")
				.setGpuLayers(43)
				.enableJinja()
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
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testGenerateAnswer() {
		System.out.println("***** Running the test:  testGenerateAnswer");
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters(prefix)
				.setTemperature(0.95f)
				.setStopStrings("\"\"\"")
				.setNPredict(nPredict)
				.setTokenIdBias(logitBias);

		int generated = 0;
		for (LlamaOutput ignored : model.generate(params)) {
			generated++;
		}
		// todo: currently, after generating nPredict tokens, there is an additional empty output
		Assert.assertTrue(generated > 0 && generated <= nPredict + 1);
	}

	@Test
	public void testGenerateInfill() {
		System.out.println("***** Running the test:  testGenerateInfill");
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters("")
				.setInputPrefix(prefix)
				.setInputSuffix(suffix )
				.setTemperature(0.95f)
				.setStopStrings("\"\"\"")
				.setNPredict(nPredict)
				.setTokenIdBias(logitBias)
				.setSeed(42);

		int generated = 0;
		for (LlamaOutput ignored : model.generate(params)) {
			generated++;
		}
		Assert.assertTrue(generated > 0 && generated <= nPredict + 1);
	}

	@Test
	public void testGenerateGrammar() {
		System.out.println("***** Running the test:  testGenerateGrammar");
		InferenceParameters params = new InferenceParameters("code")
				.setGrammar("root ::= (\"a\" | \"b\")+")
				.setNPredict(nPredict);
		List<Pair<String, String>> userMessages = new ArrayList<>();
		userMessages.add(new Pair<>("user", "Does not matter what I say, does it?"));
		
		String output = model.handleCompletions(params.toString(), false, 0);
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
		InferenceParameters params = new InferenceParameters(prefix)
				.setTemperature(0.95f)
				.setStopStrings("\"\"\"")
				.setNPredict(nPredict)
				.setTokenIdBias(logitBias)
				.setSeed(42);

		String output = model.complete(params);
		Assert.assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteInfillCustom() {
		System.out.println("***** Running the test:  testCompleteInfillCustom");
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters("code ")
				.setInputPrefix(prefix)
				.setInputSuffix(suffix)
				.setTemperature(0.95f)
				.setStopStrings("\"\"\"")
				.setNPredict(nPredict)
				.setTokenIdBias(logitBias)
				.setSeed(42);

		String output = model.complete(params);
		Assert.assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteGrammar() {
		System.out.println("***** Running the test:  testCompleteGrammar");
		InferenceParameters params = new InferenceParameters("code ")
				.setGrammar("root ::= (\"a\" | \"b\")+")
				.setTemperature(0.6f)
				.setTopP(0.95f)
				.setNPredict(nPredict);
		String output = model.complete(params);
		Assert.assertTrue(output + " doesn't match [ab]+", output.matches("[ab]+"));
		int generated = model.encode(output).length;
		Assert.assertTrue("generated count is: " + generated,  generated > 0 && generated <= nPredict + 1);
		
	}

	@Test
	public void testCancelGenerating() {
		
		System.out.println("***** Running the test:  testCancelGenerating");

		InferenceParameters params = new InferenceParameters(prefix).setNPredict(nPredict);

		int generated = 0;
		LlamaIterator iterator = model.generate(params).iterator();
		while (iterator.hasNext()) {
			iterator.next();
			generated++;
			if (generated == 5) {
				iterator.cancel();
			}
		}
		Assert.assertEquals(5, generated);
	}

	
	
	@Test
	public void testTokenization() {
		System.out.println("***** Running the test:  testTokenization");

		String prompt = "Hello, world!";
		int[] encoded = model.encode(prompt);
		String decoded = model.decode(encoded);
		// the llama tokenizer adds a space before the prompt
		Assert.assertEquals(prompt, decoded);
	}

	@Ignore
	public void testLogText() {
		List<LogMessage> messages = new ArrayList<>();
		LlamaModel.setLogger(LogFormat.TEXT, (level, msg) -> messages.add(new LogMessage(level, msg)));

		InferenceParameters params = new InferenceParameters(prefix)
				.setNPredict(nPredict)
				.setSeed(42);
		model.complete(params);

		Assert.assertFalse(messages.isEmpty());

		Pattern jsonPattern = Pattern.compile("^\\s*[\\[{].*[}\\]]\\s*$");
		for (LogMessage message : messages) {
			Assert.assertNotNull(message.level);
			Assert.assertFalse(jsonPattern.matcher(message.text).matches());
		}
	}

	@Ignore
	public void testLogJSON() {
		List<LogMessage> messages = new ArrayList<>();
		LlamaModel.setLogger(LogFormat.JSON, (level, msg) -> messages.add(new LogMessage(level, msg)));

		InferenceParameters params = new InferenceParameters(prefix)
				.setNPredict(nPredict)
				.setSeed(42);
		model.complete(params);

		Assert.assertFalse(messages.isEmpty());

		Pattern jsonPattern = Pattern.compile("^\\s*[\\[{].*[}\\]]\\s*$");
		for (LogMessage message : messages) {
			Assert.assertNotNull(message.level);
			Assert.assertTrue(jsonPattern.matcher(message.text).matches());
		}
	}

	@Test
	public void testLogStdout() {
		System.out.println("***** Running the test:  testLogStdout");
		
		// Unfortunately, `printf` can't be easily re-directed to Java. This test only works manually, thus.
		InferenceParameters params = new InferenceParameters(prefix)
				.setNPredict(nPredict)
				.setSeed(42);

		System.out.println("########## Log Text ##########");
		LlamaModel.setLogger(LogFormat.TEXT, null);
		model.complete(params);

		System.out.println("########## Log JSON ##########");
		LlamaModel.setLogger(LogFormat.JSON, null);
		model.complete(params);

		System.out.println("########## Log None ##########");
		LlamaModel.setLogger(LogFormat.TEXT, (level, msg) -> {});
		model.complete(params);

		System.out.println("##############################");
	}

	private String completeAndReadStdOut() {
		PrintStream stdOut = System.out;
		ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
		@SuppressWarnings("ImplicitDefaultCharsetUsage") PrintStream printStream = new PrintStream(outputStream);
		System.setOut(printStream);

		try {
			InferenceParameters params = new InferenceParameters(prefix)
					.setNPredict(nPredict)
					.setSeed(42);
			model.complete(params);
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
		
		String actualGrammar = LlamaModel.jsonSchemaToGrammar(schema);
		Assert.assertEquals(expectedGrammar, actualGrammar);
	}
	
	@Test
	public void testTemplate() {
		System.out.println("***** Running the test:  testTemplate");
		List<Pair<String, String>> userMessages = new ArrayList<>();
        userMessages.add(new Pair<>("user", "What is the best book?"));
        userMessages.add(new Pair<>("assistant", "It depends on your interests. Do you like fiction or non-fiction?"));
        
		InferenceParameters params = new InferenceParameters("A book recommendation system.")
				.setMessages("Book", userMessages)
				.setTemperature(0.95f)
				.setStopStrings("\"\"\"")
				.setNPredict(nPredict)
				.setSeed(42);
		Assert.assertEquals(model.applyTemplate(params), "[|system|]Book[|endofturn|]\n"
				+ "[|user|]What is the best book?\n"
				+ "[|assistant|]It depends on your interests. Do you like fiction or non-fiction?[|endofturn|]\n"
				+ "[|assistant|]<thought>\n");
	}
}
