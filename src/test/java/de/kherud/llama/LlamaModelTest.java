package de.kherud.llama;

import java.util.HashMap;
import java.util.Map;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

@Ignore
public class LlamaModelTest {

	private static final String prefix = "def remove_non_ascii(s: str) -> str:\n    \"\"\" ";
	private static final String suffix = "\n    return result\n";
	private static String logOutput = "";

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {
		LlamaModel.setLogger((level, msg) -> logOutput += msg);
		ModelParameters params = new ModelParameters.Builder()
				.setNGpuLayers(43)
				.setEmbedding(true)
				.build();
		model = new LlamaModel("/run/media/konstantin/Seagate/models/codellama-13b.q5_k_m.gguf", params);
	}

	@AfterClass
	public static void tearDown() {
		model.close();
	}

	@Test
	public void testLogOutput() {
		Assert.assertFalse(logOutput.isEmpty());
	}

	@Test
	public void testGenerateAnswerDefault() {
		int generated = 0;
		for (LlamaModel.Output ignored : model.generate(prefix)) {
			generated++;
		}
		Assert.assertTrue(generated > 0);
	}

	@Test
	public void testGenerateAnswerCustom() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters.Builder()
				.setTemperature(0.95f)
				.setAntiPrompt("\"\"\"")
				.setLogitBias(logitBias)
				.setSeed(42)
				.build();

		int generated = 0;
		for (LlamaModel.Output ignored : model.generate(prefix, params)) {
			generated++;
		}
		Assert.assertTrue(generated > 0);
	}

	@Test
	public void testGenerateInfillDefault() {
		int generated = 0;
		for (LlamaModel.Output ignored : model.generate(prefix, suffix)) {
			generated++;
		}
		Assert.assertTrue(generated > 0);
	}

	@Test
	public void testGenerateInfillCustom() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters.Builder()
				.setTemperature(0.95f)
				.setAntiPrompt("\"\"\"")
				.setLogitBias(logitBias)
				.setSeed(42)
				.build();

		int generated = 0;
		for (LlamaModel.Output ignored : model.generate(prefix, suffix, params)) {
			generated++;
		}
		Assert.assertTrue(generated > 0);
	}

	@Test
	public void testGenerateGrammar() {
		InferenceParameters params = new InferenceParameters.Builder()
				.setGrammar("root ::= (\"a\" | \"b\")+")
				.setNPredict(42)
				.build();
		StringBuilder sb = new StringBuilder();
		for (LlamaModel.Output output : model.generate("", params)) {
			sb.append(output);
		}
		String output = sb.toString();

		Assert.assertTrue(output.matches("[ab]+"));
		Assert.assertEquals(42, model.encode(output).length);
	}

	@Test
	public void testCompleteAnswerDefault() {
		String output = model.complete(prefix);
		Assert.assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteAnswerCustom() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters.Builder()
				.setTemperature(0.95f)
				.setAntiPrompt("\"\"\"")
				.setLogitBias(logitBias)
				.setSeed(42)
				.build();

		String output = model.complete(prefix, params);
		Assert.assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteInfillDefault() {
		String output = model.complete(prefix, suffix);
		Assert.assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteInfillCustom() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters.Builder()
				.setTemperature(0.95f)
				.setAntiPrompt("\"\"\"")
				.setLogitBias(logitBias)
				.setSeed(42)
				.build();

		String output = model.complete(prefix, suffix, params);
		Assert.assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteGrammar() {
		InferenceParameters params = new InferenceParameters.Builder()
				.setGrammar("root ::= (\"a\" | \"b\")+")
				.setNPredict(42)
				.build();
		String output = model.complete("", params);
		Assert.assertTrue(output.matches("[ab]+"));
		Assert.assertEquals(42, model.encode(output).length);
	}

	@Test
	public void testEmbedding() {
		float[] embedding = model.embed(prefix);
		Assert.assertEquals(5120, embedding.length);
	}

	@Test
	public void testTokenization() {
		String prompt = "Hello, world!";
		int[] encoded = model.encode(prompt);
		Assert.assertArrayEquals(new int[]{15043, 29892, 3186, 29991}, encoded);
		String decoded = model.decode(encoded);
		// the llama tokenizer adds a space before the prompt
		Assert.assertEquals(" " + prompt, decoded);
	}

}
