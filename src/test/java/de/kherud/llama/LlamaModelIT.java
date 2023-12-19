package de.kherud.llama;

import java.util.HashMap;
import java.util.Map;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

public class LlamaModelIT {

	private static final String prefix = "def remove_non_ascii(s: str) -> str:\n    \"\"\" ";
	private static final String suffix = "\n    return result\n";
	private static String logOutput = "";
	private static final int nPredict = 10;

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {
		LlamaModel.setLogger((level, msg) -> logOutput += msg);
		ModelParameters params = new ModelParameters()
				.setNGpuLayers(43)
				.setEmbedding(true);
		model = new LlamaModel(ModelResolver.getPathToITModel(), params);
	}

	@AfterClass
	public static void tearDown() {
		if(model != null) {
			model.close();
		}
	}

	@Test
	public void testLogOutput() {
		Assert.assertFalse(logOutput.isEmpty());
	}

	@Test
	public void testGenerateAnswer() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters()
				.setTemperature(0.95f)
				.setAntiPrompt("\"\"\"")
				.setNPredict(nPredict)
				.setLogitBias(logitBias)
				.setSeed(42);

		int generated = 0;
		for (LlamaModel.Output ignored : model.generate(prefix, params)) {
			generated++;
		}
		Assert.assertTrue(generated > 0 && generated <= nPredict);
	}

	@Test
	public void testGenerateInfill() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters()
				.setTemperature(0.95f)
				.setAntiPrompt("\"\"\"")
				.setNPredict(nPredict)
				.setLogitBias(logitBias)
				.setSeed(42);

		int generated = 0;
		for (LlamaModel.Output ignored : model.generate(prefix, suffix, params)) {
			generated++;
		}
		Assert.assertTrue(generated > 0 && generated <= nPredict);
	}

	@Test
	public void testGenerateGrammar() {
		InferenceParameters params = new InferenceParameters()
				.setGrammar("root ::= (\"a\" | \"b\")+")
				.setNPredict(nPredict);
		StringBuilder sb = new StringBuilder();
		for (LlamaModel.Output output : model.generate("", params)) {
			sb.append(output);
		}
		String output = sb.toString();

		Assert.assertTrue(output.matches("[ab]+"));
		int generated = model.encode(output).length;
		Assert.assertTrue(generated > 0 && generated <= nPredict);
	}

	@Test
	public void testCompleteAnswer() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters()
				.setTemperature(0.95f)
				.setAntiPrompt("\"\"\"")
				.setNPredict(nPredict)
				.setLogitBias(logitBias)
				.setSeed(42);

		String output = model.complete(prefix, params);
		Assert.assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteInfillCustom() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters()
				.setTemperature(0.95f)
				.setAntiPrompt("\"\"\"")
				.setNPredict(nPredict)
				.setLogitBias(logitBias)
				.setSeed(42);

		String output = model.complete(prefix, suffix, params);
		Assert.assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteGrammar() {
		InferenceParameters params = new InferenceParameters()
				.setGrammar("root ::= (\"a\" | \"b\")+")
				.setNPredict(nPredict);
		String output = model.complete("", params);
		Assert.assertTrue(output.matches("[ab]+"));
		int generated = model.encode(output).length;
		Assert.assertTrue(generated > 0 && generated <= nPredict);
	}

	@Test
	public void testEmbedding() {
		float[] embedding = model.embed(prefix);
		Assert.assertEquals(4096, embedding.length);
	}

	@Test
	public void testTokenization() {
		String prompt = "Hello, world!";
		int[] encoded = model.encode(prompt);
		String decoded = model.decode(encoded);
		// the llama tokenizer adds a space before the prompt
		Assert.assertEquals(" " + prompt, decoded);
	}

}
