package de.kherud.llama;

import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

public class LlamaModelIT {

	public static final String MODEL_HOME_PROPERTY = "model.home";
	public static final String MODEL_NAME_PROPERTY = "integration.test.model";
	private static final String prefix = "def remove_non_ascii(s: str) -> str:\n    \"\"\" ";
	private static final String suffix = "\n    return result\n";
	private static String logOutput = "";

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {
		LlamaModel.setLogger((level, msg) -> logOutput += msg);
		String modelHome = System.getProperty(MODEL_HOME_PROPERTY);
		String modelName= System.getProperty(MODEL_NAME_PROPERTY);
		if(modelHome == null) {
			throw new RuntimeException("Please pass the system property \"" + MODEL_HOME_PROPERTY + "\" to the test.  This should represent the location on local disk where your models are located.");
		}
		if(modelName == null) {
			throw new RuntimeException("The system property \"" + MODEL_NAME_PROPERTY + "\" is not set.  If you are running this from an IDE, please set it.  If you are running this from Maven, this should be set automatically and there is something strange going on." );
		}
		String modelPath= Paths.get(modelHome, modelName).toString();
		ModelParameters params = new ModelParameters().setNGpuLayers(43).setEmbedding(true);
		model =
				new LlamaModel(modelPath, params);
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
		InferenceParameters params = new InferenceParameters()
				.setTemperature(0.95f)
				.setAntiPrompt("\"\"\"")
				.setLogitBias(logitBias)
				.setSeed(42);

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
		InferenceParameters params = new InferenceParameters()
				.setTemperature(0.95f)
				.setAntiPrompt("\"\"\"")
				.setLogitBias(logitBias)
				.setSeed(42);

		int generated = 0;
		for (LlamaModel.Output ignored : model.generate(prefix, suffix, params)) {
			generated++;
		}
		Assert.assertTrue(generated > 0);
	}

	@Test
	public void testGenerateGrammar() {
		InferenceParameters params = new InferenceParameters()
				.setGrammar("root ::= (\"a\" | \"b\")+")
				.setNPredict(42);
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
		InferenceParameters params = new InferenceParameters()
				.setTemperature(0.95f)
				.setAntiPrompt("\"\"\"")
				.setLogitBias(logitBias)
				.setSeed(42);

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
		InferenceParameters params = new InferenceParameters()
				.setTemperature(0.95f)
				.setAntiPrompt("\"\"\"")
				.setLogitBias(logitBias)
				.setSeed(42);

		String output = model.complete(prefix, suffix, params);
		Assert.assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteGrammar() {
		InferenceParameters params = new InferenceParameters()
				.setGrammar("root ::= (\"a\" | \"b\")+")
				.setNPredict(42);
		String output = model.complete("", params);
		Assert.assertTrue(output.matches("[ab]+"));
		Assert.assertEquals(42, model.encode(output).length);
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
