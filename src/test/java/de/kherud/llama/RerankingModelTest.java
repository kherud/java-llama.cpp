package de.kherud.llama;

import java.util.HashMap;
import java.util.Map;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import com.fasterxml.jackson.databind.JsonNode;

public class RerankingModelTest {

	private static LlamaModel model;

	String query = "Machine learning is";
	String[] TEST_DOCUMENTS = new String[] {
			"A machine is a physical system that uses power to apply forces and control movement to perform an action. The term is commonly applied to artificial devices, such as those employing engines or motors, but also to natural biological macromolecules, such as molecular machines.",
			"Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences. The ability to learn is possessed by humans, non-human animals, and some machines; there is also evidence for some kind of learning in certain plants.",
			"Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.",
			"Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine." };

	@BeforeClass
	public static void setup() {
		model = new LlamaModel(
				new ModelParameters().setCtxSize(4096).setModel("models/jina-reranker-v1-tiny-en-Q4_0.gguf")
						.enableReranking().enableLogTimestamps().enableLogPrefix());
	}

	@AfterClass
	public static void tearDown() throws Exception {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testReRanking() {

		InferenceParameters params = new InferenceParameters();
		params.setQuery(query);
		params.setDocuments(TEST_DOCUMENTS);
		String llamaOutput = model.handleRerank(params.toString());

		JsonNode resultNode = JsonUtils.INSTANCE.jsonToNode(llamaOutput).get("results");

		Map<Integer, Float> relevanceScores = new HashMap<>();

		// Iterate through the results array
		if (resultNode.isArray()) {
			for (JsonNode item : resultNode) {
				// Extract index and relevance_score from each item
				int index = item.get("index").asInt();
				float score = item.get("relevance_score").floatValue();

				// Add to map
				relevanceScores.put(index, score);
			}
		}
		Assert.assertTrue(relevanceScores.size() == TEST_DOCUMENTS.length);

		// Finding the most and least relevant documents
		Integer mostRelevantDoc = null;
		Integer leastRelevantDoc = null;
		float maxScore = Float.MIN_VALUE;
		float minScore = Float.MAX_VALUE;

		for (Map.Entry<Integer, Float> entry : relevanceScores.entrySet()) {
			if (entry.getValue() > maxScore) {
				maxScore = entry.getValue();
				mostRelevantDoc = entry.getKey();
			}
			if (entry.getValue() < minScore) {
				minScore = entry.getValue();
				leastRelevantDoc = entry.getKey();
			}
		}

		// Assertions
		Assert.assertTrue(maxScore > minScore);
		Assert.assertEquals(
				"Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.",
				TEST_DOCUMENTS[mostRelevantDoc]);
		Assert.assertEquals(
				"Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine.",
				TEST_DOCUMENTS[leastRelevantDoc]);

	}

}
