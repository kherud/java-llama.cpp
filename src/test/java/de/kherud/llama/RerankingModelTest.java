package de.kherud.llama;

import java.util.List;
import java.util.Map;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

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
				new ModelParameters().setCtxSize(128).setModel("models/jina-reranker-v1-tiny-en-Q4_0.gguf")
						.setGpuLayers(43).enableReranking().enableLogTimestamps().enableLogPrefix());
	}

	@AfterClass
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testReRanking() {

		
		LlamaOutput llamaOutput = model.rerank(query, TEST_DOCUMENTS[0], TEST_DOCUMENTS[1], TEST_DOCUMENTS[2],
				TEST_DOCUMENTS[3]);

		Map<String, Float> rankedDocumentsMap = llamaOutput.probabilities;
		Assert.assertTrue(rankedDocumentsMap.size()==TEST_DOCUMENTS.length);
		
		 // Finding the most and least relevant documents
        String mostRelevantDoc = null;
        String leastRelevantDoc = null;
        float maxScore = Float.MIN_VALUE;
        float minScore = Float.MAX_VALUE;

        for (Map.Entry<String, Float> entry : rankedDocumentsMap.entrySet()) {
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
        Assert.assertEquals("Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.", mostRelevantDoc);
        Assert.assertEquals("Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine.", leastRelevantDoc);

		
	}
	
	@Test
	public void testSortedReRanking() {
		List<Pair<String, Float>> rankedDocuments = model.rerank(true, query, TEST_DOCUMENTS);
		Assert.assertEquals(rankedDocuments.size(), TEST_DOCUMENTS.length);
		
		// Check the ranking order: each score should be >= the next one
	    for (int i = 0; i < rankedDocuments.size() - 1; i++) {
	        float currentScore = rankedDocuments.get(i).getValue();
	        float nextScore = rankedDocuments.get(i + 1).getValue();
	        Assert.assertTrue("Ranking order incorrect at index " + i, currentScore >= nextScore);
	    }
	}
}
