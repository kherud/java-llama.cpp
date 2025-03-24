package de.kherud.llama;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

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
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
	}
	
	@Test
	public void testEmbedding() {
		float[] embedding = model.embed("You are an  AI Assistant");
		Assert.assertEquals(2560, embedding.length);
	}
}
