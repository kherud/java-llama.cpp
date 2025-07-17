package de.kherud.llama;

import de.kherud.llama.args.PoolingType;
import org.junit.*;

import java.lang.management.ManagementFactory;
import java.lang.management.RuntimeMXBean;

public class LlamaEmbeddingsTest {

    private static final String modelPath = "models/codellama-7b.Q2_K.gguf";
    private static LlamaModel model;

    @BeforeClass
    public static void setup() {
        // Print PID of the current process to attach with GDB
        // Remember to set 'echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope' to attach.
        RuntimeMXBean runtime = ManagementFactory.getRuntimeMXBean();
        System.out.println("PID: " + runtime.getName().split("@")[0]);
    }

    @After
    public void tearDownTest() {
        if (model != null) {
            model.close();
        }
    }

    @Test
    public void testEmbeddingTypes() {
        for (PoolingType type : PoolingType.values()) {
            System.out.println("Testing embedding with pooling type: " + type);
            if (type == PoolingType.RANK) {
                continue; // Only supported by reranking models
            }
            model = new LlamaModel(new ModelParameters().setModel(modelPath).setGpuLayers(99).enableEmbedding().setPoolingType(type));
            String text = "This is a test sentence for embedding.";
            float[] embedding = model.embed(text);
            Assert.assertEquals(4096, embedding.length);
            model.close();
        }
    }
}
