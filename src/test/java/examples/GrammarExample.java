package examples;

import de.kherud.llama.LlamaModel;
import de.kherud.llama.Parameters;

public class GrammarExample {

    public static void main(String... args) {
        String grammar = "root  ::= (expr \"=\" term \"\\n\")+\n" +
                "expr  ::= term ([-+*/] term)*\n" +
                "term  ::= [0-9]+";
        Parameters params = new Parameters.Builder()
                .setNGpuLayers(43)
                .setTemperature(0.7f)
                .setMirostat(Parameters.MiroStat.V2)
                .setGrammar(grammar)
                .build();

        String modelPath = "/Users/konstantin.herud/denkbares/projects/llama.cpp/models/13B/gguf-model-q4_0.bin";
        try (LlamaModel model = new LlamaModel(modelPath, params)) {
            for (LlamaModel.Output output : model.generate("1+1=")) {
                System.out.print(output);
            }
        }
    }

}
