package de.kherud.llama;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public class LlamaGrammar {

    public LlamaGrammar(File file) throws IOException {
        this(file.toPath());
    }

    public LlamaGrammar(Path path) throws IOException {
        this(Files.readString(path, StandardCharsets.UTF_8));
    }

    public LlamaGrammar(String grammar) {

    }

    private void parse() {

    }


}
