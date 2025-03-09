package de.kherud.llama;

import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

abstract class CliParameters {

    final Map<String, @Nullable String> parameters = new HashMap<>();

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (String key : parameters.keySet()) {
            String value = parameters.get(key);
            builder.append(key).append(" ");
            if (value != null) {
                builder.append(value).append(" ");
            }
        }
        return builder.toString();
    }

    public String[] toArray() {
        List<String> result = new ArrayList<>();
        result.add(""); // c args contain the program name as the first argument, so we add an empty entry
        for (String key : parameters.keySet()) {
            result.add(key);
            String value = parameters.get(key);
            if (value != null) {
                result.add(value);
            }
        }
        return result.toArray(new String[0]);
    }

}
