package de.kherud.llama.args;

public enum PoolingType {

    UNSPECIFIED("unspecified"),
    NONE("none"),
    MEAN("mean"),
    CLS("cls"),
    LAST("last"),
    RANK("rank");

    private final String argValue;

    PoolingType(String value) {
        this.argValue = value;
    }

    public String getArgValue() {
        return argValue;
    }
}