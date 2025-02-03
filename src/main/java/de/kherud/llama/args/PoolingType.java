package de.kherud.llama.args;

public enum PoolingType {

    UNSPECIFIED(-1),
    NONE(0),
    MEAN(1),
    CLS(2),
    LAST(3),
    RANK(4);

    private final int id;

    PoolingType(int value) {
        this.id = value;
    }

    public int getId() {
        return id;
    }
}
