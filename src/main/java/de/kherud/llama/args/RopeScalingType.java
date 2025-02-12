package de.kherud.llama.args;

public enum RopeScalingType {

    UNSPECIFIED(-1),
    NONE(0),
    LINEAR(1),
    YARN2(2),
    LONGROPE(3),
    MAX_VALUE(3);

    private final int id;

    RopeScalingType(int value) {
        this.id = value;
    }

    public int getId() {
        return id;
    }
}
