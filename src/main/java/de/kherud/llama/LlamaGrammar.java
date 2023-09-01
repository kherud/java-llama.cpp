package de.kherud.llama;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import de.kherud.llama.foreign.LlamaLibrary;
import de.kherud.llama.foreign.NativeSize;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static de.kherud.llama.foreign.LlamaLibrary.llama_gretype.*;

/**
 * Implements a parser for an extended Backus-Naur form (BNF), producing the binary context-free grammar format
 * specified by llama.h. Supports character ranges, grouping, and repetition operators. As an example, a grammar for
 * arithmetic might look like:
 * <pre>
 * root  ::= expr
 * expr  ::= term ([-+/] term)
 * term  ::= num | "(" space expr ")" space
 * num   ::= [0-9]+ space
 * space ::= [ \t\n]*
 * </pre>
 * This class is implemented as an {@link AutoCloseable} in order to free the natively allocated memory for this
 * grammar, after it is no longer needed.
 */
public class LlamaGrammar implements AutoCloseable {

    private final ParseState state;
    final LlamaLibrary.llama_grammar foreign;

    /**
     * Loads a grammar from a file, directly parses it, and allocates a native {@link LlamaLibrary.llama_grammar}.
     *
     * @param file the file containing a grammar
     * @throws IOException on problems reading the file
     */
    public LlamaGrammar(File file) throws IOException {
        this(file.toPath());
    }

    /**
     * Loads a grammar from a path, directly parses it, and allocates a native {@link LlamaLibrary.llama_grammar}.
     *
     * @param path the path to a file containing a grammar
     * @throws IOException on problems reading the file
     */
    public LlamaGrammar(Path path) throws IOException {
        this(Files.readString(path, StandardCharsets.UTF_8));
    }

    /**
     * Parses a grammar given as string and allocates a native {@link LlamaLibrary.llama_grammar}.
     *
     * @param grammar the grammar string
     */
    public LlamaGrammar(String grammar) {
        this.state = new ParseState(grammar);
        this.foreign = state.create();
    }

    @Override
    public String toString() {
        return state.toString();
    }

    @Override
    public void close() {
        LlamaLibrary.llama_grammar_free(foreign);
    }

    /**
     * The state object used for parsing a grammar.
     * It can't be modified and is used to allocate the native {@link LlamaLibrary.llama_grammar}.
     */
    static final class ParseState {

        final Map<String, Integer> symbolIds = new HashMap<>();
        final List<List<Pair<Integer, Integer>>> rules = new ArrayList<>();
        private String string;

        ParseState(String grammar) {
            parse(grammar.getBytes(StandardCharsets.UTF_8));
        }

        private void parse(byte[] src) {
            int pos = parseSpace(src, 0, true);
            while (pos < src.length) {
                pos = parseRule(src, pos);
            }
        }

        private LlamaLibrary.llama_grammar create() {
            // creating the grammar is a bit tricky, since we need to allocate contiguous memory segments for the
            // elements of each rule

            List<Pointer> rulePointers = new ArrayList<>();
            for (List<Pair<Integer, Integer>> rule : rules) {
                if (rule.isEmpty()) {
                    continue;
                }
                Memory memory = new Memory((long) rule.size() * 8);
                for (int i = 0; i < rule.size(); i++) {
                    Pair<Integer, Integer> element = rule.get(i);
                    memory.setInt((long) i * 8, element.a);
                    memory.setInt((long) i * 8 + 4, element.b);
                }
                rulePointers.add(memory);
            }

            Pointer memory = new Memory((long) Native.POINTER_SIZE * rulePointers.size());
            for (int i = 0; i < rulePointers.size(); i++) {
                memory.setPointer((long) i * Native.POINTER_SIZE, rulePointers.get(i));
            }

            NativeSize nRules = new NativeSize(rulePointers.size());
            NativeSize startRuleIndex = new NativeSize(symbolIds.get("root"));

            return LlamaLibrary.llama_grammar_init(memory, nRules, startRuleIndex);
        }

        private int parseSpace(byte[] src, int pos, boolean newlineOk) {
            while (pos < src.length) {
                byte currentChar = src[pos];
                if (currentChar == ' '
                        || currentChar == '\t'
                        || currentChar == '#'
                        || (newlineOk && (currentChar == '\r' || currentChar == '\n'))) {
                    if (currentChar == '#') {
                        while (pos < src.length
                                && src[pos] != '\r'
                                && src[pos] != '\n') {
                            pos++;
                        }
                    } else {
                        pos++;
                    }
                } else {
                    break;
                }
            }
            return pos;
        }

        private int parseRule(byte[] src, int startPos) throws RuntimeException {
            int nameEnd = parseName(src, startPos);
            int pos = parseSpace(src, nameEnd, false);
            int nameLen = nameEnd - startPos;

            int ruleId = getSymbolId(src, startPos, nameLen);
            String name = new String(src, startPos, nameLen, StandardCharsets.UTF_8);

            if (src.length <= pos + 2 || !(src[pos] == ':' && src[pos + 1] == ':' && src[pos + 2] == '=')) {
                throw new RuntimeException("Expecting ::= at position " + pos);
            }
            pos = parseSpace(src, pos + 3, true);

            pos = parseAlternates(src, pos, name, ruleId, false);

            if (pos < src.length) {
                if (src[pos] == '\r') {
                    pos += (pos + 1 < src.length && src[pos + 1] == '\n') ? 2 : 1;
                } else if (src[pos] == '\n') {
                    pos++;
                } else {
                    throw new RuntimeException("Expecting newline or end at position " + pos);
                }
            }

            return parseSpace(src, pos, true);
        }

        private int parseName(byte[] src, int startPos) {
            int pos = startPos;
            while (pos < src.length && isWordChar(src[pos])) {
                pos++;
            }
            if (pos == startPos) {
                throw new RuntimeException("Expecting name at position " + startPos);
            }
            return pos;
        }

        private boolean isWordChar(byte c) {
            return 'a' <= c && c <= 'z'
                    || 'A' <= c && c <= 'Z'
                    || c == '-'
                    || ('0' <= c && c <= '9');
        }

        private int getSymbolId(byte[] src, int pos, int len) {
            String str = new String(src, pos, len, StandardCharsets.UTF_8);
            if (!symbolIds.containsKey(str)) {
                int nextId = symbolIds.size();
                symbolIds.put(str, nextId);
            }
            return symbolIds.get(str);
        }

        private int parseAlternates(byte[] src, int startPos, String ruleName, int ruleId, boolean isNested) {
            List<Pair<Integer, Integer>> rule = new ArrayList<>();
            int pos = parseSequence(ruleName, src, startPos, rule, isNested);

            while (pos < src.length && src[pos] == '|') {
                rule.add(new Pair<>(LLAMA_GRETYPE_ALT, 0));
                pos = parseSpace(src, pos + 1, true);
                pos = parseSequence(ruleName, src, pos, rule, isNested);
            }

            rule.add(new Pair<>(LLAMA_GRETYPE_END, 0));
            addRule(ruleId, rule);

            return pos;
        }

        private void addRule(int ruleId, List<Pair<Integer, Integer>> rule) {
            // resize the rules ArrayList if necessary
            while (rules.size() <= ruleId) {
                rules.add(null);
            }
            rules.set(ruleId, rule);
        }

        private int parseSequence(String ruleName, byte[] src, int startPos, List<Pair<Integer, Integer>> outElements, boolean isNested) {
            int lastSymStart = outElements.size();
            int pos = startPos;

            while (pos < src.length) {
                if (src[pos] == '"') {
                    pos++;
                    lastSymStart = outElements.size();
                    while (src[pos] != '"') {
                        Pair<Integer, Integer> charPair = parseChar(src, pos);
                        pos = charPair.b;
                        outElements.add(new Pair<>(LLAMA_GRETYPE_CHAR, charPair.a));
                    }
                    pos = parseSpace(src, pos + 1, isNested);
                } else if (src[pos] == '[') { // char range(s)
                    pos++;
                    int startType = LLAMA_GRETYPE_CHAR;
                    if (src[pos] == '^') {
                        pos++;
                        startType = LLAMA_GRETYPE_CHAR_NOT;
                    }
                    lastSymStart = outElements.size();
                    while (src[pos] != ']') {
                        Pair<Integer, Integer> charPair = parseChar(src, pos);
                        pos = charPair.b;
                        int type = lastSymStart < outElements.size() ? LLAMA_GRETYPE_CHAR_ALT : startType;

                        outElements.add(new Pair<>(type, charPair.a));
                        if (src[pos] == '-' && src[pos + 1] != ']') {
                            Pair<Integer, Integer> endCharPair = parseChar(src, pos + 1);
                            pos = endCharPair.b;
                            outElements.add(new Pair<>(LLAMA_GRETYPE_CHAR_RNG_UPPER, endCharPair.a));
                        }
                    }
                    pos = parseSpace(src, pos + 1, isNested);
                } else if (isWordChar(src[pos])) { // rule reference
                    int nameEnd = parseName(src, pos);
                    int refRuleId = getSymbolId(src, pos, nameEnd - pos);
                    pos = parseSpace(src, nameEnd, isNested);
                    lastSymStart = outElements.size();
                    outElements.add(new Pair<>(LLAMA_GRETYPE_RULE_REF, refRuleId));
                } else if (src[pos] == '(') { // grouping
                    pos = parseSpace(src, pos + 1, true);
                    int subRuleId = generateSymbolId(ruleName);
                    pos = parseAlternates(src, pos, ruleName, subRuleId, true);
                    lastSymStart = outElements.size();
                    outElements.add(new Pair<>(LLAMA_GRETYPE_RULE_REF, subRuleId));
                    if (src[pos] != ')') {
                        throw new RuntimeException("expecting ')' at " + pos);
                    }
                    pos = parseSpace(src, pos + 1, isNested);
                } else if (src[pos] == '*' || src[pos] == '+' || src[pos] == '?') {
                    if (lastSymStart == outElements.size()) {
                        throw new RuntimeException("expecting preceding item to */+/? at " + pos);
                    }
                    // apply transformation to previous symbol (last_sym_start to end) according to
                    // rewrite rules:
                    // S* --> S' ::= S S' |
                    // S+ --> S' ::= S S' | S
                    // S? --> S' ::= S |
                    int subRuleId = generateSymbolId(ruleName);
                    // Add preceding symbol to generated rule
                    List<Pair<Integer, Integer>> subRule = new ArrayList<>(outElements.subList(lastSymStart, outElements.size()));
                    if (src[pos] == '*' || src[pos] == '+') {
                        // Cause generated rule to recurse
                        subRule.add(new Pair<>(LLAMA_GRETYPE_RULE_REF, subRuleId));
                    }
                    // Mark start of alternate def
                    subRule.add(new Pair<>(LLAMA_GRETYPE_ALT, 0));
                    if (src[pos] == '+') {
                        // Add preceding symbol as alternate only for '+' (otherwise empty)
                        subRule.addAll(outElements.subList(lastSymStart, outElements.size()));
                    }
                    subRule.add(new Pair<>(LLAMA_GRETYPE_END, 0));
                    addRule(subRuleId, subRule);

                    // In original rule, replace previous symbol with reference to generated rule
                    outElements.subList(lastSymStart, outElements.size()).clear();
                    outElements.add(new Pair<>(LLAMA_GRETYPE_RULE_REF, subRuleId));

                    pos = parseSpace(src, pos + 1, isNested);
                } else {
                    break;
                }
            }
            return pos;
        }

        private Pair<Integer, Integer> parseChar(byte[] src, int pos) throws RuntimeException {
            if (src[pos] == '\\') {
                switch (src[pos + 1]) {
                    case 'x':
                        return parseHex(src, pos + 2, 2);
                    case 'u':
                        return parseHex(src, pos + 2, 4);
                    case 'U':
                        return parseHex(src, pos + 2, 8);
                    case 't':
                        return new Pair<>((int) '\t', pos + 2);
                    case 'r':
                        return new Pair<>((int) '\r', pos + 2);
                    case 'n':
                        return new Pair<>((int) '\n', pos + 2);
                    case '\\':
                    case '"':
                    case '[':
                    case ']':
                        return new Pair<>((int) src[pos + 1], pos + 2);
                    default:
                        throw new RuntimeException("Unknown escape at " + pos);
                }
            } else {
                return decodeUtf8(src, pos);
            }
        }

        private Pair<Integer, Integer> parseHex(byte[] src, int pos, int size) throws RuntimeException {
            int end = pos + size;
            int value = 0;
            for (; pos < end && src[pos] != 0; pos++) {
                value <<= 4;
                byte c = src[pos];
                if ('a' <= c && c <= 'f') {
                    value += c - 'a' + 10;
                } else if ('A' <= c && c <= 'F') {
                    value += c - 'A' + 10;
                } else if ('0' <= c && c <= '9') {
                    value += c - '0';
                } else {
                    break;
                }
            }
            if (pos != end) {
                throw new RuntimeException("Expecting " + size + " hex chars at " + pos);
            }
            return new Pair<>(value, pos);
        }

        private Pair<Integer, Integer> decodeUtf8(byte[] src, int pos) {
            final int[] lookup = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
            byte firstByte = src[pos];
            byte highBits = (byte) (firstByte >> 4);
            int len = lookup[highBits];
            byte mask = (byte) ((1 << (8 - len)) - 1);
            int value = firstByte & mask;
            int end = pos + len;
            pos = pos + 1;
            for (; pos < end && src[pos] != 0; pos++) {
                value = (value << 6) + (src[pos] & 0x3F);
            }
            return new Pair<>(value, pos);
        }

        private int generateSymbolId(String baseName) {
            int nextId = symbolIds.size();
            symbolIds.put(baseName + "_" + nextId, nextId);
            return nextId;
        }

        @Override
        public String toString() {
            if (string != null) {
                return string;
            }
            Map<Integer, String> symbolIdNames = new HashMap<>();
            for (Map.Entry<String, Integer> entry : symbolIds.entrySet()) {
                symbolIdNames.put(entry.getValue(), entry.getKey());
            }
            StringBuilder grammarBuilder = new StringBuilder();
            for (int i = 0, end = rules.size(); i < end; i++) {
                grammarBuilder.append(i).append(": ");
                List<Pair<Integer, Integer>> rule = rules.get(i);
                appendRule(grammarBuilder, symbolIdNames, i, rule);
            }
            string = grammarBuilder.toString();
            return string;
        }

        private void appendRule(StringBuilder builder, Map<Integer, String> symbolIdNames, int ruleId, List<Pair<Integer, Integer>> rule) {
            if (rule.isEmpty() || rule.get(rule.size() - 1).a != LLAMA_GRETYPE_END) {
                throw new RuntimeException("Malformed rule, does not end with LLAMA_GRETYPE_END: " + ruleId);
            }
            builder.append(symbolIdNames.get(ruleId)).append(" ::= ");
            for (int i = 0, end = rule.size() - 1; i < end; i++) {
                Pair<Integer, Integer> elem = rule.get(i);
                switch (elem.a) {
                    case LLAMA_GRETYPE_END:
                        throw new RuntimeException("Unexpected end of rule: " + ruleId + "," + i);
                    case LLAMA_GRETYPE_ALT:
                        builder.append("| ");
                        break;
                    case LLAMA_GRETYPE_RULE_REF:
                        builder.append(symbolIdNames.get(elem.b)).append(" ");
                        break;
                    case LLAMA_GRETYPE_CHAR:
                        builder.append("[");
                        printGrammarChar(builder, elem.b);
                        break;
                    case LLAMA_GRETYPE_CHAR_NOT:
                        builder.append("[^");
                        printGrammarChar(builder, elem.b);
                        break;
                    case LLAMA_GRETYPE_CHAR_RNG_UPPER:
                        if (i == 0 || !isCharElement(rule.get(i - 1))) {
                            throw new RuntimeException(
                                    "LLAMA_GRETYPE_CHAR_RNG_UPPER without preceding char: " + ruleId + "," + i);
                        }
                        builder.append("-");
                        printGrammarChar(builder, elem.b);
                        break;
                    case LLAMA_GRETYPE_CHAR_ALT:
                        if (i == 0 || !isCharElement(rule.get(i - 1))) {
                            throw new RuntimeException(
                                    "LLAMA_GRETYPE_CHAR_ALT without preceding char: " + ruleId + "," + i);
                        }
                        printGrammarChar(builder, elem.b);
                        break;
                    default:
                        throw new RuntimeException("Unknown type: " + elem.b);
                }
                if (isCharElement(elem)) {
                    switch (rule.get(i + 1).b) {
                        case LLAMA_GRETYPE_CHAR_ALT:
                        case LLAMA_GRETYPE_CHAR_RNG_UPPER:
                            break;
                        default:
                            builder.append("] ");
                    }
                }
            }
            builder.append("\n");
        }

        private void printGrammarChar(StringBuilder builder, int c) {
            if (c >= 0x20 && c <= 0x7f) {
                builder.append((char) c);
            } else {
                builder.append(String.format("<U+%04X>", c));
            }
        }

        private boolean isCharElement(Pair<Integer, Integer> element) {
            switch (element.a) {
                case LLAMA_GRETYPE_CHAR:
                case LLAMA_GRETYPE_CHAR_NOT:
                case LLAMA_GRETYPE_CHAR_ALT:
                case LLAMA_GRETYPE_CHAR_RNG_UPPER:
                    return true;
                default:
                    return false;
            }
        }
    }

    static final class Pair<A, B> {
        final A a;
        final B b;

        Pair(A a, B b) {
            this.a = a;
            this.b = b;
        }
    }
}
