package de.kherud.llama;

import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static de.kherud.llama.LlamaGrammar.Pair;
import static de.kherud.llama.foreign.LlamaLibrary.llama_gretype.*;

public class GrammarTest {

    @Test
    public void testParseGrammar1() {
        String grammar = "root  ::= (expr \"=\" term \"\\n\")+\n" +
                "expr  ::= term ([-+*/] term)*\n" +
                "term  ::= [0-9]+";

        LlamaGrammar.ParseState state = new LlamaGrammar.ParseState(grammar);

        Map<String, Integer> expectedIds = Map.of(
                "expr", 2,
                "expr_5", 5,
                "expr_6", 6,
                "root", 0,
                "root_1", 1,
                "root_4", 4,
                "term", 3,
                "term_7", 7
        );

        Assert.assertEquals("unexpected amount of symbol ids", expectedIds.size(), state.symbolIds.size());
        expectedIds.forEach((expectedKey, expectedValue) -> {
            Integer actualValue = state.symbolIds.get(expectedKey);
            Assert.assertNotNull("expected key " + expectedKey + " missing ", actualValue);
            Assert.assertEquals("values of rule " + expectedValue + " do not match", expectedValue, actualValue);
        });

        List<Pair<Integer, Integer>> expectedRules = List.of(
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 4),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 2),
                new Pair<>(LLAMA_GRETYPE_CHAR, 61),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 3),
                new Pair<>(LLAMA_GRETYPE_CHAR, 10),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 3),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 6),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 7),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 1),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 4),
                new Pair<>(LLAMA_GRETYPE_ALT, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 1),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_CHAR, 45),
                new Pair<>(LLAMA_GRETYPE_CHAR_ALT, 43),
                new Pair<>(LLAMA_GRETYPE_CHAR_ALT, 42),
                new Pair<>(LLAMA_GRETYPE_CHAR_ALT, 47),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 3),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 5),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 6),
                new Pair<>(LLAMA_GRETYPE_ALT, 0),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_CHAR, 48),
                new Pair<>(LLAMA_GRETYPE_CHAR_RNG_UPPER, 57),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 7),
                new Pair<>(LLAMA_GRETYPE_ALT, 0),
                new Pair<>(LLAMA_GRETYPE_CHAR, 48),
                new Pair<>(LLAMA_GRETYPE_CHAR_RNG_UPPER, 57),
                new Pair<>(LLAMA_GRETYPE_END, 0)
        );


        int index = 0;
        for (List<Pair<Integer, Integer>> rule : state.rules) {
            for (Pair<Integer, Integer> actualElement : rule) {
                Pair<Integer, Integer> expectedElement = expectedRules.get(index);
                Assert.assertEquals("grammar element types do not match", expectedElement.a, actualElement.a);
                Assert.assertEquals("grammar element values do not match", expectedElement.b, actualElement.b);
                index++;
            }
        }

        Assert.assertFalse("parse state has no rules", state.rules.isEmpty());
    }

    @Test
    public void testParseGrammar2() {
        String grammar = "root  ::= (expr \"=\" ws term \"\\n\")+\n" +
                "expr  ::= term ([-+*/] term)*\n" +
                "term  ::= ident | num | \"(\" ws expr \")\" ws\n" +
                "ident ::= [a-z] [a-z0-9_]* ws\n" +
                "num   ::= [0-9]+ ws\n" +
                "ws    ::= [ \\t\\n]*";

        LlamaGrammar.ParseState state = new LlamaGrammar.ParseState(grammar);

        Map<String, Integer> expectedIds = new HashMap<>();
        expectedIds.put("expr", 2);
        expectedIds.put("expr_6", 6);
        expectedIds.put("expr_7", 7);
        expectedIds.put("ident", 8);
        expectedIds.put("ident_10", 10);
        expectedIds.put("num", 9);
        expectedIds.put("num_11", 11);
        expectedIds.put("root", 0);
        expectedIds.put("root_1", 1);
        expectedIds.put("root_5", 5);
        expectedIds.put("term", 4);
        expectedIds.put("ws", 3);
        expectedIds.put("ws_12", 12);

        Assert.assertEquals("unexpected amount of symbol ids", expectedIds.size(), state.symbolIds.size());
        expectedIds.forEach((expectedKey, expectedValue) -> {
            Integer actualValue = state.symbolIds.get(expectedKey);
            Assert.assertNotNull("expected key " + expectedKey + " missing ", actualValue);
            Assert.assertEquals("values of rule " + expectedValue + " do not match", expectedValue, actualValue);
        });

        List<Pair<Integer, Integer>> expectedRules = List.of(
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 5),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 2),
                new Pair<>(LLAMA_GRETYPE_CHAR, 61),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 3),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 4),
                new Pair<>(LLAMA_GRETYPE_CHAR, 10),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 4),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 7),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 12),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 8),
                new Pair<>(LLAMA_GRETYPE_ALT, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 9),
                new Pair<>(LLAMA_GRETYPE_ALT, 0),
                new Pair<>(LLAMA_GRETYPE_CHAR, 40),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 3),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 2),
                new Pair<>(LLAMA_GRETYPE_CHAR, 41),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 3),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 1),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 5),
                new Pair<>(LLAMA_GRETYPE_ALT, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 1),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_CHAR, 45),
                new Pair<>(LLAMA_GRETYPE_CHAR_ALT, 43),
                new Pair<>(LLAMA_GRETYPE_CHAR_ALT, 42),
                new Pair<>(LLAMA_GRETYPE_CHAR_ALT, 47),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 4),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 6),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 7),
                new Pair<>(LLAMA_GRETYPE_ALT, 0),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_CHAR, 97),
                new Pair<>(LLAMA_GRETYPE_CHAR_RNG_UPPER, 122),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 10),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 3),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 11),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 3),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_CHAR, 97),
                new Pair<>(LLAMA_GRETYPE_CHAR_RNG_UPPER, 122),
                new Pair<>(LLAMA_GRETYPE_CHAR_ALT, 48),
                new Pair<>(LLAMA_GRETYPE_CHAR_RNG_UPPER, 57),
                new Pair<>(LLAMA_GRETYPE_CHAR_ALT, 95),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 10),
                new Pair<>(LLAMA_GRETYPE_ALT, 0),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_CHAR, 48),
                new Pair<>(LLAMA_GRETYPE_CHAR_RNG_UPPER, 57),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 11),
                new Pair<>(LLAMA_GRETYPE_ALT, 0),
                new Pair<>(LLAMA_GRETYPE_CHAR, 48),
                new Pair<>(LLAMA_GRETYPE_CHAR_RNG_UPPER, 57),
                new Pair<>(LLAMA_GRETYPE_END, 0),
                new Pair<>(LLAMA_GRETYPE_CHAR, 32),
                new Pair<>(LLAMA_GRETYPE_CHAR_ALT, 9),
                new Pair<>(LLAMA_GRETYPE_CHAR_ALT, 10),
                new Pair<>(LLAMA_GRETYPE_RULE_REF, 12),
                new Pair<>(LLAMA_GRETYPE_ALT, 0),
                new Pair<>(LLAMA_GRETYPE_END, 0)
        );


        int index = 0;
        for (List<Pair<Integer, Integer>> rule : state.rules) {
            for (Pair<Integer, Integer> actualElement : rule) {
                Pair<Integer, Integer> expectedElement = expectedRules.get(index);
                Assert.assertEquals("grammar element types do not match", expectedElement.a, actualElement.a);
                Assert.assertEquals("grammar element values do not match", expectedElement.b, actualElement.b);
                index++;
            }
        }

        Assert.assertFalse("parse state has no rules", state.rules.isEmpty());
    }

}
