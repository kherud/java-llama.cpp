package de.kherud.llama;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import de.kherud.llama.foreign.LlamaLibrary;
import de.kherud.llama.foreign.NativeSize;
import de.kherud.llama.foreign.llama_grammar_element;

import static de.kherud.llama.foreign.LlamaLibrary.llama_gretype.*;

public class LlamaGrammar {

	private final ParseState state;
	final LlamaLibrary.llama_grammar grammar;

	public LlamaGrammar(File file) throws IOException {
		this(file.toPath());
	}

	public LlamaGrammar(Path path) throws IOException {
		this(Files.readString(path, StandardCharsets.UTF_8));
	}

	public LlamaGrammar(String grammar) {
		this.state = new ParseState(grammar);
		this.grammar = state.create();
	}

	@Override
	public String toString() {
		return state.toString();
	}

	private static final class ParseState {

		private final Map<String, Integer> symbolIds = new HashMap<>();
		private final List<List<llama_grammar_element>> rules = new ArrayList<>();
		private String string;

		private ParseState(String grammar) {
			parse(grammar);
		}

		private void parse(String grammar) {
			String pos = parseSpace(grammar, true);
			while (!pos.isEmpty()) {
				pos = parseRule(pos);
			}
		}

		private LlamaLibrary.llama_grammar create() {
			// 1. Flatten the 'rules' and get their native pointers.
			List<Pointer> pointersToRules = new ArrayList<>();
			int elementSize = 4;

			for (List<llama_grammar_element> rule : rules) {
				llama_grammar_element[] ruleArray = rule.toArray(new llama_grammar_element[0]);
				Memory ruleMemory = new Memory((long) ruleArray.length * elementSize);
				for (int i = 0; i < ruleArray.length; i++) {
					byte[] buffer = new byte[elementSize];
					ruleArray[i].write();
					ruleArray[i].getPointer().read(0, buffer, 0, elementSize);
					ruleMemory.write((long) i * elementSize, buffer, 0, elementSize);
				}
				pointersToRules.add(ruleMemory);
			}

			// 3. Prepare the array of pointers
			PointerByReference rulesArray = new PointerByReference();
			Memory rulesMemory = new Memory((long) pointersToRules.size() * Native.POINTER_SIZE);
			for (int i = 0; i < pointersToRules.size(); i++) {
				rulesMemory.setPointer((long) i * Native.POINTER_SIZE, pointersToRules.get(i));
			}
			rulesArray.setValue(rulesMemory);

			NativeSize nRules = new NativeSize(pointersToRules.size());
			NativeSize rootId = new NativeSize(symbolIds.get("root"));

			return LlamaLibrary.llama_grammar_init(rulesArray, nRules, rootId);
		}

		private String parseSpace(String src, boolean newlineOk) {
			int pos = 0;
			while (pos < src.length()) {
				char currentChar = src.charAt(pos);
				if (currentChar == ' '
						|| currentChar == '\t'
						|| currentChar == '#'
						|| (newlineOk && (currentChar == '\r' || currentChar == '\n'))) {
					if (currentChar == '#') {
						while (pos < src.length()
								&& src.charAt(pos) != '\r'
								&& src.charAt(pos) != '\n') {
							pos++;
						}
					}
					else {
						pos++;
					}
				}
				else {
					break;
				}
			}
			return src.substring(pos);
		}

		private String parseRule(String src) throws RuntimeException {
			String nameEnd = parseName(src);
			String pos = parseSpace(nameEnd, false);
			int nameLen = nameEnd.length();
			int ruleId = getSymbolId(src, nameLen);
			String name = src.substring(0, nameLen);

			if (!(pos.length() >= 3 && pos.charAt(0) == ':' && pos.charAt(1) == ':' && pos.charAt(2) == '=')) {
				throw new RuntimeException("expecting ::= at " + pos);
			}
			pos = parseSpace(pos.substring(3), true);

			pos = parseAlternates(pos, name, ruleId, false);

			if (!pos.isEmpty()) {
				char firstChar = pos.charAt(0);
				if (firstChar == '\r') {
					pos = pos.substring((pos.length() > 1 && pos.charAt(1) == '\n') ? 2 : 1);
				}
				else if (firstChar == '\n') {
					pos = pos.substring(1);
				}
				else {
					throw new RuntimeException("expecting newline or end at " + pos);
				}
			}
			return parseSpace(pos, true);
		}

		private String parseName(String src) throws RuntimeException {
			int pos = 0;
			while (pos < src.length() && isWordChar(src.charAt(pos))) {
				pos++;
			}
			if (pos == 0) {
				throw new RuntimeException("expecting name at " + src);
			}
			return src.substring(pos);
		}

		private boolean isWordChar(char c) {
			return 'a' <= c && c <= 'z'
					|| 'A' <= c && c <= 'Z'
					|| c == '-'
					|| ('0' <= c && c <= '9');
		}

		private int getSymbolId(String src, int len) {
			int nextId = symbolIds.size();
			String key = src.substring(0, len);
			if (!symbolIds.containsKey(key)) {
				symbolIds.put(key, nextId);
			}
			return symbolIds.get(key);
		}

		private String parseAlternates(String src, String ruleName, int ruleId, boolean isNested) {

			ArrayList<llama_grammar_element> rule = new ArrayList<>();
			String pos = parseSequence(src, ruleName, rule, isNested);

			while (!pos.isEmpty() && pos.charAt(0) == '|') {
				rule.add(new llama_grammar_element(LLAMA_GRETYPE_ALT, 0));
				pos = parseSpace(pos.substring(1), true);
				pos = parseSequence(pos, ruleName, rule, isNested);
			}

			rule.add(new llama_grammar_element(LLAMA_GRETYPE_END, 0));
			addRule(ruleId, rule);

			return pos;
		}

		public void addRule(int ruleId, List<llama_grammar_element> rule) {
			// resize the rules ArrayList if necessary
			while (rules.size() <= ruleId) {
				rules.add(null);
			}
			rules.set(ruleId, rule);
		}

		private String parseSequence(String src, String ruleName, List<llama_grammar_element> outElements, boolean isNested) {
			int lastSymStart = outElements.size();
			String pos = src;
			while (!pos.isEmpty()) {
				char currentChar = pos.charAt(0);
				if (currentChar == '"') { // literal string
					pos = pos.substring(1);
					lastSymStart = outElements.size();
					while (pos.charAt(0) != '"') {
						Pair<Integer, String> charPair = parseChar(pos);
						pos = charPair.b;
						outElements.add(new llama_grammar_element(LLAMA_GRETYPE_CHAR, charPair.a));
					}
					pos = parseSpace(pos.substring(1), isNested);
				}
				else if (currentChar == '[') { // char range(s)
					pos = pos.substring(1);
					int startType = LLAMA_GRETYPE_CHAR;
					if (pos.charAt(0) == '^') {
						pos = pos.substring(1);
						startType = LLAMA_GRETYPE_CHAR_NOT;
					}
					lastSymStart = outElements.size();
					while (pos.charAt(0) != ']') {
						Pair<Integer, String> charPair = parseChar(pos);
						pos = charPair.b;
						int type = lastSymStart < outElements.size() ? LLAMA_GRETYPE_CHAR_ALT : startType;

						outElements.add(new llama_grammar_element(type, charPair.a));
						if (pos.charAt(0) == '-' && pos.charAt(1) != ']') {
							Pair<Integer, String> endCharPair = parseChar(pos.substring(1));
							pos = endCharPair.b;
							outElements.add(new llama_grammar_element(LLAMA_GRETYPE_CHAR_RNG_UPPER, endCharPair.a));
						}
					}
					pos = parseSpace(pos.substring(1), isNested);
				}
				else if (isWordChar(currentChar)) { // rule reference
					String nameEnd = parseName(pos);
					int refRuleId = getSymbolId(pos, nameEnd.length() - pos.length());
					pos = parseSpace(nameEnd, isNested);
					lastSymStart = outElements.size();
					outElements.add(new llama_grammar_element(LLAMA_GRETYPE_RULE_REF, (char) refRuleId));
				}
				else if (currentChar == '(') { // grouping
					pos = parseSpace(pos.substring(1), true);
					int subRuleId = generateSymbolId(ruleName);
					pos = parseAlternates(pos, ruleName, subRuleId, true);
					lastSymStart = outElements.size();
					outElements.add(new llama_grammar_element(LLAMA_GRETYPE_RULE_REF, (char) subRuleId));
					if (pos.charAt(0) != ')') {
						throw new RuntimeException("expecting ')' at " + pos);
					}
					pos = parseSpace(pos.substring(1), isNested);
				}
				else if (currentChar == '*' || currentChar == '+' || currentChar == '?') {
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
					List<llama_grammar_element> subRule = new ArrayList<>(outElements.subList(lastSymStart, outElements.size()));
					if (currentChar == '*' || currentChar == '+') {
						// Cause generated rule to recurse
						subRule.add(new llama_grammar_element(LLAMA_GRETYPE_RULE_REF, (char) subRuleId));
					}
					// Mark start of alternate def
					subRule.add(new llama_grammar_element(LLAMA_GRETYPE_ALT, (char) 0));
					if (currentChar == '+') {
						// Add preceding symbol as alternate only for '+' (otherwise empty)
						subRule.addAll(outElements.subList(lastSymStart, outElements.size()));
					}
					subRule.add(new llama_grammar_element(LLAMA_GRETYPE_END, (char) 0));
					addRule(subRuleId, subRule);

					// In original rule, replace previous symbol with reference to generated rule
					outElements.subList(lastSymStart, outElements.size()).clear();
					outElements.add(new llama_grammar_element(LLAMA_GRETYPE_RULE_REF, (char) subRuleId));

					pos = parseSpace(pos.substring(1), isNested);
				}
				else {
					break;
				}
			}
			return pos;
		}

		private Pair<Integer, String> parseChar(String src) {
			if (src.isEmpty()) {
				throw new RuntimeException("unexpected end of input");
			}
			char firstChar = src.charAt(0);
			if (firstChar == '\\') {
				switch (src.charAt(1)) {
					case 'x':
						return parseHex(src.substring(2), 2);
					case 'u':
						return parseHex(src.substring(2), 4);
					case 'U':
						return parseHex(src.substring(2), 8);
					case 't':
						return new Pair<>((int) '\t', src.substring(2));
					case 'r':
						return new Pair<>((int) '\r', src.substring(2));
					case 'n':
						return new Pair<>((int) '\n', src.substring(2));
					case '\\':
					case '"':
					case '[':
					case ']':
						return new Pair<>((int) src.charAt(1), src.substring(2));
					default:
						throw new RuntimeException("unknown escape at " + src);
				}
			}
			else {
				return decodeUtf8(src);
			}
		}

		private Pair<Integer, String> parseHex(String src, int size) {
			int pos = 0;
			int end = size;
			int value = 0;
			while (pos < end && pos < src.length()) {
				value <<= 4;
				char c = src.charAt(pos);
				if ('a' <= c && c <= 'f') {
					value += c - 'a' + 10;
				}
				else if ('A' <= c && c <= 'F') {
					value += c - 'A' + 10;
				}
				else if ('0' <= c && c <= '9') {
					value += c - '0';
				}
				else {
					break;
				}
				pos++;
			}
			if (pos != end) {
				throw new RuntimeException("expecting " + size + " hex chars at " + src);
			}
			return new Pair<>(value, src.substring(pos));
		}

		private Pair<Integer, String> decodeUtf8(String src) {
			int[] lookup = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
			byte firstByte = (byte) src.charAt(0);
			int highBits = firstByte >> 4;
			int len = lookup[highBits];
			int mask = (1 << (8 - len)) - 1;
			int value = firstByte & mask;
			int end = len;
			int pos = 1;
			while (pos < end && pos < src.length()) {
				value = (value << 6) + (src.charAt(pos) & 0x3F);
				pos++;
			}
			return new Pair<>(value, src.substring(pos));
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
				List<llama_grammar_element> rule = rules.get(i);
				appendRule(grammarBuilder, symbolIdNames, i, rule);
			}
			string = grammarBuilder.toString();
			return string;
		}

		private void appendRule(StringBuilder builder, Map<Integer, String> symbolIdNames, int ruleId, List<llama_grammar_element> rule) {
			if (rule.isEmpty() || rule.get(rule.size() - 1).type != LLAMA_GRETYPE_END) {
				throw new RuntimeException("Malformed rule, does not end with LLAMA_GRETYPE_END: " + ruleId);
			}
			builder.append(symbolIdNames.get(ruleId)).append(" ::= ");
			for (int i = 0, end = rule.size() - 1; i < end; i++) {
				llama_grammar_element elem = rule.get(i);
				switch (elem.type) {
					case LLAMA_GRETYPE_END:
						throw new RuntimeException("Unexpected end of rule: " + ruleId + "," + i);
					case LLAMA_GRETYPE_ALT:
						builder.append("| ");
						break;
					case LLAMA_GRETYPE_RULE_REF:
						builder.append(symbolIdNames.get(elem.value)).append(" ");
						break;
					case LLAMA_GRETYPE_CHAR:
						builder.append("[");
						printGrammarChar(builder, elem.value);
						break;
					case LLAMA_GRETYPE_CHAR_NOT:
						builder.append("[^");
						printGrammarChar(builder, elem.value);
						break;
					case LLAMA_GRETYPE_CHAR_RNG_UPPER:
						if (i == 0 || !isCharElement(rule.get(i - 1))) {
							throw new RuntimeException(
									"LLAMA_GRETYPE_CHAR_RNG_UPPER without preceding char: " + ruleId + "," + i);
						}
						builder.append("-");
						printGrammarChar(builder, elem.value);
						break;
					case LLAMA_GRETYPE_CHAR_ALT:
						if (i == 0 || !isCharElement(rule.get(i - 1))) {
							throw new RuntimeException(
									"LLAMA_GRETYPE_CHAR_ALT without preceding char: " + ruleId + "," + i);
						}
						printGrammarChar(builder, elem.value);
						break;
					default:
						throw new RuntimeException("Unknown type: " + elem.type);
				}
				if (isCharElement(elem)) {
					switch (rule.get(i + 1).type) {
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

		private boolean isCharElement(llama_grammar_element element) {
			switch (element.type) {
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

	private static final class Pair<A, B> {
		private final A a;
		private final B b;

		private Pair(A a, B b) {
			this.a = a;
			this.b = b;
		}
	}
}
