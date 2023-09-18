#include "llama.h"
#include "jllama.h"
#include "common.h"
#include "build-info.h"
#include "grammar-parser.h"

#include <cstddef>
#include <iostream>
#include <string>
#include <locale>
#include <codecvt>
#include <mutex>

// classes
static jclass c_llama_model = 0;
static jclass c_llama_iterator = 0;
static jclass c_model_params = 0;
static jclass c_infer_params = 0;
static jclass c_string = 0;
static jclass c_map = 0;
static jclass c_set = 0;
static jclass c_entry = 0;
static jclass c_iterator = 0;
static jclass c_integer = 0;
static jclass c_float = 0;
static jclass c_log_level = 0;
static jclass c_biconsumer = 0;
static jclass c_llama_error = 0;
static jclass c_error_oom = 0;

// methods
static jmethodID m_get_bytes = 0;
static jmethodID m_entry_set = 0;
static jmethodID m_set_iterator = 0;
static jmethodID m_iterator_has_next = 0;
static jmethodID m_iterator_next = 0;
static jmethodID m_entry_key = 0;
static jmethodID m_entry_value = 0;
static jmethodID m_int_value = 0;
static jmethodID m_float_value = 0;
static jmethodID m_biconsumer_accept = 0;

// fields
static jfieldID f_model_pointer = 0;
// iterator
static jfieldID f_iter_has_next = 0;
static jfieldID f_iter_n_generated = 0;
static jfieldID f_iter_token_index = 0;
// inference parameters
static jfieldID f_n_predict = 0;
static jfieldID f_n_keep = 0;
static jfieldID f_n_probs = 0;
static jfieldID f_logit_bias = 0;
static jfieldID f_top_k = 0;
static jfieldID f_top_p = 0;
static jfieldID f_tfs_z = 0;
static jfieldID f_typical_p = 0;
static jfieldID f_temperature = 0;
static jfieldID f_repeat_penalty = 0;
static jfieldID f_repeat_last_n = 0;
static jfieldID f_frequency_penalty = 0;
static jfieldID f_presence_penalty = 0;
static jfieldID f_penalize_nl = 0;
static jfieldID f_ignore_eos = 0;
static jfieldID f_mirostat = 0;
static jfieldID f_mirostat_tau = 0;
static jfieldID f_mirostat_eta = 0;
static jfieldID f_beam_search = 0;
static jfieldID f_n_beams = 0;
static jfieldID f_grammar = 0;
static jfieldID f_antiprompt = 0;
static jfieldID f_infer_seed = 0;
// model parameters
static jfieldID f_n_threads = 0;
static jfieldID f_model_seed = 0;
static jfieldID f_n_ctx = 0;
static jfieldID f_n_batch = 0;
static jfieldID f_n_gpu_layers = 0;
static jfieldID f_main_gpu = 0;
static jfieldID f_tensor_split = 0;
static jfieldID f_rope_freq_base = 0;
static jfieldID f_rope_freq_scale = 0;
static jfieldID f_low_vram = 0;
static jfieldID f_mul_mat_q = 0;
static jfieldID f_f16_kv = 0;
static jfieldID f_logits_all = 0;
static jfieldID f_vocab_only = 0;
static jfieldID f_use_mmap = 0;
static jfieldID f_use_mlock = 0;
static jfieldID f_embedding = 0;
static jfieldID f_lora_adapter = 0;
static jfieldID f_lora_base = 0;
static jfieldID f_hellaswag = 0;
static jfieldID f_hellaswag_tasks = 0;
static jfieldID f_memory_f16 = 0;
static jfieldID f_mem_test = 0;
static jfieldID f_numa = 0;
static jfieldID f_verbose_prompt = 0;
// log level
static jfieldID f_log_level_debug = 0;
static jfieldID f_log_level_info = 0;
static jfieldID f_log_level_warn = 0;
static jfieldID f_log_level_error = 0;
// objects
static jobject o_log_level_debug = 0;
static jobject o_log_level_info = 0;
static jobject o_log_level_warn = 0;
static jobject o_log_level_error = 0;

static JavaVM* g_vm = nullptr;
static jobject g_log_callback = nullptr;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved)
{
    JNIEnv *env = 0;

    if (JNI_OK != vm->GetEnv((void **)&env, JNI_VERSION_1_1))
    {
        goto error;
    }

    // find classes
    c_llama_model = env->FindClass("de/kherud/llama/LlamaModel");
    c_llama_iterator = env->FindClass("de/kherud/llama/LlamaModel$LlamaIterator");
    c_infer_params = env->FindClass("de/kherud/llama/InferenceParameters");
    c_model_params = env->FindClass("de/kherud/llama/ModelParameters");
    c_string = env->FindClass("java/lang/String");
    c_map = env->FindClass("java/util/Map");
    c_set = env->FindClass("java/util/Set");
    c_entry = env->FindClass("java/util/Map$Entry");
    c_iterator = env->FindClass("java/util/Iterator");
    c_integer = env->FindClass("java/lang/Integer");
    c_float = env->FindClass("java/lang/Float");
    c_log_level = env->FindClass("de/kherud/llama/LogLevel");
    c_biconsumer = env->FindClass("java/util/function/BiConsumer");
    c_llama_error = env->FindClass("de/kherud/llama/LlamaException");
    c_error_oom = env->FindClass("java/lang/OutOfMemoryError");

    if (!(c_llama_model && c_llama_iterator && c_infer_params && c_model_params && c_string && c_map && c_set && c_entry && c_iterator && c_integer && c_float && c_log_level && c_biconsumer && c_llama_error && c_error_oom))
    {
        goto error;
    }

    // create references
    c_llama_model = (jclass)env->NewWeakGlobalRef(c_llama_model);
    c_llama_iterator = (jclass)env->NewWeakGlobalRef(c_llama_iterator);
    c_infer_params = (jclass)env->NewWeakGlobalRef(c_infer_params);
    c_model_params = (jclass)env->NewWeakGlobalRef(c_model_params);
    c_string = (jclass)env->NewWeakGlobalRef(c_string);
    c_map = (jclass)env->NewWeakGlobalRef(c_map);
    c_set = (jclass)env->NewWeakGlobalRef(c_set);
    c_entry = (jclass)env->NewWeakGlobalRef(c_entry);
    c_iterator = (jclass)env->NewWeakGlobalRef(c_iterator);
    c_integer = (jclass)env->NewWeakGlobalRef(c_integer);
    c_float = (jclass)env->NewWeakGlobalRef(c_float);
    c_log_level = (jclass)env->NewWeakGlobalRef(c_log_level);
    c_biconsumer = (jclass)env->NewWeakGlobalRef(c_biconsumer);
    c_llama_error = (jclass)env->NewWeakGlobalRef(c_llama_error);
    c_error_oom = (jclass)env->NewWeakGlobalRef(c_error_oom);

    // find methods
    m_get_bytes = env->GetMethodID(c_string, "getBytes", "(Ljava/lang/String;)[B");
    m_entry_set = env->GetMethodID(c_map, "entrySet", "()Ljava/util/Set;");
    m_set_iterator = env->GetMethodID(c_set, "iterator", "()Ljava/util/Iterator;");
    m_iterator_has_next = env->GetMethodID(c_iterator, "hasNext", "()Z");
    m_iterator_next = env->GetMethodID(c_iterator, "next", "()Ljava/lang/Object;");
    m_entry_key = env->GetMethodID(c_entry, "getKey", "()Ljava/lang/Object;");
    m_entry_value = env->GetMethodID(c_entry, "getValue", "()Ljava/lang/Object;");
    m_int_value = env->GetMethodID(c_integer, "intValue", "()I");
    m_float_value = env->GetMethodID(c_float, "floatValue", "()F");
    m_biconsumer_accept = env->GetMethodID(c_biconsumer, "accept", "(Ljava/lang/Object;Ljava/lang/Object;)V");

    if (!(m_get_bytes && m_entry_set && m_set_iterator && m_iterator_has_next && m_iterator_next && m_entry_key && m_entry_value && m_int_value && m_float_value && m_biconsumer_accept))
    {
        goto error;
    }

    // find fields
    f_model_pointer = env->GetFieldID(c_llama_model, "ctx", "J");
    f_iter_has_next = env->GetFieldID(c_llama_iterator, "hasNext", "Z");
    f_iter_n_generated = env->GetFieldID(c_llama_iterator, "generatedCount", "J");
    f_iter_token_index = env->GetFieldID(c_llama_iterator, "tokenIndex", "J");

    f_n_predict = env->GetFieldID(c_infer_params, "nPredict", "I");
    f_n_keep = env->GetFieldID(c_infer_params, "nKeep", "I");
    f_n_probs = env->GetFieldID(c_infer_params, "nProbs", "I");
    f_logit_bias = env->GetFieldID(c_infer_params, "logitBias", "Ljava/util/Map;");
    f_top_k = env->GetFieldID(c_infer_params, "topK", "I");
    f_top_p = env->GetFieldID(c_infer_params, "topP", "F");
    f_tfs_z = env->GetFieldID(c_infer_params, "tfsZ", "F");
    f_typical_p = env->GetFieldID(c_infer_params, "typicalP", "F");
    f_temperature = env->GetFieldID(c_infer_params, "temperature", "F");
    f_repeat_penalty = env->GetFieldID(c_infer_params, "repeatPenalty", "F");
    f_repeat_last_n = env->GetFieldID(c_infer_params, "repeatLastN", "I");
    f_frequency_penalty = env->GetFieldID(c_infer_params, "frequencyPenalty", "F");
    f_presence_penalty = env->GetFieldID(c_infer_params, "presencePenalty", "F");
    f_penalize_nl = env->GetFieldID(c_infer_params, "penalizeNL", "Z");
    f_ignore_eos = env->GetFieldID(c_infer_params, "ignoreEos", "Z");
    f_mirostat = env->GetFieldID(c_infer_params, "mirostat", "I");
    f_mirostat_tau = env->GetFieldID(c_infer_params, "mirostatTau", "F");
    f_mirostat_eta = env->GetFieldID(c_infer_params, "mirostatEta", "F");
    f_beam_search = env->GetFieldID(c_infer_params, "beamSearch", "Z");
    f_n_beams = env->GetFieldID(c_infer_params, "nBeams", "I");
    f_grammar = env->GetFieldID(c_infer_params, "grammar", "Ljava/lang/String;");
    f_antiprompt = env->GetFieldID(c_infer_params, "antiprompt", "[Ljava/lang/String;");
    f_infer_seed = env->GetFieldID(c_infer_params, "seed", "I");

    f_n_threads = env->GetFieldID(c_model_params, "nThreads", "I");
    f_model_seed = env->GetFieldID(c_model_params, "seed", "I");
    f_n_ctx = env->GetFieldID(c_model_params, "nCtx", "I");
    f_n_batch = env->GetFieldID(c_model_params, "nBatch", "I");
    f_n_gpu_layers = env->GetFieldID(c_model_params, "nGpuLayers", "I");
    f_main_gpu = env->GetFieldID(c_model_params, "mainGpu", "I");
    f_tensor_split = env->GetFieldID(c_model_params, "tensorSplit", "[F");
    f_rope_freq_base = env->GetFieldID(c_model_params, "ropeFreqBase", "F");
    f_rope_freq_scale = env->GetFieldID(c_model_params, "ropeFreqScale", "F");
    f_low_vram = env->GetFieldID(c_model_params, "lowVram", "Z");
    f_mul_mat_q = env->GetFieldID(c_model_params, "mulMatQ", "Z");
    f_f16_kv = env->GetFieldID(c_model_params, "f16Kv", "Z");
    f_logits_all = env->GetFieldID(c_model_params, "logitsAll", "Z");
    f_vocab_only = env->GetFieldID(c_model_params, "vocabOnly", "Z");
    f_use_mmap = env->GetFieldID(c_model_params, "useMmap", "Z");
    f_use_mlock = env->GetFieldID(c_model_params, "useMlock", "Z");
    f_embedding = env->GetFieldID(c_model_params, "embedding", "Z");
    f_lora_adapter = env->GetFieldID(c_model_params, "loraAdapter", "Ljava/lang/String;");
    f_lora_base = env->GetFieldID(c_model_params, "loraBase", "Ljava/lang/String;");
    f_hellaswag = env->GetFieldID(c_model_params, "hellaswag", "Z");
    f_hellaswag_tasks = env->GetFieldID(c_model_params, "hellaswagTasks", "S");
    f_memory_f16 = env->GetFieldID(c_model_params, "memoryF16", "Z");
    f_mem_test = env->GetFieldID(c_model_params, "memTest", "Z");
    f_numa = env->GetFieldID(c_model_params, "numa", "Z");
    f_verbose_prompt = env->GetFieldID(c_model_params, "verbosePrompt", "Z");

    if (!(f_model_pointer && f_iter_has_next && f_iter_n_generated && f_iter_token_index))
    {
        goto error;
    }
    if (!(f_n_predict && f_n_keep && f_n_probs && f_logit_bias && f_top_k && f_top_p && f_tfs_z && f_typical_p && f_temperature && f_repeat_penalty && f_repeat_last_n && f_frequency_penalty && f_presence_penalty && f_penalize_nl && f_ignore_eos && f_mirostat && f_mirostat_tau && f_mirostat_eta && f_beam_search && f_n_beams && f_grammar && f_antiprompt && f_infer_seed))
    {
        goto error;
    }
    if (!(f_n_threads && f_model_seed && f_n_ctx && f_n_batch && f_n_gpu_layers && f_main_gpu && f_tensor_split && f_rope_freq_base && f_rope_freq_scale && f_low_vram && f_mul_mat_q && f_f16_kv && f_logits_all && f_vocab_only && f_use_mmap && f_use_mlock && f_embedding && f_lora_adapter && f_lora_base && f_hellaswag && f_hellaswag_tasks && f_memory_f16 && f_mem_test && f_numa && f_verbose_prompt))
    {
        goto error;
    }

	f_log_level_debug = env->GetStaticFieldID(c_log_level, "DEBUG", "Lde/kherud/llama/LogLevel;");
	f_log_level_info = env->GetStaticFieldID(c_log_level, "INFO", "Lde/kherud/llama/LogLevel;");
	f_log_level_warn = env->GetStaticFieldID(c_log_level, "WARN", "Lde/kherud/llama/LogLevel;");
	f_log_level_error = env->GetStaticFieldID(c_log_level, "ERROR", "Lde/kherud/llama/LogLevel;");

	if (!(f_log_level_debug && f_log_level_info && f_log_level_warn && f_log_level_error))
	{
		goto error;
	}

	o_log_level_debug = env->GetStaticObjectField(c_log_level, f_log_level_debug);
    o_log_level_info = env->GetStaticObjectField(c_log_level, f_log_level_info);
    o_log_level_warn = env->GetStaticObjectField(c_log_level, f_log_level_warn);
    o_log_level_error = env->GetStaticObjectField(c_log_level, f_log_level_error);

    if (!(o_log_level_debug && o_log_level_info && o_log_level_warn && o_log_level_error))
	{
		goto error;
	}

    if (env->ExceptionCheck())
    {
        env->ExceptionDescribe();
        goto error;
    }

    goto success;

error:
    return JNI_ERR;

success:
    return JNI_VERSION_1_1;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved)
{
    JNIEnv *env = 0;

    if (JNI_OK != vm->GetEnv((void **)&env, JNI_VERSION_1_1))
        return;

    env->DeleteWeakGlobalRef(c_llama_model);
    env->DeleteWeakGlobalRef(c_llama_iterator);
    env->DeleteWeakGlobalRef(c_infer_params);
    env->DeleteWeakGlobalRef(c_model_params);
    env->DeleteWeakGlobalRef(c_string);
    env->DeleteWeakGlobalRef(c_map);
    env->DeleteWeakGlobalRef(c_set);
    env->DeleteWeakGlobalRef(c_entry);
    env->DeleteWeakGlobalRef(c_iterator);
    env->DeleteWeakGlobalRef(c_integer);
    env->DeleteWeakGlobalRef(c_float);
    env->DeleteWeakGlobalRef(c_log_level);
    env->DeleteWeakGlobalRef(c_biconsumer);
    env->DeleteWeakGlobalRef(c_llama_error);
    env->DeleteWeakGlobalRef(c_error_oom);
}

static void jllama_log_callback(enum llama_log_level level, const char * text, void * user_data) {
    JNIEnv* env;
    g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_2);

    jobject java_log_level;
    switch (level) {
		case LLAMA_LOG_LEVEL_ERROR: java_log_level = o_log_level_error; break;
        case LLAMA_LOG_LEVEL_WARN: java_log_level = o_log_level_warn; break;
        case LLAMA_LOG_LEVEL_INFO: java_log_level = o_log_level_info; break;
        default: java_log_level = o_log_level_debug; break;
    }
    jstring java_text = env->NewStringUTF(text);

    env->CallVoidMethod(g_log_callback, m_biconsumer_accept, java_log_level, java_text);

    env->DeleteLocalRef(java_log_level);
    env->DeleteLocalRef(java_text);
}

static std::string parse_jstring(JNIEnv *env, jstring java_string)
{
    const jbyteArray string_bytes = (jbyteArray)env->CallObjectMethod(java_string, m_get_bytes, env->NewStringUTF("UTF-8"));

    size_t length = (size_t)env->GetArrayLength(string_bytes);
    jbyte *byte_elements = env->GetByteArrayElements(string_bytes, nullptr);

    std::string string = std::string((char *)byte_elements, length);
    env->ReleaseByteArrayElements(string_bytes, byte_elements, JNI_ABORT);

    env->DeleteLocalRef(string_bytes);
    return string;
}

static int parse_jinteger(JNIEnv *env, jobject java_integer)
{
    if (!java_integer)
        return 0;
    return env->CallIntMethod(java_integer, m_int_value);
}

static float parse_jfloat(JNIEnv *env, jobject java_float)
{
    if (!java_float)
        return 0;
    return env->CallFloatMethod(java_float, m_float_value);
}

static jstring parse_utf16_string(JNIEnv *env, std::string string)
{
    // this only works correctly on platforms where wchar_t is 16 bits
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wstring = converter.from_bytes(string);
    std::u16string string_utf16(wstring.begin(), wstring.end());
    return env->NewString((jchar *)string_utf16.data(), string_utf16.size());
}

// completion token output with probabilities
struct completion_token_output
{
    struct token_prob
    {
        llama_token tok;
        float prob;
    };

    std::vector<token_prob> probs;
    llama_token tok;
};

static size_t common_part(const std::vector<llama_token> &a, const std::vector<llama_token> &b)
{
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++)
    {
    }
    return i;
}

enum stop_type
{
    STOP_FULL,
    STOP_PARTIAL,
};

static bool ends_with(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop,
                                       const std::string &text)
{
    if (!text.empty() && !stop.empty())
    {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--)
        {
            if (stop[char_index] == text_last_char)
            {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial))
                {
                    return text.size() - char_index - 1;
                }
            }
        }
    }
    return std::string::npos;
}

template <class Iter>
static std::string tokens_to_str(llama_context *ctx, Iter begin, Iter end)
{
    std::string ret;
    for (; begin != end; ++begin)
    {
        ret += llama_token_to_piece(ctx, *begin);
    }
    return ret;
}

struct jllama_context
{
    bool has_next_token = false;
    std::string generated_text;
    std::vector<completion_token_output> generated_token_probs;

    size_t num_prompt_tokens = 0;
    size_t num_tokens_predicted = 0;
    size_t n_past = 0;
    size_t n_remain = 0;

    std::string prompt;
    std::vector<llama_token> embd;
    std::vector<llama_token> last_n_tokens;

    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    gpt_params params;

    grammar_parser::parse_state parsed_grammar;
    llama_grammar *grammar = nullptr;

    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;
    std::string stopping_word;
    int32_t multibyte_pending = 0;

    std::mutex mutex;

    std::unique_lock<std::mutex> lock()
    {
        return std::unique_lock<std::mutex>(mutex);
    }

    ~jllama_context()
    {
        if (ctx)
        {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model)
        {
            llama_free_model(model);
            model = nullptr;
        }
    }

    void rewind()
    {
        params.antiprompt.clear();
        params.grammar.clear();
        num_prompt_tokens = 0;
        num_tokens_predicted = 0;
        generated_text = "";
        generated_text.reserve(params.n_ctx);
        generated_token_probs.clear();
        truncated = false;
        stopped_eos = false;
        stopped_word = false;
        stopped_limit = false;
        stopping_word = "";
        multibyte_pending = 0;
        n_remain = 0;
        n_past = 0;

        if (grammar != nullptr)
        {
            llama_grammar_free(grammar);
            grammar = nullptr;
        }
    }

    bool loadModel(const gpt_params &params_)
    {
        params = params_;
        std::tie(model, ctx) = llama_init_from_gpt_params(params);
        if (model == nullptr)
        {
            // LOG_ERROR("unable to load model", {{"model", params_.model}});
            return false;
        }

        last_n_tokens.resize(params.n_ctx);
        std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
        return true;
    }

    std::vector<llama_token> tokenize(std::string prompt, bool add_bos) const
    {
        // If `add_bos` is true, we only add BOS, when json_prompt is a string,
        // or the first element of the json_prompt array is a string.
        return ::llama_tokenize(ctx, prompt, add_bos);
    }

    bool loadGrammar()
    {
        if (!params.grammar.empty())
        {
            parsed_grammar = grammar_parser::parse(params.grammar.c_str());
            // will be empty (default) if there are parse errors
            if (parsed_grammar.rules.empty())
            {
                // LOG_ERROR("grammar parse error", {{"grammar", params.grammar}});
                return false;
            }
            grammar_parser::print_grammar(stderr, parsed_grammar);

            {
                auto it = params.logit_bias.find(llama_token_eos(ctx));
                if (it != params.logit_bias.end() && it->second == -INFINITY)
                {
                    // LOG_WARNING("EOS token is disabled, which will cause most grammars to fail", {});
                }
            }

            std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
            grammar = llama_grammar_init(
                grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
        }
        return true;
    }

    void loadPrompt()
    {
        auto prompt_tokens = tokenize(prompt, true); // always add BOS

        num_prompt_tokens = prompt_tokens.size();

        if (params.n_keep < 0)
        {
            params.n_keep = (int)num_prompt_tokens;
        }
        params.n_keep = std::min(params.n_ctx - 4, params.n_keep);

        // if input prompt is too big, truncate like normal
        if (num_prompt_tokens >= (size_t)params.n_ctx)
        {
            const int n_left = (params.n_ctx - params.n_keep) / 2;
            std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);
            const int erased_blocks = (num_prompt_tokens - params.n_keep - n_left - 1) / n_left;
            new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + params.n_keep + erased_blocks * n_left, prompt_tokens.end());
            std::copy(prompt_tokens.end() - params.n_ctx, prompt_tokens.end(), last_n_tokens.begin());

            // LOG_VERBOSE("input truncated", {
            //                                    {"n_ctx", params.n_ctx},
            //                                    {"n_keep", params.n_keep},
            //                                    {"n_left", n_left},
            //                                    {"new_tokens", tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend())},
            //                                });

            truncated = true;
            prompt_tokens = new_tokens;
        }
        else
        {
            const size_t ps = num_prompt_tokens;
            std::fill(last_n_tokens.begin(), last_n_tokens.end() - ps, 0);
            std::copy(prompt_tokens.begin(), prompt_tokens.end(), last_n_tokens.end() - ps);
        }

        // compare the evaluated prompt with the new prompt
        n_past = common_part(embd, prompt_tokens);
        embd = prompt_tokens;
        if (n_past == num_prompt_tokens)
        {
            // we have to evaluate at least 1 token to generate logits.
            n_past--;
        }

        // LOG_VERBOSE("prompt ingested", {
        //                                    {"n_past", n_past},
        //                                    {"cached", tokens_to_str(ctx, embd.cbegin(), embd.cbegin() + n_past)},
        //                                    {"to_eval", tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend())},
        //                                });

        has_next_token = true;
    }

    void beginCompletion()
    {
        // number of tokens to keep when resetting context
        n_remain = params.n_predict;
        llama_set_rng_seed(ctx, params.seed);
    }

    completion_token_output nextToken()
    {
        completion_token_output result;
        result.tok = -1;

        if (embd.size() >= (size_t)params.n_ctx)
        {
            // Reset context
            const int n_left = (params.n_ctx - params.n_keep) / 2;

            std::vector<llama_token> new_tokens(embd.begin(), embd.begin() + params.n_keep);
            new_tokens.insert(new_tokens.end(), embd.end() - n_left, embd.end());
            embd = new_tokens;
            n_past = params.n_keep;
            truncated = true;
            // LOG_VERBOSE("input truncated", {
            //                                    {"n_ctx", params.n_ctx},
            //                                    {"n_keep", params.n_keep},
            //                                    {"n_left", n_left},
            //                                    {"new_tokens", tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend())},
            //                                });
        }

        while (n_past < embd.size())
        {
            int n_eval = (int)embd.size() - n_past;
            if (n_eval > params.n_batch)
            {
                n_eval = params.n_batch;
            }
            if (llama_eval(ctx, &embd[n_past], n_eval, n_past, params.n_threads))
            {
                // LOG_ERROR("failed to eval", {
                //                                 {"n_eval", n_eval},
                //                                 {"n_past", n_past},
                //                                 {"n_threads", params.n_threads},
                //                                 {"embd", tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend())},
                //                             });
                has_next_token = false;
                return result;
            }
            n_past += n_eval;
        }

        if (params.n_predict == 0)
        {
            has_next_token = false;
            result.tok = llama_token_eos(ctx);
            return result;
        }

        // out of user input, sample next token
        const float temp = params.temp;
        const int32_t top_k = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
        const float top_p = params.top_p;
        const float tfs_z = params.tfs_z;
        const float typical_p = params.typical_p;
        const int32_t repeat_last_n = params.repeat_last_n < 0 ? params.n_ctx : params.repeat_last_n;
        const float repeat_penalty = params.repeat_penalty;
        const float alpha_presence = params.presence_penalty;
        const float alpha_frequency = params.frequency_penalty;
        const int mirostat = params.mirostat;
        const float mirostat_tau = params.mirostat_tau;
        const float mirostat_eta = params.mirostat_eta;
        const bool penalize_nl = params.penalize_nl;
        const int32_t n_probs = params.n_probs;

        {
            auto *logits = llama_get_logits(ctx);
            auto n_vocab = llama_n_vocab(ctx);

            // Apply params.logit_bias map
            for (const auto &it : params.logit_bias)
            {
                logits[it.first] += it.second;
            }

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++)
            {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }

            llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

            // Apply penalties
            float nl_logit = logits[llama_token_nl(ctx)];
            auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), params.n_ctx);
            llama_sample_repetition_penalty(ctx, &candidates_p,
                                            last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                            last_n_repeat, repeat_penalty);
            llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                                          last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                          last_n_repeat, alpha_frequency, alpha_presence);
            if (!penalize_nl)
            {
                logits[llama_token_nl(ctx)] = nl_logit;
            }

            if (grammar != nullptr)
            {
                llama_sample_grammar(ctx, &candidates_p, grammar);
            }

            if (temp <= 0)
            {
                // Greedy sampling
                result.tok = llama_sample_token_greedy(ctx, &candidates_p);
                if (n_probs > 0)
                {
                    llama_sample_softmax(ctx, &candidates_p);
                }
            }
            else
            {
                if (mirostat == 1)
                {
                    static float mirostat_mu = 2.0f * mirostat_tau;
                    const int mirostat_m = 100;
                    llama_sample_temperature(ctx, &candidates_p, temp);
                    result.tok = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                }
                else if (mirostat == 2)
                {
                    static float mirostat_mu = 2.0f * mirostat_tau;
                    llama_sample_temperature(ctx, &candidates_p, temp);
                    result.tok = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                }
                else
                {
                    // Temperature sampling
                    size_t min_keep = std::max(1, n_probs);
                    llama_sample_top_k(ctx, &candidates_p, top_k, min_keep);
                    llama_sample_tail_free(ctx, &candidates_p, tfs_z, min_keep);
                    llama_sample_typical(ctx, &candidates_p, typical_p, min_keep);
                    llama_sample_top_p(ctx, &candidates_p, top_p, min_keep);
                    llama_sample_temperature(ctx, &candidates_p, temp);
                    result.tok = llama_sample_token(ctx, &candidates_p);
                }
            }

            if (grammar != nullptr)
            {
                llama_grammar_accept_token(ctx, grammar, result.tok);
            }

            for (size_t i = 0; i < std::min(candidates_p.size, (size_t)n_probs); ++i)
            {
                result.probs.push_back({candidates_p.data[i].id, candidates_p.data[i].p});
            }

            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(result.tok);
            num_tokens_predicted++;
        }

        // add it to the context
        embd.push_back(result.tok);
        // decrement remaining sampling budget
        --n_remain;

        if (!embd.empty() && embd.back() == llama_token_eos(ctx))
        {
            // stopping_word = llama_token_to_piece(ctx, embd.back());
            has_next_token = false;
            stopped_eos = true;
            // LOG_VERBOSE("eos token found", {});
            return result;
        }

        has_next_token = params.n_predict == -1 || n_remain != 0;
        return result;
    }

    size_t findStoppingStrings(const std::string &text, const size_t last_token_size,
                               const stop_type type)
    {
        size_t stop_pos = std::string::npos;
        for (const std::string &word : params.antiprompt)
        {
            size_t pos;
            if (type == STOP_FULL)
            {
                const size_t tmp = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
                pos = text.find(word, from_pos);
            }
            else
            {
                pos = find_partial_stop_string(word, text);
            }
            if (pos != std::string::npos &&
                (stop_pos == std::string::npos || pos < stop_pos))
            {
                if (type == STOP_FULL)
                {
                    stopping_word = word;
                    stopped_word = true;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }
        return stop_pos;
    }

    completion_token_output doCompletion()
    {
        auto token_with_probs = nextToken();

        const std::string token_text = token_with_probs.tok == -1 ? "" : llama_token_to_piece(ctx, token_with_probs.tok);
        generated_text += token_text;

        if (params.n_probs > 0)
        {
            generated_token_probs.push_back(token_with_probs);
        }

        if (multibyte_pending > 0)
        {
            multibyte_pending -= token_text.size();
        }
        else if (token_text.size() == 1)
        {
            const char c = token_text[0];
            // 2-byte characters: 110xxxxx 10xxxxxx
            if ((c & 0xE0) == 0xC0)
            {
                multibyte_pending = 1;
                // 3-byte characters: 1110xxxx 10xxxxxx 10xxxxxx
            }
            else if ((c & 0xF0) == 0xE0)
            {
                multibyte_pending = 2;
                // 4-byte characters: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
            }
            else if ((c & 0xF8) == 0xF0)
            {
                multibyte_pending = 3;
            }
            else
            {
                multibyte_pending = 0;
            }
        }

        if (multibyte_pending > 0 && !has_next_token)
        {
            has_next_token = true;
            n_remain++;
        }

        if (!has_next_token && n_remain == 0)
        {
            stopped_limit = true;
        }

        // LOG_VERBOSE("next token", {
        //                               {"token", token_with_probs.tok},
        //                               {"token_text", tokens_to_output_formatted_string(ctx, token_with_probs.tok)},
        //                               {"has_next_token", has_next_token},
        //                               {"n_remain", n_remain},
        //                               {"num_tokens_predicted", num_tokens_predicted},
        //                               {"stopped_eos", stopped_eos},
        //                               {"stopped_word", stopped_word},
        //                               {"stopped_limit", stopped_limit},
        //                               {"stopping_word", stopping_word},
        //                           });

        return token_with_probs;
    }

    std::vector<float> getEmbedding()
    {
        static const int n_embd = llama_n_embd(ctx);
        if (!params.embedding)
        {
            // LOG_WARNING("embedding disabled", {
            //                                       {"params.embedding", params.embedding},
            //                                   });
            return std::vector<float>(n_embd, 0.0f);
        }
        const float *data = llama_get_embeddings(ctx);
        std::vector<float> embedding(data, data + n_embd);
        return embedding;
    }
};

static gpt_params parse_model_params(JNIEnv *env, jobject jparams, jstring java_file_path)
{
    gpt_params params;

    params.model = parse_jstring(env, java_file_path);
    params.seed = env->GetIntField(jparams, f_model_seed);
    params.n_threads = env->GetIntField(jparams, f_n_threads);
    params.n_ctx = env->GetIntField(jparams, f_n_ctx);
    params.n_batch = env->GetIntField(jparams, f_n_batch);
    params.n_gpu_layers = env->GetIntField(jparams, f_n_gpu_layers);
    params.main_gpu = env->GetIntField(jparams, f_main_gpu);
    params.rope_freq_base = env->GetFloatField(jparams, f_rope_freq_base);
    params.rope_freq_scale = env->GetFloatField(jparams, f_rope_freq_scale);
    params.hellaswag = env->GetBooleanField(jparams, f_hellaswag);
    params.hellaswag_tasks = env->GetShortField(jparams, f_hellaswag_tasks);
    params.low_vram = env->GetBooleanField(jparams, f_low_vram);
    params.mul_mat_q = env->GetBooleanField(jparams, f_mul_mat_q);
    params.memory_f16 = env->GetBooleanField(jparams, f_memory_f16);
    params.embedding = env->GetBooleanField(jparams, f_embedding);
    params.escape = env->GetIntField(jparams, f_n_predict);
    params.use_mmap = env->GetBooleanField(jparams, f_use_mmap);
    params.use_mlock = env->GetBooleanField(jparams, f_use_mlock);
    params.numa = env->GetBooleanField(jparams, f_numa);
    params.verbose_prompt = env->GetBooleanField(jparams, f_verbose_prompt);

     jstring j_lora_adapter = (jstring)env->GetObjectField(jparams, f_lora_adapter);
     if (j_lora_adapter != nullptr)
     {
         params.lora_adapter = parse_jstring(env, j_lora_adapter);
         env->DeleteLocalRef(j_lora_adapter);
     }
     jstring j_lora_base = (jstring)env->GetObjectField(jparams, f_lora_base);
     if (j_lora_base != nullptr)
     {
         params.lora_base = parse_jstring(env, j_lora_base);
         env->DeleteLocalRef(j_lora_base);
     }

//     jfloatArray j_tensor_split = (jfloatArray)env->GetObjectField(jparams, f_tensor_split);
//     if (j_tensor_split != nullptr)
//     {
// #ifndef GGML_USE_CUBLAS
//         // LOG_WARNING("llama.cpp was compiled without cuBLAS. It is not possible to set a tensor split.\n", {});
// #endif
//         jsize array_length = env->GetArrayLength(j_tensor_split);
//         GGML_ASSERT(array_length <= LLAMA_MAX_DEVICES);
//         float *tensor_split = new float[array_length];
//         env->GetFloatArrayRegion(j_tensor_split, 0, array_length, tensor_split);
//         for (size_t i_device = 0; i_device < LLAMA_MAX_DEVICES; ++i_device)
//         {
//             if (i_device < array_length)
//             {
//                 params.tensor_split[i_device] = tensor_split[i_device];
//             }
//             else
//             {
//                 params.tensor_split[i_device] = 0.0f;
//             }
//         }
//         delete[] tensor_split;
//     }
    //
    // #ifndef LLAMA_SUPPORTS_GPU_OFFLOAD
    //		if (params.n_gpu_layers > 0) {
    //			// LOG_WARNING("Not compiled with GPU offload support, --n-gpu-layers option will be ignored. "
    //			// 			"See main README.md for information on enabling GPU BLAS support",
    //			// 			{{"n_gpu_layers", params.n_gpu_layers}});
    //		}
    // #endif
    //
    // #ifndef GGML_USE_CUBLAS
    //	if (params.low_vram) {
    //		// LOG_WARNING("warning: llama.cpp was compiled without cuBLAS. It is not possible to set lower vram usage.\n", {});
    //	}
    //	if (!params.mul_mat_q) {
    //		// LOG_WARNING("warning: llama.cpp was compiled without cuBLAS. Disabling mul_mat_q kernels has no effect.\n", {});
    //	}
    //	if (params.main_gpu != 0) {
    //		// LOG_WARNING("llama.cpp was compiled without cuBLAS. It is not possible to set a main GPU.", {});
    //	}
    // #endif
    //
    //	// todo: these have to be set in llama_context_params
    //	//  f_logits_all
    //	//  f_vocab_only
    //	//  f_memory_f16
    //	//	f_f16_kv

    return params;
}

static void parse_inference_params(JNIEnv *env, jllama_context *llama, jstring prompt, jobject params)
{
    llama->prompt = parse_jstring(env, prompt);
    llama->params.n_predict = env->GetIntField(params, f_n_predict);
    llama->params.n_keep = env->GetIntField(params, f_n_keep);
    llama->params.n_probs = env->GetIntField(params, f_n_probs);
    llama->params.top_k = env->GetIntField(params, f_top_k);
    llama->params.top_p = env->GetFloatField(params, f_top_p);
    llama->params.tfs_z = env->GetFloatField(params, f_tfs_z);
    llama->params.typical_p = env->GetFloatField(params, f_typical_p);
    llama->params.temp = env->GetFloatField(params, f_temperature);
    llama->params.repeat_penalty = env->GetFloatField(params, f_repeat_penalty);
    llama->params.repeat_last_n = env->GetIntField(params, f_repeat_last_n);
    llama->params.frequency_penalty = env->GetFloatField(params, f_frequency_penalty);
    llama->params.presence_penalty = env->GetFloatField(params, f_presence_penalty);
    llama->params.penalize_nl = env->GetBooleanField(params, f_penalize_nl);
    llama->params.mirostat = env->GetIntField(params, f_mirostat);
    llama->params.mirostat_tau = env->GetFloatField(params, f_mirostat_tau);
    llama->params.mirostat_eta = env->GetFloatField(params, f_mirostat_eta);
    llama->params.seed = env->GetIntField(params, f_infer_seed);

    jstring j_grammar = (jstring)env->GetObjectField(params, f_grammar);
    if (j_grammar != nullptr)
    {
        llama->params.grammar = parse_jstring(env, j_grammar);
        env->DeleteLocalRef(j_grammar);
    }

    llama->params.logit_bias.clear();
    jboolean ignore_eos = env->GetBooleanField(params, f_ignore_eos);
    if (ignore_eos)
    {
        llama->params.logit_bias[llama_token_eos(llama->ctx)] = -INFINITY;
    }

    jobject logit_bias = env->GetObjectField(params, f_logit_bias);
    if (logit_bias != nullptr)
    {
        const int n_vocab = llama_n_vocab(llama->ctx);
        jobject entry_set = env->CallObjectMethod(logit_bias, m_entry_set);
        jobject iterator = env->CallObjectMethod(entry_set, m_set_iterator);
        while (env->CallBooleanMethod(iterator, m_iterator_has_next))
        {
            jobject entry = env->CallObjectMethod(iterator, m_iterator_next);
            jobject key = env->CallObjectMethod(entry, m_entry_key);
            jobject value = env->CallObjectMethod(entry, m_entry_value);

            int tok = parse_jinteger(env, key);
            float bias = parse_jfloat(env, value);
            llama->params.logit_bias[tok] = bias;

            env->DeleteLocalRef(entry);
            env->DeleteLocalRef(key);
            env->DeleteLocalRef(value);
        }
    }

    llama->params.antiprompt.clear();
    jobjectArray antiprompt = (jobjectArray)env->GetObjectField(params, f_antiprompt);
    if (antiprompt != nullptr)
    {
        jsize array_length = env->GetArrayLength(antiprompt);
        for (jsize i = 0; i < array_length; i++)
        {
            jstring java_string = (jstring)env->GetObjectArrayElement(antiprompt, i);
            if (java_string != nullptr)
            {
                std::string string = parse_jstring(env, java_string);
                llama->params.antiprompt.push_back(string);
                env->DeleteLocalRef(java_string);
            }
        }
    }

    // LOG_VERBOSE("completion parameters parsed", format_generation_settings(*llama));
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getSystemInfo(JNIEnv *env, jobject obj)
{
    const char *sys_info = llama_print_system_info();
    return env->NewStringUTF(sys_info);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_loadModel(JNIEnv *env, jobject obj, jstring file_path, jobject jparams)
{
    gpt_params params = parse_model_params(env, jparams, file_path);

    jllama_context *llama = new jllama_context;
    llama_backend_init(false);

    std::cout << params.model << std::endl;
    if (!llama->loadModel(params))
    {
        env->ThrowNew(c_llama_error, "could not load model from given file path");
        return;
    }

    // LOG_INFO("build info", {{"build", BUILD_NUMBER},
    //                             {"commit", BUILD_COMMIT}});
    // LOG_INFO("system info", {
    //                             {"n_threads", params.n_threads},
    //                             {"total_threads", std::thread::hardware_concurrency()},
    //                             {"system_info", llama_print_system_info()},
    //                         });

    env->SetLongField(obj, f_model_pointer, reinterpret_cast<jlong>(llama));
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setupInference(JNIEnv *env, jobject obj, jstring prompt, jobject params)
{
    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
    jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);

    auto lock = llama->lock();

    llama->rewind();

    llama_reset_timings(llama->ctx);

    parse_inference_params(env, llama, prompt, params);

    if (!llama->loadGrammar())
    {
        env->ThrowNew(c_llama_error, "could not load grammar");
    }

    llama->loadPrompt();
    llama->beginCompletion();
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getNext(JNIEnv *env, jobject obj, jobject iter)
{
    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
    jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);

    size_t sent_count = env->GetLongField(iter, f_iter_n_generated);
    size_t sent_token_probs_index = env->GetLongField(iter, f_iter_token_index);

    completion_token_output token_with_probs;
    while (llama->has_next_token)
    {
        token_with_probs = llama->doCompletion();
        if (token_with_probs.tok >= 0 && llama->multibyte_pending <= 0)
        {
            break;
        }
    }
    const std::string token_text = llama_token_to_piece(llama->ctx, token_with_probs.tok);

    size_t pos = std::min(sent_count, llama->generated_text.size());

    const std::string str_test = llama->generated_text.substr(pos);
    bool is_stop_full = false;
    size_t stop_pos = llama->findStoppingStrings(str_test, token_text.size(), STOP_FULL);
    if (stop_pos != std::string::npos)
    {
        is_stop_full = true;
        llama->generated_text.erase(
            llama->generated_text.begin() + pos + stop_pos,
            llama->generated_text.end());
        pos = std::min(sent_count, llama->generated_text.size());
    }
    else
    {
        is_stop_full = false;
        stop_pos = llama->findStoppingStrings(str_test, token_text.size(), STOP_PARTIAL);
    }

    std::string to_send;
    if (stop_pos == std::string::npos || (!llama->has_next_token && !is_stop_full && stop_pos > 0))
    {
        to_send = llama->generated_text.substr(pos, std::string::npos);
        sent_count += to_send.size();
        env->SetLongField(iter, f_iter_n_generated, sent_count);

        std::vector<completion_token_output> probs_output = {};

        if (llama->params.n_probs > 0)
        {
            const std::vector<llama_token> to_send_toks = llama_tokenize(llama->ctx, to_send, false);
            size_t probs_pos = std::min(sent_token_probs_index, llama->generated_token_probs.size());
            size_t probs_stop_pos = std::min(
                sent_token_probs_index + to_send_toks.size(),
                llama->generated_token_probs.size());
            if (probs_pos < probs_stop_pos)
            {
                probs_output = std::vector<completion_token_output>(
                    llama->generated_token_probs.begin() + probs_pos,
                    llama->generated_token_probs.begin() + probs_stop_pos);
            }
            sent_token_probs_index = probs_stop_pos;
            env->SetLongField(iter, f_iter_token_index, sent_token_probs_index);
        }
    }
    else
    {
        to_send = "";
    }

    if (!llama->has_next_token)
    {
        env->SetLongField(iter, f_iter_has_next, false);
    }

    return parse_utf16_string(env, to_send);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getFull(JNIEnv *env, jobject obj, jstring prompt, jobject params)
{
    Java_de_kherud_llama_LlamaModel_setupInference(env, obj, prompt, params);

    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
    jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);

    size_t stop_pos = std::string::npos;

    while (llama->has_next_token)
    {
        const completion_token_output token_with_probs = llama->doCompletion();
        const std::string token_text = token_with_probs.tok == -1 ? "" : llama_token_to_piece(llama->ctx, token_with_probs.tok);

        stop_pos = llama->findStoppingStrings(llama->generated_text,
                                              token_text.size(), STOP_FULL);
    }

    if (stop_pos == std::string::npos)
    {
        stop_pos = llama->findStoppingStrings(llama->generated_text, 0, STOP_PARTIAL);
    }
    if (stop_pos != std::string::npos)
    {
        llama->generated_text.erase(llama->generated_text.begin() + stop_pos, llama->generated_text.end());
    }

    auto probs = llama->generated_token_probs;
    if (llama->params.n_probs > 0 && llama->stopped_word)
    {
        const std::vector<llama_token> stop_word_toks = llama_tokenize(llama->ctx, llama->stopping_word, false);
        probs = std::vector<completion_token_output>(llama->generated_token_probs.begin(), llama->generated_token_probs.end() - stop_word_toks.size());
    }

    llama_print_timings(llama->ctx);

    //    llama->lock().release();
    //    llama->mutex.unlock();

    return parse_utf16_string(env, llama->generated_text);
}

JNIEXPORT jfloatArray JNICALL Java_de_kherud_llama_LlamaModel_embed(JNIEnv *env, jobject obj, jstring java_prompt)
{
    //    auto lock = llama.lock();
    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
    jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);

    llama->rewind();
    llama_reset_timings(llama->ctx);
    llama->prompt = parse_jstring(env, java_prompt);
    llama->params.n_predict = 0;
    llama->loadPrompt();
    llama->beginCompletion();
    llama->doCompletion();

    static const int n_embd = llama_n_embd(llama->ctx);
    //    if (!llama->params.embedding)
    //    {
    //        // LOG_WARNING("embedding disabled", {
    //        //                                       {"params.embedding", params.embedding},
    //        //                                   });
    //        return std::vector<float>(n_embd, 0.0f);
    //    }
    const float *data = llama_get_embeddings(llama->ctx);
    std::vector<float> embedding(data, data + n_embd);

    jfloatArray java_embedding = env->NewFloatArray(embedding.size());
    if (java_embedding == nullptr)
    {
        env->ThrowNew(c_error_oom, "could not allocate embedding");
        return nullptr;
    }

    env->SetFloatArrayRegion(java_embedding, 0, embedding.size(), reinterpret_cast<const jfloat *>(embedding.data()));

    return java_embedding;
}

JNIEXPORT jintArray JNICALL Java_de_kherud_llama_LlamaModel_encode(JNIEnv *env, jobject obj, jstring java_prompt)
{
    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
    jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);

    //    auto lock = llama->lock();

    std::string prompt = parse_jstring(env, java_prompt);
    std::vector<llama_token> tokens = llama->tokenize(prompt, false);

    jintArray java_tokens = env->NewIntArray(tokens.size());
    if (java_tokens == nullptr)
    {
        env->ThrowNew(c_error_oom, "could not allocate tokens");
        return nullptr;
    }

    env->SetIntArrayRegion(java_tokens, 0, tokens.size(), reinterpret_cast<const jint *>(tokens.data()));

    //    lock.release();
    return java_tokens;
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_decode(JNIEnv *env, jobject obj, jintArray java_tokens)
{
    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
    jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
    //    auto lock = llama.lock();

    jsize length = env->GetArrayLength(java_tokens);
    jint *elements = env->GetIntArrayElements(java_tokens, nullptr);
    std::vector<llama_token> tokens(elements, elements + length);
    std::string text = tokens_to_str(llama->ctx, tokens.cbegin(), tokens.cend());

    env->ReleaseIntArrayElements(java_tokens, elements, 0);

    return parse_utf16_string(env, text);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setLogger(JNIEnv * env, jclass clazz, jobject callback) {
	env->GetJavaVM(&g_vm);

	if (g_log_callback != nullptr)
	{
		env->DeleteGlobalRef(g_log_callback);
	}

	g_log_callback = env->NewGlobalRef(callback);

	llama_log_set(jllama_log_callback, nullptr);
}
