#include <cstddef>
#include <iostream>
#include <string>
#include <mutex>

#include "llama.h"
#include "jllama.h"
#include "common.h"
#include "build-info.h"
#include "grammar-parser.h"

// classes
static jclass c_llama_model = 0;
static jclass c_llama_iterator = 0;
static jclass c_model_params = 0;
static jclass c_infer_params = 0;
static jclass c_standard_charsets = 0;
static jclass c_output = 0;
static jclass c_string = 0;
static jclass c_hash_map = 0;
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

// constructors
static jmethodID cc_output = 0;
static jmethodID cc_hash_map = 0;
static jmethodID cc_integer = 0;
static jmethodID cc_float = 0;

// methods
static jmethodID m_get_bytes = 0;
static jmethodID m_entry_set = 0;
static jmethodID m_set_iterator = 0;
static jmethodID m_iterator_has_next = 0;
static jmethodID m_iterator_next = 0;
static jmethodID m_entry_key = 0;
static jmethodID m_entry_value = 0;
static jmethodID m_map_put = 0;
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
static jfieldID f_mul_mat_q = 0;
static jfieldID f_f16_kv = 0;
static jfieldID f_logits_all = 0;
static jfieldID f_vocab_only = 0;
static jfieldID f_use_mmap = 0;
static jfieldID f_use_mlock = 0;
static jfieldID f_embedding = 0;
static jfieldID f_lora_adapter = 0;
static jfieldID f_lora_base = 0;
static jfieldID f_memory_f16 = 0;
static jfieldID f_mem_test = 0;
static jfieldID f_numa = 0;
static jfieldID f_verbose_prompt = 0;
// log level
static jfieldID f_utf_8 = 0;
static jfieldID f_log_level_debug = 0;
static jfieldID f_log_level_info = 0;
static jfieldID f_log_level_warn = 0;
static jfieldID f_log_level_error = 0;
// objects
static jobject o_utf_8 = 0;
static jobject o_log_level_debug = 0;
static jobject o_log_level_info = 0;
static jobject o_log_level_warn = 0;
static jobject o_log_level_error = 0;

static JavaVM *g_vm = nullptr;
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
    c_standard_charsets = env->FindClass("java/nio/charset/StandardCharsets");
    c_output = env->FindClass("de/kherud/llama/LlamaModel$Output");
    c_string = env->FindClass("java/lang/String");
    c_hash_map = env->FindClass("java/util/HashMap");
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

    if (!(c_llama_model && c_llama_iterator && c_infer_params && c_model_params && c_standard_charsets && c_output && c_string && c_hash_map && c_map && c_set && c_entry && c_iterator && c_integer && c_float && c_log_level && c_biconsumer && c_llama_error && c_error_oom))
    {
        goto error;
    }

    // create references
    c_llama_model = (jclass)env->NewGlobalRef(c_llama_model);
    c_llama_iterator = (jclass)env->NewGlobalRef(c_llama_iterator);
    c_infer_params = (jclass)env->NewGlobalRef(c_infer_params);
    c_model_params = (jclass)env->NewGlobalRef(c_model_params);
    c_output = (jclass)env->NewGlobalRef(c_output);
    c_string = (jclass)env->NewGlobalRef(c_string);
    c_hash_map = (jclass)env->NewGlobalRef(c_hash_map);
    c_map = (jclass)env->NewGlobalRef(c_map);
    c_set = (jclass)env->NewGlobalRef(c_set);
    c_entry = (jclass)env->NewGlobalRef(c_entry);
    c_iterator = (jclass)env->NewGlobalRef(c_iterator);
    c_integer = (jclass)env->NewGlobalRef(c_integer);
    c_float = (jclass)env->NewGlobalRef(c_float);
    c_log_level = (jclass)env->NewGlobalRef(c_log_level);
    c_biconsumer = (jclass)env->NewGlobalRef(c_biconsumer);
    c_llama_error = (jclass)env->NewGlobalRef(c_llama_error);
    c_error_oom = (jclass)env->NewGlobalRef(c_error_oom);

  	// find constructors
    cc_output = env->GetMethodID(c_output, "<init>", "(I[BLjava/util/Map;)V");
    cc_hash_map = env->GetMethodID(c_hash_map, "<init>", "()V");
    cc_integer = env->GetMethodID(c_integer, "<init>", "(I)V");
    cc_float = env->GetMethodID(c_float, "<init>", "(F)V");

	if (!(cc_output && cc_hash_map && cc_integer && cc_float))
	{
		goto error;
	}

    // find methods
//    m_get_bytes = env->GetMethodID(c_string, "getBytes", "(Ljava/nio/charset/Charset;)[B");
    m_get_bytes = env->GetMethodID(c_string, "getBytes", "(Ljava/lang/String;)[B");
    m_entry_set = env->GetMethodID(c_map, "entrySet", "()Ljava/util/Set;");
    m_set_iterator = env->GetMethodID(c_set, "iterator", "()Ljava/util/Iterator;");
    m_iterator_has_next = env->GetMethodID(c_iterator, "hasNext", "()Z");
    m_iterator_next = env->GetMethodID(c_iterator, "next", "()Ljava/lang/Object;");
    m_entry_key = env->GetMethodID(c_entry, "getKey", "()Ljava/lang/Object;");
    m_entry_value = env->GetMethodID(c_entry, "getValue", "()Ljava/lang/Object;");
    m_map_put = env->GetMethodID(c_map, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    m_int_value = env->GetMethodID(c_integer, "intValue", "()I");
    m_float_value = env->GetMethodID(c_float, "floatValue", "()F");
    m_biconsumer_accept = env->GetMethodID(c_biconsumer, "accept", "(Ljava/lang/Object;Ljava/lang/Object;)V");

    if (!(m_get_bytes && m_entry_set && m_set_iterator && m_iterator_has_next && m_iterator_next && m_entry_key && m_entry_value && m_map_put && m_int_value && m_float_value && m_biconsumer_accept))
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
    f_penalize_nl = env->GetFieldID(c_infer_params, "penalizeNl", "Z");
    f_ignore_eos = env->GetFieldID(c_infer_params, "ignoreEos", "Z");
    f_mirostat = env->GetFieldID(c_infer_params, "mirostat", "I");
    f_mirostat_tau = env->GetFieldID(c_infer_params, "mirostatTau", "F");
    f_mirostat_eta = env->GetFieldID(c_infer_params, "mirostatEta", "F");
    f_beam_search = env->GetFieldID(c_infer_params, "beamSearch", "Z");
    f_n_beams = env->GetFieldID(c_infer_params, "nBeams", "I");
    f_grammar = env->GetFieldID(c_infer_params, "grammar", "Ljava/lang/String;");
    f_antiprompt = env->GetFieldID(c_infer_params, "antiPrompt", "[Ljava/lang/String;");
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
    f_mul_mat_q = env->GetFieldID(c_model_params, "mulMatQ", "Z");
    f_f16_kv = env->GetFieldID(c_model_params, "f16Kv", "Z");
    f_logits_all = env->GetFieldID(c_model_params, "logitsAll", "Z");
    f_vocab_only = env->GetFieldID(c_model_params, "vocabOnly", "Z");
    f_use_mmap = env->GetFieldID(c_model_params, "useMmap", "Z");
    f_use_mlock = env->GetFieldID(c_model_params, "useMlock", "Z");
    f_embedding = env->GetFieldID(c_model_params, "embedding", "Z");
    f_lora_adapter = env->GetFieldID(c_model_params, "loraAdapter", "Ljava/lang/String;");
    f_lora_base = env->GetFieldID(c_model_params, "loraBase", "Ljava/lang/String;");
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
    if (!(f_n_threads && f_model_seed && f_n_ctx && f_n_batch && f_n_gpu_layers && f_main_gpu && f_tensor_split && f_rope_freq_base && f_rope_freq_scale && f_mul_mat_q && f_f16_kv && f_logits_all && f_vocab_only && f_use_mmap && f_use_mlock && f_embedding && f_lora_adapter && f_lora_base && f_memory_f16 && f_mem_test && f_numa && f_verbose_prompt))
    {
        goto error;
    }

    f_utf_8 = env->GetStaticFieldID(c_standard_charsets, "UTF_8", "Ljava/nio/charset/Charset;");
    f_log_level_debug = env->GetStaticFieldID(c_log_level, "DEBUG", "Lde/kherud/llama/LogLevel;");
    f_log_level_info = env->GetStaticFieldID(c_log_level, "INFO", "Lde/kherud/llama/LogLevel;");
    f_log_level_warn = env->GetStaticFieldID(c_log_level, "WARN", "Lde/kherud/llama/LogLevel;");
    f_log_level_error = env->GetStaticFieldID(c_log_level, "ERROR", "Lde/kherud/llama/LogLevel;");

    if (!(f_utf_8 && f_log_level_debug && f_log_level_info && f_log_level_warn && f_log_level_error))
    {
        goto error;
    }

//    o_utf_8 = env->GetStaticObjectField(c_standard_charsets, f_utf_8);
    o_utf_8 = env->NewStringUTF("UTF-8");
    o_utf_8 = (jclass)env->NewGlobalRef(o_utf_8);
    o_log_level_debug = env->GetStaticObjectField(c_log_level, f_log_level_debug);
    o_log_level_info = env->GetStaticObjectField(c_log_level, f_log_level_info);
    o_log_level_warn = env->GetStaticObjectField(c_log_level, f_log_level_warn);
    o_log_level_error = env->GetStaticObjectField(c_log_level, f_log_level_error);

    if (!(o_utf_8 && o_log_level_debug && o_log_level_info && o_log_level_warn && o_log_level_error))
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

    env->DeleteGlobalRef(c_llama_model);
    env->DeleteGlobalRef(c_llama_iterator);
    env->DeleteGlobalRef(c_infer_params);
    env->DeleteGlobalRef(c_model_params);
    env->DeleteGlobalRef(c_output);
    env->DeleteGlobalRef(c_string);
    env->DeleteGlobalRef(c_hash_map);
    env->DeleteGlobalRef(c_map);
    env->DeleteGlobalRef(c_set);
    env->DeleteGlobalRef(c_entry);
    env->DeleteGlobalRef(c_iterator);
    env->DeleteGlobalRef(c_integer);
    env->DeleteGlobalRef(c_float);
    env->DeleteGlobalRef(c_log_level);
    env->DeleteGlobalRef(c_biconsumer);
    env->DeleteGlobalRef(c_llama_error);
    env->DeleteGlobalRef(c_error_oom);

    env->DeleteGlobalRef(o_utf_8);
}

static void jllama_log_callback(enum ggml_log_level level, const char *text, void *user_data)
{
    if (g_log_callback == nullptr)
        return;

    JNIEnv *env;
    g_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_2);

    jobject java_log_level;
    switch (level)
    {
    case GGML_LOG_LEVEL_ERROR:
        java_log_level = o_log_level_error;
        break;
    case GGML_LOG_LEVEL_WARN:
        java_log_level = o_log_level_warn;
        break;
    case GGML_LOG_LEVEL_INFO:
        java_log_level = o_log_level_info;
        break;
    default:
        java_log_level = o_log_level_debug;
        break;
    }
    jstring java_text = env->NewStringUTF(text);

    env->CallVoidMethod(g_log_callback, m_biconsumer_accept, java_log_level, java_text);

    env->DeleteLocalRef(java_log_level);
    env->DeleteLocalRef(java_text);
}

static void jllama_log_callback(enum ggml_log_level level, std::string text) {
    jllama_log_callback(level, text.c_str(), nullptr);
}

static std::string parse_jstring(JNIEnv *env, jstring java_string)
{
    const jbyteArray string_bytes = (jbyteArray)env->CallObjectMethod(java_string, m_get_bytes, o_utf_8);

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

// Since Java expects utf16 but std::strings are utf8, we can't directly use `env->NewString` or `env-NewString`, but
// we simply send the bytes directly and do the conversion in Java. Unfortunately, there isn't a nice/standardized way
// to do this conversion in C++
static jbyteArray parse_jbytes(JNIEnv *env, std::string string)
{
    jsize len = string.size();
    jbyteArray bytes = env->NewByteArray(len);
    env->SetByteArrayRegion(bytes, 0, len, (jbyte *)string.c_str());
    return bytes;
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

// format incomplete utf-8 multibyte character for output
static std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token)
{
    std::string out = token == -1 ? "" : llama_token_to_piece(ctx, token);
    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80)
    {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }
    return out;
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
    llama_sampling_context ctx_sampling;
    int n_ctx;

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
        if (grammar)
        {
            llama_grammar_free(grammar);
            grammar = nullptr;
        }
    }

    void rewind()
    {
        params.antiprompt.clear();
        params.grammar.clear();
        num_prompt_tokens = 0;
        num_tokens_predicted = 0;
        generated_text = "";
        generated_text.reserve(n_ctx);
        generated_token_probs.clear();
        truncated = false;
        stopped_eos = false;
        stopped_word = false;
        stopped_limit = false;
        stopping_word = "";
        multibyte_pending = 0;
        n_remain = 0;
        n_past = 0;

        if (grammar != nullptr) {
            llama_grammar_free(grammar);
            grammar = nullptr;
            ctx_sampling = llama_sampling_context_init(params, NULL);
        }
    }

    bool loadModel(const gpt_params &params_)
    {
        params = params_;
        std::tie(model, ctx) = llama_init_from_gpt_params(params);
        if (model == nullptr)
        {
            return false;
        }
        n_ctx = llama_n_ctx(ctx);
        last_n_tokens.resize(n_ctx);
        std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
        return true;
    }

    std::vector<llama_token> tokenize(std::string prompt, bool add_bos) const
    {
        return ::llama_tokenize(ctx, prompt, add_bos);
    }

    bool loadGrammar()
    {
        if (!params.grammar.empty()) {
            parsed_grammar = grammar_parser::parse(params.grammar.c_str());
            // will be empty (default) if there are parse errors
            if (parsed_grammar.rules.empty()) {
                jllama_log_callback(GGML_LOG_LEVEL_ERROR, "grammar parse error");
                return false;
            }
            grammar_parser::print_grammar(stderr, parsed_grammar);

            {
                auto it = params.sampling_params.logit_bias.find(llama_token_eos(ctx));
                if (it != params.sampling_params.logit_bias.end() && it->second == -INFINITY) {
                    jllama_log_callback(GGML_LOG_LEVEL_WARN, "EOS token is disabled, which will cause most grammars to fail");
                }
            }

            std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
            grammar = llama_grammar_init(
                grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
        }
        ctx_sampling = llama_sampling_context_init(params, grammar);
        return true;
    }

    void loadInfill()
    {
        bool suff_rm_leading_spc = true;
        if (params.input_suffix.find_first_of(" ") == 0 && params.input_suffix.size() > 1) {
            params.input_suffix.erase(0, 1);
            suff_rm_leading_spc = false;
        }

        auto prefix_tokens = tokenize(params.input_prefix, false);
        auto suffix_tokens = tokenize(params.input_suffix, false);
        const int space_token = 29871;
        if (suff_rm_leading_spc  && suffix_tokens[0] == space_token) {
            suffix_tokens.erase(suffix_tokens.begin());
        }
        prefix_tokens.insert(prefix_tokens.begin(), llama_token_prefix(ctx));
        prefix_tokens.insert(prefix_tokens.begin(), llama_token_bos(ctx)); // always add BOS
        prefix_tokens.insert(prefix_tokens.end(), llama_token_suffix(ctx));
        prefix_tokens.insert(prefix_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
        prefix_tokens.push_back(llama_token_middle(ctx));
        auto prompt_tokens = prefix_tokens;

        num_prompt_tokens = prompt_tokens.size();

        if (params.n_keep < 0)
        {
            params.n_keep = (int)num_prompt_tokens;
        }
        params.n_keep = std::min(params.n_ctx - 4, params.n_keep);

        // if input prompt is too big, truncate like normal
        if (num_prompt_tokens >= (size_t)params.n_ctx)
        {
            // todo we probably want to cut from both sides
            const int n_left = (params.n_ctx - params.n_keep) / 2;
            std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);
            const int erased_blocks = (num_prompt_tokens - params.n_keep - n_left - 1) / n_left;
            new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + params.n_keep + erased_blocks * n_left, prompt_tokens.end());
            std::copy(prompt_tokens.end() - params.n_ctx, prompt_tokens.end(), last_n_tokens.begin());

            jllama_log_callback(GGML_LOG_LEVEL_INFO, "input truncated n_left=" + n_left);

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

        // since #3228 we now have to manually manage the KV cache
        llama_kv_cache_seq_rm(ctx, 0, n_past, -1);

        has_next_token = true;
    }

    void loadPrompt()
    {
        auto prompt_tokens = tokenize(prompt, true);  // always add BOS

        num_prompt_tokens = prompt_tokens.size();

        if (params.n_keep < 0)
        {
            params.n_keep = (int)num_prompt_tokens;
        }
        params.n_keep = std::min(n_ctx - 4, params.n_keep);

        // if input prompt is too big, truncate like normal
        if (num_prompt_tokens >= (size_t)n_ctx)
        {
            const int n_left = (n_ctx - params.n_keep) / 2;
            std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);
            const int erased_blocks = (num_prompt_tokens - params.n_keep - n_left - 1) / n_left;
            new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + params.n_keep + erased_blocks * n_left, prompt_tokens.end());
            std::copy(prompt_tokens.end() - n_ctx, prompt_tokens.end(), last_n_tokens.begin());

            jllama_log_callback(GGML_LOG_LEVEL_INFO, "input truncated n_left=" + n_left);

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

        // since #3228 we now have to manually manage the KV cache
        llama_kv_cache_seq_rm(ctx, 0, n_past, -1);

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

        if (embd.size() >= (size_t)n_ctx)
        {
            // Shift context

            const int n_left    = n_past - params.n_keep - 1;
            const int n_discard = n_left/2;

            llama_kv_cache_seq_rm   (ctx, 0, params.n_keep + 1            , params.n_keep + n_discard + 1);
            llama_kv_cache_seq_shift(ctx, 0, params.n_keep + 1 + n_discard, n_past, -n_discard);

            for (size_t i = params.n_keep + 1 + n_discard; i < embd.size(); i++)
            {
                embd[i - n_discard] = embd[i];
            }
            embd.resize(embd.size() - n_discard);

            n_past -= n_discard;

            truncated = true;
            jllama_log_callback(GGML_LOG_LEVEL_INFO, "input truncated n_left=" + n_left);
        }

        bool tg = true;
        while (n_past < embd.size())
        {
            int n_eval = (int)embd.size() - n_past;
            tg = n_eval == 1;
            if (n_eval > params.n_batch)
            {
                n_eval = params.n_batch;
            }

            if (llama_decode(ctx, llama_batch_get_one(&embd[n_past], n_eval, n_past, 0)))
            {
                jllama_log_callback(GGML_LOG_LEVEL_ERROR, "failed to eval n_eval=" + n_eval);
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

        {
            // out of user input, sample next token
            std::vector<llama_token_data> candidates;
            candidates.reserve(llama_n_vocab(model));

            result.tok = llama_sampling_sample(ctx, NULL, ctx_sampling, last_n_tokens, candidates);

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            const int32_t n_probs = params.sampling_params.n_probs;
            if (params.sampling_params.temp <= 0 && n_probs > 0)
            {
                // For llama_sample_token_greedy we need to sort candidates
                llama_sample_softmax(ctx, &candidates_p);
            }

            for (size_t i = 0; i < std::min(candidates_p.size, (size_t)n_probs); ++i)
            {
                result.probs.push_back({candidates_p.data[i].id, candidates_p.data[i].p});
            }

            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(result.tok);
            if (tg) {
                num_tokens_predicted++;
            }
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

        if (params.sampling_params.n_probs > 0)
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

        return token_with_probs;
    }

    std::vector<float> getEmbedding()
    {
        static const int n_embd = llama_n_embd(model);
        if (!params.embedding)
        {
            jllama_log_callback(GGML_LOG_LEVEL_ERROR, "embedding disabled");
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
    params.mul_mat_q = env->GetBooleanField(jparams, f_mul_mat_q);
    params.memory_f16 = env->GetBooleanField(jparams, f_memory_f16);
    params.embedding = env->GetBooleanField(jparams, f_embedding);
    params.escape = env->GetIntField(jparams, f_n_predict);
    params.use_mmap = env->GetBooleanField(jparams, f_use_mmap);
    params.use_mlock = env->GetBooleanField(jparams, f_use_mlock);
    params.numa = env->GetBooleanField(jparams, f_numa);
    params.verbose_prompt = env->GetBooleanField(jparams, f_verbose_prompt);

//    jstring j_lora_adapter = (jstring)env->GetObjectField(jparams, f_lora_adapter);
//    if (j_lora_adapter != nullptr)
//    {
//        params.lora_adapter = parse_jstring(env, j_lora_adapter);
//        std::cout << params.lora_adapter << std::endl;
//        env->DeleteLocalRef(j_lora_adapter);
//    }
//    jstring j_lora_base = (jstring)env->GetObjectField(jparams, f_lora_base);
//    if (j_lora_base != nullptr)
//    {
//        params.lora_base = parse_jstring(env, j_lora_base);
//        std::cout << params.lora_base << std::endl;
//        env->DeleteLocalRef(j_lora_base);
//    }

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

    if (params.model_alias == "unknown")
    {
        params.model_alias = params.model;
    }

    return params;
}

static void setup_infer_params(JNIEnv *env, jllama_context *llama, jobject jparams)
{
	auto & params = llama->params;

	params.seed = env->GetIntField(jparams, f_infer_seed);
    params.n_predict = env->GetIntField(jparams, f_n_predict);
    params.n_keep = env->GetIntField(jparams, f_n_keep);

    auto & sparams = params.sampling_params;

    sparams.top_k = env->GetIntField(jparams, f_top_k);
    sparams.top_p = env->GetFloatField(jparams, f_top_p);
    sparams.tfs_z = env->GetFloatField(jparams, f_tfs_z);
    sparams.typical_p = env->GetFloatField(jparams, f_typical_p);
    sparams.temp = env->GetFloatField(jparams, f_temperature);
    sparams.repeat_penalty = env->GetFloatField(jparams, f_repeat_penalty);
    sparams.repeat_last_n = env->GetIntField(jparams, f_repeat_last_n);
    sparams.frequency_penalty = env->GetFloatField(jparams, f_frequency_penalty);
    sparams.presence_penalty = env->GetFloatField(jparams, f_presence_penalty);
    sparams.penalize_nl = env->GetBooleanField(jparams, f_penalize_nl);
    sparams.mirostat = env->GetIntField(jparams, f_mirostat);
    sparams.mirostat_tau = env->GetFloatField(jparams, f_mirostat_tau);
    sparams.mirostat_eta = env->GetFloatField(jparams, f_mirostat_eta);
    sparams.n_probs = env->GetIntField(jparams, f_n_probs);

    jstring j_grammar = (jstring)env->GetObjectField(jparams, f_grammar);
    if (j_grammar != nullptr)
    {
        params.grammar = parse_jstring(env, j_grammar);
        env->DeleteLocalRef(j_grammar);
        if (!llama->loadGrammar())
		{
			env->ThrowNew(c_llama_error, "could not load grammar");
		}
    }

    sparams.logit_bias.clear();
    jboolean ignore_eos = env->GetBooleanField(jparams, f_ignore_eos);
    if (ignore_eos)
    {
        sparams.logit_bias[llama_token_eos(llama->ctx)] = -INFINITY;
    }

    jobject logit_bias = env->GetObjectField(jparams, f_logit_bias);
    if (logit_bias != nullptr)
    {
        jobject entry_set = env->CallObjectMethod(logit_bias, m_entry_set);
        jobject iterator = env->CallObjectMethod(entry_set, m_set_iterator);
        while (env->CallBooleanMethod(iterator, m_iterator_has_next))
        {
            jobject entry = env->CallObjectMethod(iterator, m_iterator_next);
            jobject key = env->CallObjectMethod(entry, m_entry_key);
            jobject value = env->CallObjectMethod(entry, m_entry_value);

            int tok = parse_jinteger(env, key);
            float bias = parse_jfloat(env, value);
            sparams.logit_bias[tok] = bias;

            env->DeleteLocalRef(entry);
            env->DeleteLocalRef(key);
            env->DeleteLocalRef(value);
        }
    }

    params.antiprompt.clear();
    jobjectArray antiprompt = (jobjectArray)env->GetObjectField(jparams, f_antiprompt);
    if (antiprompt != nullptr)
    {
        jsize array_length = env->GetArrayLength(antiprompt);
        for (jsize i = 0; i < array_length; i++)
        {
            jstring java_string = (jstring)env->GetObjectArrayElement(antiprompt, i);
            if (java_string != nullptr)
            {
                std::string string = parse_jstring(env, java_string);
                params.antiprompt.push_back(string);
                env->DeleteLocalRef(java_string);
            }
        }
    }

    llama->ctx_sampling = llama_sampling_context_init(params, llama->grammar);
}

static void setup_answering(JNIEnv *env, jllama_context *llama, jstring prompt, jobject params)
{
    llama->prompt = parse_jstring(env, prompt);
    llama->params.input_prefix = "";
	llama->params.input_suffix = "";
    setup_infer_params(env, llama, params);
}

static void setup_infilling(JNIEnv *env, jllama_context *llama, jstring prefix, jstring suffix, jobject params)
{
	llama->prompt = "";
	llama->params.input_prefix = parse_jstring(env, prefix);
	llama->params.input_suffix = parse_jstring(env, suffix);
	setup_infer_params(env, llama, params);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_loadModel(JNIEnv *env, jobject obj, jstring file_path, jobject jparams)
{
    gpt_params params = parse_model_params(env, jparams, file_path);

    jllama_context *llama = new jllama_context;
    llama_backend_init(false);

    if (!llama->loadModel(params))
    {
        env->ThrowNew(c_llama_error, "could not load model from given file path");
        return;
    }

    // jllama_log_callback(GGML_LOG_LEVEL_INFO, "build=" + BUILD_NUMBER);
    // jllama_log_callback(GGML_LOG_LEVEL_INFO, "commit=" + BUILD_COMMIT);
    // jllama_log_callback(GGML_LOG_LEVEL_INFO, "n_threads=" + params.n_threads);
    // jllama_log_callback(GGML_LOG_LEVEL_INFO, "total_threads=" + std::thread::hardware_concurrency());
    // jllama_log_callback(GGML_LOG_LEVEL_INFO, "system_info=" + llama_print_system_info());

    env->SetLongField(obj, f_model_pointer, reinterpret_cast<jlong>(llama));
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_newAnswerIterator(JNIEnv *env, jobject obj, jstring prompt, jobject params)
{
    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
    jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);

//    auto lock = llama->lock();

    llama->rewind();

    llama_reset_timings(llama->ctx);

    setup_answering(env, llama, prompt, params);

    llama->loadPrompt();
    llama->beginCompletion();
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_newInfillIterator(JNIEnv *env, jobject obj, jstring prefix, jstring suffix, jobject params)
{
    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
    jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);

//    auto lock = llama->lock();

    llama->rewind();

    llama_reset_timings(llama->ctx);

    setup_infilling(env, llama, prefix, suffix, params);

    llama->loadInfill();
    llama->beginCompletion();
}

JNIEXPORT jobject JNICALL Java_de_kherud_llama_LlamaModel_getNext(JNIEnv *env, jobject obj, jobject iter)
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
	if (stop_pos != std::string::npos) {
		is_stop_full = true;
		llama->generated_text.erase(
			llama->generated_text.begin() + pos + stop_pos,
			llama->generated_text.end());
		pos = std::min(sent_count, llama->generated_text.size());
	} else {
		is_stop_full = false;
		stop_pos = llama->findStoppingStrings(str_test, token_text.size(),
			STOP_PARTIAL);
	}

    std::string to_send;
    if (
		stop_pos == std::string::npos ||
		// Send rest of the text if we are at the end of the generation
		(!llama->has_next_token && !is_stop_full && stop_pos > 0)
	) {
		to_send = llama->generated_text.substr(pos, std::string::npos);

		sent_count += to_send.size();
		env->SetLongField(iter, f_iter_n_generated, sent_count);

		std::vector<completion_token_output> probs_output = {};

		if (llama->params.sampling_params.n_probs > 0) {
			const std::vector<llama_token> to_send_toks = llama_tokenize(llama->ctx, to_send, false);
			size_t probs_pos = std::min(sent_token_probs_index, llama->generated_token_probs.size());
			size_t probs_stop_pos = std::min(sent_token_probs_index + to_send_toks.size(), llama->generated_token_probs.size());
			if (probs_pos < probs_stop_pos) {
				probs_output = std::vector<completion_token_output>(llama->generated_token_probs.begin() + probs_pos, llama->generated_token_probs.begin() + probs_stop_pos);
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
        // llama.mutex.unlock();
        // lock.release();
    }

	jobject o_probabilities = env->NewObject(c_hash_map, cc_hash_map);
	for (const auto& tp : token_with_probs.probs)
    {
    	jobject jtoken = env->NewObject(c_integer, cc_integer, tp.tok);
    	jobject jprob = env->NewObject(c_float, cc_float, tp.prob);
    	env->CallObjectMethod(o_probabilities, m_map_put, jtoken, jprob);
    }
	jbyteArray jbytes = parse_jbytes(env, to_send);
	return env->NewObject(c_output, cc_output, token_with_probs.tok, jbytes, o_probabilities);
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_getAnswer(JNIEnv *env, jobject obj, jstring prompt, jobject params)
{
    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
	jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);

//	auto lock = llama->lock();

	llama->rewind();

	llama_reset_timings(llama->ctx);

	setup_answering(env, llama, prompt, params);

	llama->loadPrompt();
	llama->beginCompletion();

    size_t stop_pos = std::string::npos;

	while (llama->has_next_token) {
		const completion_token_output token_with_probs = llama->doCompletion();
		const std::string token_text = token_with_probs.tok == -1 ? "" : llama_token_to_piece(llama->ctx, token_with_probs.tok);

		stop_pos = llama->findStoppingStrings(llama->generated_text,
			token_text.size(), STOP_FULL);
	}

	if (stop_pos == std::string::npos) {
		stop_pos = llama->findStoppingStrings(llama->generated_text, 0, STOP_PARTIAL);
	}
	if (stop_pos != std::string::npos) {
		llama->generated_text.erase(llama->generated_text.begin() + stop_pos,
			llama->generated_text.end());
	}

//	llama->lock().release();
//	llama->mutex.unlock();

    return parse_jbytes(env, llama->generated_text);
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_getInfill(JNIEnv *env, jobject obj, jstring prefix, jstring suffix, jobject params)
{
    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
	jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);

//	auto lock = llama->lock();

	llama->rewind();

	llama_reset_timings(llama->ctx);

	setup_infilling(env, llama, prefix, suffix, params);

	llama->loadInfill();
	llama->beginCompletion();

    size_t stop_pos = std::string::npos;

	while (llama->has_next_token) {
		const completion_token_output token_with_probs = llama->doCompletion();
		const std::string token_text = token_with_probs.tok == -1 ? "" : llama_token_to_piece(llama->ctx, token_with_probs.tok);

		stop_pos = llama->findStoppingStrings(llama->generated_text,
			token_text.size(), STOP_FULL);
	}

	if (stop_pos == std::string::npos) {
		stop_pos = llama->findStoppingStrings(llama->generated_text, 0, STOP_PARTIAL);
	}
	if (stop_pos != std::string::npos) {
		llama->generated_text.erase(llama->generated_text.begin() + stop_pos,
			llama->generated_text.end());
	}

//	llama->lock().release();
//	llama->mutex.unlock();

    return parse_jbytes(env, llama->generated_text);
}

JNIEXPORT jfloatArray JNICALL Java_de_kherud_llama_LlamaModel_embed(JNIEnv *env, jobject obj, jstring java_prompt)
{
    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
    jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);

//	auto lock = llama->lock();

	llama->rewind();
	llama_reset_timings(llama->ctx);
	llama->prompt = parse_jstring(env, java_prompt);
	llama->params.n_predict = 0;
	llama->loadPrompt();
	llama->beginCompletion();
	llama->doCompletion();

    static const int n_embd = llama_n_embd(llama->model);
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

JNIEXPORT jintArray JNICALL Java_de_kherud_llama_LlamaModel_encode(JNIEnv *env, jobject obj, jstring jprompt)
{
	jlong llama_handle = env->GetLongField(obj, f_model_pointer);
	jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);

//	auto lock = llama->lock();

	std::string prompt = parse_jstring(env, jprompt);
    std::vector<llama_token> tokens = llama->tokenize(prompt, false);

    jintArray java_tokens = env->NewIntArray(tokens.size());
    if (java_tokens == nullptr)
    {
        env->ThrowNew(c_error_oom, "could not allocate tokens");
        return nullptr;
    }

    env->SetIntArrayRegion(java_tokens, 0, tokens.size(), reinterpret_cast<const jint *>(tokens.data()));

//	lock.release();
    return java_tokens;
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_decodeBytes(JNIEnv *env, jobject obj, jintArray java_tokens)
{
    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
    jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);

//    auto lock = llama->lock();

    jsize length = env->GetArrayLength(java_tokens);
    jint *elements = env->GetIntArrayElements(java_tokens, nullptr);
    std::vector<llama_token> tokens(elements, elements + length);
    std::string text = tokens_to_str(llama->ctx, tokens.cbegin(), tokens.cend());

    env->ReleaseIntArrayElements(java_tokens, elements, 0);

//	lock.release();
	return parse_jbytes(env, text);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setLogger(JNIEnv *env, jclass clazz, jobject callback)
{
    env->GetJavaVM(&g_vm);

    if (g_log_callback != nullptr)
    {
        env->DeleteGlobalRef(g_log_callback);
    }

    if (callback == nullptr)
    {
        llama_log_set(nullptr, nullptr);
    }
    else
    {
        g_log_callback = env->NewGlobalRef(callback);
        llama_log_set(jllama_log_callback, nullptr);
    }
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_delete(JNIEnv * env, jobject obj) {
	jlong llama_handle = env->GetLongField(obj, f_model_pointer);
	jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
	delete llama;
}
