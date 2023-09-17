#include "llama.h"
#include "jllama.h"
#include "server.cpp"

#include <iostream>
#include <string>

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
static jclass c_llama_error = 0;

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
static jfieldID f_model_seed = 0;
// model parameters
static jfieldID f_n_threads = 0;
static jfieldID f_infer_seed = 0;
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


JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    JNIEnv *env = 0;

    if (JNI_OK != vm->GetEnv((void **)&env, JNI_VERSION_1_1)) {
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
    c_llama_error = env->FindClass("de/kherud/llama/LlamaException");

    if (!(c_llama_model && c_llama_iterator && c_infer_params && c_model_params && c_string && c_map && c_set && c_entry && c_iterator && c_integer && c_float && c_llama_error)) {
        goto error;
    }

    // create references
    c_llama_model = (jclass) env->NewWeakGlobalRef(c_llama_model);
    c_llama_iterator = (jclass) env->NewWeakGlobalRef(c_llama_iterator);
    c_infer_params = (jclass) env->NewWeakGlobalRef(c_infer_params);
    c_model_params = (jclass) env->NewWeakGlobalRef(c_model_params);
    c_string = (jclass) env->NewWeakGlobalRef(c_string);
    c_map = (jclass) env->NewWeakGlobalRef(c_map);
    c_set = (jclass) env->NewWeakGlobalRef(c_set);
    c_entry = (jclass) env->NewWeakGlobalRef(c_entry);
    c_iterator = (jclass) env->NewWeakGlobalRef(c_iterator);
    c_integer = (jclass) env->NewWeakGlobalRef(c_integer);
    c_float = (jclass) env->NewWeakGlobalRef(c_float);
    c_llama_error = (jclass) env->NewWeakGlobalRef(c_llama_error);

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

    if (!(m_get_bytes && m_entry_set && m_set_iterator && m_iterator_has_next && m_iterator_next && m_entry_key && m_entry_value && m_int_value && m_float_value)) {
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
    f_model_seed = env->GetFieldID(c_infer_params, "seed", "I");

    f_n_threads = env->GetFieldID(c_model_params, "nThreads", "I");
    f_infer_seed = env->GetFieldID(c_model_params, "seed", "I");
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
    f_hellaswag_tasks = env->GetFieldID(c_model_params, "hellaswagTasks", "I");
    f_memory_f16 = env->GetFieldID(c_model_params, "memoryF16", "Z");
    f_mem_test = env->GetFieldID(c_model_params, "memTest", "Z");
    f_numa = env->GetFieldID(c_model_params, "numa", "Z");
    f_verbose_prompt = env->GetFieldID(c_model_params, "verbosePrompt", "Z");

    if (!(f_model_pointer && f_iter_has_next && f_iter_n_generated && f_iter_token_index)) {
        goto error;
    }
    if (!(f_n_predict && f_n_keep && f_n_probs && f_logit_bias && f_top_k && f_top_p && f_tfs_z && f_typical_p && f_temperature && f_repeat_penalty && f_repeat_last_n && f_frequency_penalty && f_presence_penalty && f_penalize_nl && f_ignore_eos && f_mirostat && f_mirostat_tau && f_mirostat_eta && f_beam_search && f_n_beams && f_grammar && f_antiprompt && f_model_seed)) {
        goto error;
    }
    if (!(f_n_threads && f_infer_seed && f_n_ctx && f_n_batch && f_n_gpu_layers && f_main_gpu && f_tensor_split && f_rope_freq_base && f_rope_freq_scale && f_low_vram && f_mul_mat_q && f_f16_kv && f_logits_all && f_vocab_only && f_use_mmap && f_use_mlock && f_embedding && f_lora_adapter && f_lora_base && f_hellaswag && f_hellaswag_tasks && f_memory_f16 && f_mem_test && f_numa && f_verbose_prompt)) {
        goto error;
    }

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        goto error;
    }

    goto success;

error:
    return JNI_ERR;

success:
    return JNI_VERSION_1_1;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv *env = 0;

    if (JNI_OK != vm->GetEnv((void **)&env, JNI_VERSION_1_2)) return;

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
    env->DeleteWeakGlobalRef(c_llama_error);
}

static std::string parse_jstring(JNIEnv* env, jstring java_string) {
    const jbyteArray string_bytes = (jbyteArray) env->CallObjectMethod(java_string, m_get_bytes, env->NewStringUTF("UTF-8"));

    size_t length = (size_t) env->GetArrayLength(string_bytes);
    jbyte* byte_elements = env->GetByteArrayElements(string_bytes, nullptr);

    std::string string = std::string((char*) byte_elements, length);
    env->ReleaseByteArrayElements(string_bytes, byte_elements, JNI_ABORT);

    env->DeleteLocalRef(string_bytes);
    return string;
}

static int parse_jinteger(JNIEnv* env, jobject java_integer) {
    if (!java_integer) return 0;
    return env->CallIntMethod(java_integer, m_int_value);
}

static float parse_jfloat(JNIEnv* env, jobject java_float) {
    if (!java_float) return 0;
    return env->CallFloatMethod(java_float, m_float_value);
}

float getJavaFloatValue(JNIEnv* env, jobject javaFloat) {
    if (!javaFloat) return 0.0f;  // Return a default value if the input object is null

    jclass floatClass = env->FindClass("java/lang/Float");
    jmethodID floatValueMethodID = env->GetMethodID(floatClass, "floatValue", "()F");

    float value = env->CallFloatMethod(javaFloat, floatValueMethodID);

    // Clean up the local reference
    env->DeleteLocalRef(floatClass);

    return value;
}

static jstring parse_utf16_string(JNIEnv* env, std::string string) {
    std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> converter;
    std::u16string string_utf16 = converter.from_bytes(string);
    return env->NewString((jchar*)string_utf16.data(), string_utf16.size());
}

static gpt_params parse_model_params(JNIEnv* env, jobject jparams, jstring java_file_path) {
    gpt_params params;

    params.model = parse_jstring(env, java_file_path);

    return params;
}

static void parse_inference_params(JNIEnv* env, llama_server_context *llama, jstring prompt, jobject params) {
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
    llama->params.seed = env->GetIntField(params, f_model_seed);

    jstring j_grammar = (jstring) env->GetObjectField(params, f_grammar);
    if (j_grammar == nullptr) {
        llama->params.grammar = "";
    } else {
        llama->params.grammar = parse_jstring(env, j_grammar);
        env->DeleteLocalRef(j_grammar);
    }

    llama->params.logit_bias.clear();
    jboolean ignore_eos = env->GetBooleanField(params, f_ignore_eos);
    if (ignore_eos) {
        llama->params.logit_bias[llama_token_eos(llama->ctx)] = -INFINITY;
    }

    jobject logit_bias = env->GetObjectField(params, f_logit_bias);
    if (logit_bias != nullptr) {
        const int n_vocab = llama_n_vocab(llama->ctx);
        jobject entry_set = env->CallObjectMethod(logit_bias, m_entry_set);
        jobject iterator = env->CallObjectMethod(entry_set, m_set_iterator);
        while (env->CallBooleanMethod(iterator, m_iterator_has_next)) {
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
    jobjectArray antiprompt = (jobjectArray) env->GetObjectField(params, f_antiprompt);
    if (antiprompt != nullptr) {
        jsize array_length = env->GetArrayLength(antiprompt);
        for (jsize i = 0; i < array_length; i++) {
            jstring java_string = (jstring) env->GetObjectArrayElement(antiprompt, i);
            if (java_string != nullptr) {
                std::string string = parse_jstring(env, java_string);
                llama->params.antiprompt.push_back(string);
                env->DeleteLocalRef(java_string);
            }
        }
    }

    LOG_VERBOSE("completion parameters parsed", format_generation_settings(*llama));
}


JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getSystemInfo(JNIEnv * env, jobject obj) {
	const char * sys_info = llama_print_system_info();
	return env->NewStringUTF(sys_info);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_loadModel(JNIEnv * env, jobject obj, jstring file_path, jobject jparams) {
    gpt_params params = parse_model_params(env, jparams, file_path);

    llama_server_context* llama = new llama_server_context;
	llama_backend_init(false);

	if (!llama->loadModel(params))
    {
        env->ThrowNew(c_llama_error, "could not load model from given file path");
        return;
    }

    LOG_INFO("build info", {{"build", BUILD_NUMBER},
                                {"commit", BUILD_COMMIT}});
    LOG_INFO("system info", {
                                {"n_threads", params.n_threads},
                                {"total_threads", std::thread::hardware_concurrency()},
                                {"system_info", llama_print_system_info()},
                            });

    env->SetLongField(obj, f_model_pointer, reinterpret_cast<jlong>(llama));
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setupInference(JNIEnv * env, jobject obj, jstring prompt, jobject params) {
    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
    llama_server_context* llama = reinterpret_cast<llama_server_context*>(llama_handle);

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

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getNext(JNIEnv * env, jobject obj, jobject iter) {
    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
    llama_server_context* llama = reinterpret_cast<llama_server_context*>(llama_handle);

    size_t sent_count = env->GetLongField(iter, f_iter_n_generated);;
    size_t sent_token_probs_index = env->GetLongField(iter, f_iter_token_index);

    completion_token_output token_with_probs;
    while (llama->has_next_token) {
        token_with_probs = llama->doCompletion();
        if (token_with_probs.tok >= 0 && llama->multibyte_pending <= 0) {
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
        stop_pos = llama->findStoppingStrings(str_test, token_text.size(), STOP_PARTIAL);
    }

    std::string to_send;
    if (stop_pos == std::string::npos || (!llama->has_next_token && !is_stop_full && stop_pos > 0)) {
        to_send = llama->generated_text.substr(pos, std::string::npos);
        sent_count += to_send.size();
        env->SetLongField(iter, f_iter_n_generated, sent_count);

        std::vector<completion_token_output> probs_output = {};

        if (llama->params.n_probs > 0) {
            const std::vector<llama_token> to_send_toks = llama_tokenize(llama->ctx, to_send, false);
            size_t probs_pos = std::min(sent_token_probs_index, llama->generated_token_probs.size());
            size_t probs_stop_pos = std::min(
                sent_token_probs_index + to_send_toks.size(),
                llama->generated_token_probs.size()
            );
            if (probs_pos < probs_stop_pos) {
                probs_output = std::vector<completion_token_output>(
                    llama->generated_token_probs.begin() + probs_pos,
                    llama->generated_token_probs.begin() + probs_stop_pos
                );
            }
            sent_token_probs_index = probs_stop_pos;
            env->SetLongField(iter, f_iter_token_index, sent_token_probs_index);
        }
    } else {
        to_send = "";
    }

    if (!llama->has_next_token) {
        env->SetLongField(iter, f_iter_has_next, false);
    }

    return parse_utf16_string(env, to_send);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getFull(JNIEnv * env, jobject obj, jstring prompt, jobject params) {
    Java_de_kherud_llama_LlamaModel_setupInference(env, obj, prompt, params);

    jlong llama_handle = env->GetLongField(obj, f_model_pointer);
    llama_server_context* llama = reinterpret_cast<llama_server_context*>(llama_handle);

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
        llama->generated_text.erase(llama->generated_text.begin() + stop_pos, llama->generated_text.end());
    }

    auto probs = llama->generated_token_probs;
    if (llama->params.n_probs > 0 && llama->stopped_word) {
        const std::vector<llama_token> stop_word_toks = llama_tokenize(llama->ctx, llama->stopping_word, false);
        probs = std::vector<completion_token_output>(llama->generated_token_probs.begin(), llama->generated_token_probs.end() - stop_word_toks.size());
    }

    llama_print_timings(llama->ctx);

    llama->lock().release();
    llama->mutex.unlock();

    return parse_utf16_string(env, llama->generated_text);
}
