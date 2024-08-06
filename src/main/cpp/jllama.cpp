#include "jllama.h"

#include "llama.h"
#include "nlohmann/json.hpp"
#include "server.hpp"

#include <functional>
#include <stdexcept>

// We store some references to Java classes and their fields/methods here to speed up things for later and to fail
// early on if anything can't be found. This happens when the JVM loads the shared library (see `JNI_OnLoad`).
// The references remain valid throughout the whole life of the shared library, on `JNI_OnUnload` they are released.

namespace
{
JavaVM *g_vm = nullptr;

// classes
jclass c_llama_model = nullptr;
jclass c_llama_iterator = nullptr;
jclass c_standard_charsets = nullptr;
jclass c_output = nullptr;
jclass c_string = nullptr;
jclass c_hash_map = nullptr;
jclass c_map = nullptr;
jclass c_set = nullptr;
jclass c_entry = nullptr;
jclass c_iterator = nullptr;
jclass c_integer = nullptr;
jclass c_float = nullptr;
jclass c_biconsumer = nullptr;
jclass c_llama_error = nullptr;
jclass c_log_level = nullptr;
jclass c_log_format = nullptr;
jclass c_error_oom = nullptr;

// constructors
jmethodID cc_output = nullptr;
jmethodID cc_hash_map = nullptr;
jmethodID cc_integer = nullptr;
jmethodID cc_float = nullptr;

// methods
jmethodID m_get_bytes = nullptr;
jmethodID m_entry_set = nullptr;
jmethodID m_set_iterator = nullptr;
jmethodID m_iterator_has_next = nullptr;
jmethodID m_iterator_next = nullptr;
jmethodID m_entry_key = nullptr;
jmethodID m_entry_value = nullptr;
jmethodID m_map_put = nullptr;
jmethodID m_int_value = nullptr;
jmethodID m_float_value = nullptr;
jmethodID m_biconsumer_accept = nullptr;

// fields
jfieldID f_model_pointer = nullptr;
jfieldID f_task_id = nullptr;
jfieldID f_utf_8 = nullptr;
jfieldID f_iter_has_next = nullptr;
jfieldID f_log_level_debug = nullptr;
jfieldID f_log_level_info = nullptr;
jfieldID f_log_level_warn = nullptr;
jfieldID f_log_level_error = nullptr;
jfieldID f_log_format_json = nullptr;
jfieldID f_log_format_text = nullptr;

// objects
jobject o_utf_8 = nullptr;
jobject o_log_level_debug = nullptr;
jobject o_log_level_info = nullptr;
jobject o_log_level_warn = nullptr;
jobject o_log_level_error = nullptr;
jobject o_log_format_json = nullptr;
jobject o_log_format_text = nullptr;
jobject o_log_callback = nullptr;

/**
 * Convert a Java string to a std::string
 */
std::string parse_jstring(JNIEnv *env, jstring java_string)
{
    auto *const string_bytes = (jbyteArray)env->CallObjectMethod(java_string, m_get_bytes, o_utf_8);

    auto length = (size_t)env->GetArrayLength(string_bytes);
    jbyte *byte_elements = env->GetByteArrayElements(string_bytes, nullptr);

    std::string string = std::string((char *)byte_elements, length);

    env->ReleaseByteArrayElements(string_bytes, byte_elements, JNI_ABORT);
    env->DeleteLocalRef(string_bytes);

    return string;
}

/**
 * Since Java expects utf16 but std::strings are utf8, we can't directly use `env->NewString` or `env-NewString`,
 * but we directly send the bytes and do the conversion in Java. Unfortunately, there isn't a nice/standardized way to
 * do this conversion in C++
 */
jbyteArray parse_jbytes(JNIEnv *env, const std::string &string)
{
    jsize length = string.size(); // NOLINT(*-narrowing-conversions)
    jbyteArray bytes = env->NewByteArray(length);
    env->SetByteArrayRegion(bytes, 0, length, reinterpret_cast<const jbyte *>(string.c_str()));
    return bytes;
}

/**
 * Map a llama.cpp log level to its Java enumeration option.
 */
jobject log_level_to_jobject(ggml_log_level level)
{
    switch (level)
    {
    case GGML_LOG_LEVEL_ERROR:
        return o_log_level_error;
    case GGML_LOG_LEVEL_WARN:
        return o_log_level_warn;
    default:
    case GGML_LOG_LEVEL_INFO:
        return o_log_level_info;
    case GGML_LOG_LEVEL_DEBUG:
        return o_log_level_debug;
    }
}

/**
 * Returns the JNIEnv of the current thread.
 */
JNIEnv *get_jni_env()
{
    JNIEnv *env = nullptr;
    if (g_vm == nullptr || g_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK)
    {
        throw std::runtime_error("Thread is not attached to the JVM");
    }
    return env;
}

/**
 * Invoke the log callback if there is any.
 */
void log_callback_trampoline(ggml_log_level level, const char *text, void *user_data)
{
    if (log_callback != nullptr)
    {
        log_callback(level, text, user_data);
    }
}
} // namespace

bool log_json;
std::function<void(ggml_log_level, const char *, void *)> log_callback;

/**
 * The VM calls JNI_OnLoad when the native library is loaded (for example, through `System.loadLibrary`).
 * `JNI_OnLoad` must return the JNI version needed by the native library.
 * In order to use any of the new JNI functions, a native library must export a `JNI_OnLoad` function that returns
 * `JNI_VERSION_1_2`. If the native library does not export a JNI_OnLoad function, the VM assumes that the library
 * only requires JNI version `JNI_VERSION_1_1`. If the VM does not recognize the version number returned by
 `JNI_OnLoad`, the VM will unload the library and act as if the library was never loaded.
 */
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved)
{
    g_vm = vm;
    JNIEnv *env = nullptr;

    if (JNI_OK != vm->GetEnv((void **)&env, JNI_VERSION_1_1))
    {
        goto error;
    }

    // find classes
    c_llama_model = env->FindClass("de/kherud/llama/LlamaModel");
    c_llama_iterator = env->FindClass("de/kherud/llama/LlamaIterator");
    c_standard_charsets = env->FindClass("java/nio/charset/StandardCharsets");
    c_output = env->FindClass("de/kherud/llama/LlamaOutput");
    c_string = env->FindClass("java/lang/String");
    c_hash_map = env->FindClass("java/util/HashMap");
    c_map = env->FindClass("java/util/Map");
    c_set = env->FindClass("java/util/Set");
    c_entry = env->FindClass("java/util/Map$Entry");
    c_iterator = env->FindClass("java/util/Iterator");
    c_integer = env->FindClass("java/lang/Integer");
    c_float = env->FindClass("java/lang/Float");
    c_biconsumer = env->FindClass("java/util/function/BiConsumer");
    c_llama_error = env->FindClass("de/kherud/llama/LlamaException");
    c_log_level = env->FindClass("de/kherud/llama/LogLevel");
    c_log_format = env->FindClass("de/kherud/llama/args/LogFormat");
    c_error_oom = env->FindClass("java/lang/OutOfMemoryError");

    if (!(c_llama_model && c_llama_iterator && c_standard_charsets && c_output && c_string && c_hash_map && c_map &&
          c_set && c_entry && c_iterator && c_integer && c_float && c_biconsumer && c_llama_error && c_log_level &&
          c_log_format && c_error_oom))
    {
        goto error;
    }

    // create references
    c_llama_model = (jclass)env->NewGlobalRef(c_llama_model);
    c_llama_iterator = (jclass)env->NewGlobalRef(c_llama_iterator);
    c_output = (jclass)env->NewGlobalRef(c_output);
    c_string = (jclass)env->NewGlobalRef(c_string);
    c_hash_map = (jclass)env->NewGlobalRef(c_hash_map);
    c_map = (jclass)env->NewGlobalRef(c_map);
    c_set = (jclass)env->NewGlobalRef(c_set);
    c_entry = (jclass)env->NewGlobalRef(c_entry);
    c_iterator = (jclass)env->NewGlobalRef(c_iterator);
    c_integer = (jclass)env->NewGlobalRef(c_integer);
    c_float = (jclass)env->NewGlobalRef(c_float);
    c_biconsumer = (jclass)env->NewGlobalRef(c_biconsumer);
    c_llama_error = (jclass)env->NewGlobalRef(c_llama_error);
    c_log_level = (jclass)env->NewGlobalRef(c_log_level);
    c_log_format = (jclass)env->NewGlobalRef(c_log_format);
    c_error_oom = (jclass)env->NewGlobalRef(c_error_oom);

    // find constructors
    cc_output = env->GetMethodID(c_output, "<init>", "([BLjava/util/Map;Z)V");
    cc_hash_map = env->GetMethodID(c_hash_map, "<init>", "()V");
    cc_integer = env->GetMethodID(c_integer, "<init>", "(I)V");
    cc_float = env->GetMethodID(c_float, "<init>", "(F)V");

    if (!(cc_output && cc_hash_map && cc_integer && cc_float))
    {
        goto error;
    }

    // find methods
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

    if (!(m_get_bytes && m_entry_set && m_set_iterator && m_iterator_has_next && m_iterator_next && m_entry_key &&
          m_entry_value && m_map_put && m_int_value && m_float_value && m_biconsumer_accept))
    {
        goto error;
    }

    // find fields
    f_model_pointer = env->GetFieldID(c_llama_model, "ctx", "J");
    f_task_id = env->GetFieldID(c_llama_iterator, "taskId", "I");
    f_utf_8 = env->GetStaticFieldID(c_standard_charsets, "UTF_8", "Ljava/nio/charset/Charset;");
    f_iter_has_next = env->GetFieldID(c_llama_iterator, "hasNext", "Z");
    f_log_level_debug = env->GetStaticFieldID(c_log_level, "DEBUG", "Lde/kherud/llama/LogLevel;");
    f_log_level_info = env->GetStaticFieldID(c_log_level, "INFO", "Lde/kherud/llama/LogLevel;");
    f_log_level_warn = env->GetStaticFieldID(c_log_level, "WARN", "Lde/kherud/llama/LogLevel;");
    f_log_level_error = env->GetStaticFieldID(c_log_level, "ERROR", "Lde/kherud/llama/LogLevel;");
    f_log_format_json = env->GetStaticFieldID(c_log_format, "JSON", "Lde/kherud/llama/args/LogFormat;");
    f_log_format_text = env->GetStaticFieldID(c_log_format, "TEXT", "Lde/kherud/llama/args/LogFormat;");

    if (!(f_model_pointer && f_task_id && f_utf_8 && f_iter_has_next && f_log_level_debug && f_log_level_info &&
          f_log_level_warn && f_log_level_error && f_log_format_json && f_log_format_text))
    {
        goto error;
    }

    o_utf_8 = env->NewStringUTF("UTF-8");
    o_log_level_debug = env->GetStaticObjectField(c_log_level, f_log_level_debug);
    o_log_level_info = env->GetStaticObjectField(c_log_level, f_log_level_info);
    o_log_level_warn = env->GetStaticObjectField(c_log_level, f_log_level_warn);
    o_log_level_error = env->GetStaticObjectField(c_log_level, f_log_level_error);
    o_log_format_json = env->GetStaticObjectField(c_log_format, f_log_format_json);
    o_log_format_text = env->GetStaticObjectField(c_log_format, f_log_format_text);

    if (!(o_utf_8 && o_log_level_debug && o_log_level_info && o_log_level_warn && o_log_level_error &&
          o_log_format_json && o_log_format_text))
    {
        goto error;
    }

    o_utf_8 = env->NewGlobalRef(o_utf_8);
    o_log_level_debug = env->NewGlobalRef(o_log_level_debug);
    o_log_level_info = env->NewGlobalRef(o_log_level_info);
    o_log_level_warn = env->NewGlobalRef(o_log_level_warn);
    o_log_level_error = env->NewGlobalRef(o_log_level_error);
    o_log_format_json = env->NewGlobalRef(o_log_format_json);
    o_log_format_text = env->NewGlobalRef(o_log_format_text);

    if (env->ExceptionCheck())
    {
        env->ExceptionDescribe();
        goto error;
    }

    llama_backend_init();

    goto success;

error:
    return JNI_ERR;

success:
    return JNI_VERSION_1_6;
}

/**
 * The VM calls `JNI_OnUnload` when the class loader containing the native library is garbage collected.
 * This function can be used to perform cleanup operations. Because this function is called in an unknown context
 * (such as from a finalizer), the programmer should be conservative on using Java VM services, and refrain from
 * arbitrary Java call-backs.
 * Note that `JNI_OnLoad` and `JNI_OnUnload` are two functions optionally supplied by JNI libraries, not exported from
 * the VM.
 */
JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved)
{
    JNIEnv *env = nullptr;

    if (JNI_OK != vm->GetEnv((void **)&env, JNI_VERSION_1_6))
    {
        return;
    }

    env->DeleteGlobalRef(c_llama_model);
    env->DeleteGlobalRef(c_llama_iterator);
    env->DeleteGlobalRef(c_output);
    env->DeleteGlobalRef(c_string);
    env->DeleteGlobalRef(c_hash_map);
    env->DeleteGlobalRef(c_map);
    env->DeleteGlobalRef(c_set);
    env->DeleteGlobalRef(c_entry);
    env->DeleteGlobalRef(c_iterator);
    env->DeleteGlobalRef(c_integer);
    env->DeleteGlobalRef(c_float);
    env->DeleteGlobalRef(c_biconsumer);
    env->DeleteGlobalRef(c_llama_error);
    env->DeleteGlobalRef(c_log_level);
    env->DeleteGlobalRef(c_log_level);
    env->DeleteGlobalRef(c_error_oom);

    env->DeleteGlobalRef(o_utf_8);
    env->DeleteGlobalRef(o_log_level_debug);
    env->DeleteGlobalRef(o_log_level_info);
    env->DeleteGlobalRef(o_log_level_warn);
    env->DeleteGlobalRef(o_log_level_error);
    env->DeleteGlobalRef(o_log_format_json);
    env->DeleteGlobalRef(o_log_format_text);

    if (o_log_callback != nullptr)
    {
        env->DeleteGlobalRef(o_log_callback);
    }

    llama_backend_free();
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_loadModel(JNIEnv *env, jobject obj, jstring jparams)
{
    gpt_params params;

    auto *ctx_server = new server_context();

    std::string c_params = parse_jstring(env, jparams);
    json json_params = json::parse(c_params);
    server_params_parse(json_params, params);

    if (json_value(json_params, "disable_log", false))
    {
        log_disable();
    }
    else
    {
        log_enable();
    }

    if (!params.system_prompt.empty())
    {
        ctx_server->system_prompt_set(params.system_prompt);
    }

    if (params.model_alias == "unknown")
    {
        params.model_alias = params.model;
    }

    llama_numa_init(params.numa);

    LOG_INFO("build info", {{"build", LLAMA_BUILD_NUMBER}, {"commit", LLAMA_COMMIT}});

    LOG_INFO("system info", {
                                {"n_threads", params.n_threads},
                                {"n_threads_batch", params.n_threads_batch},
                                {"total_threads", std::thread::hardware_concurrency()},
                                {"system_info", llama_print_system_info()},
                            });

    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    // Necessary similarity of prompt for slot selection
    ctx_server->slot_prompt_similarity = params.slot_prompt_similarity;

    // load the model
    if (!ctx_server->load_model(params))
    {
        state.store(SERVER_STATE_ERROR);
        env->ThrowNew(c_llama_error, "could not load model from given file path");
        return;
    }

    ctx_server->init();
    state.store(SERVER_STATE_READY);

    LOG_INFO("model loaded", {});

    const auto model_meta = ctx_server->model_meta();

    // if a custom chat template is not supplied, we will use the one that comes with the model (if any)
    if (params.chat_template.empty())
    {
        if (!ctx_server->validate_model_chat_template())
        {
            LOG_ERROR("The chat template that comes with this model is not yet supported, falling back to chatml. This "
                      "may cause the model to output suboptimal responses",
                      {});
            params.chat_template = "chatml";
        }
    }

    // if a custom chat template is not supplied, we will use the one that comes with the model (if any)
    if (params.chat_template.empty())
    {
        if (!ctx_server->validate_model_chat_template())
        {
            LOG_ERROR("The chat template that comes with this model is not yet supported, falling back to chatml. This "
                      "may cause the model to output suboptimal responses",
                      {});
            params.chat_template = "chatml";
        }
    }

    // print sample chat example to make it clear which template is used
    {
        LOG_INFO("chat template",
                 {
                     {"chat_example", llama_chat_format_example(ctx_server->model, params.chat_template)},
                     {"built_in", params.chat_template.empty()},
                 });
    }

    ctx_server->queue_tasks.on_new_task(
        std::bind(&server_context::process_single_task, ctx_server, std::placeholders::_1));
    ctx_server->queue_tasks.on_finish_multitask(
        std::bind(&server_context::on_finish_multitask, ctx_server, std::placeholders::_1));
    ctx_server->queue_tasks.on_update_slots(std::bind(&server_context::update_slots, ctx_server));
    ctx_server->queue_results.on_multitask_update(std::bind(&server_queue::update_multitask, &ctx_server->queue_tasks,
                                                            std::placeholders::_1, std::placeholders::_2,
                                                            std::placeholders::_3));

    std::thread t([ctx_server]() {
        JNIEnv *env;
        jint res = g_vm->GetEnv((void **)&env, JNI_VERSION_1_6);
        if (res == JNI_EDETACHED)
        {
            res = g_vm->AttachCurrentThread((void **)&env, nullptr);
            if (res != JNI_OK)
            {
                throw std::runtime_error("Failed to attach thread to JVM");
            }
        }
        ctx_server->queue_tasks.start_loop();
    });
    t.detach();

    env->SetLongField(obj, f_model_pointer, reinterpret_cast<jlong>(ctx_server));
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_requestCompletion(JNIEnv *env, jobject obj, jstring jparams)
{
    jlong server_handle = env->GetLongField(obj, f_model_pointer);
    auto *ctx_server = reinterpret_cast<server_context *>(server_handle); // NOLINT(*-no-int-to-ptr)

    std::string c_params = parse_jstring(env, jparams);
    json json_params = json::parse(c_params);
    const bool infill = json_params.contains("input_prefix") || json_params.contains("input_suffix");

    if (json_params.value("use_chat_template", false))
    {
        json chat;
        chat.push_back({{"role", "system"}, {"content", ctx_server->system_prompt}});
        chat.push_back({{"role", "user"}, {"content", json_params["prompt"]}});
        json_params["prompt"] = format_chat(ctx_server->model, ctx_server->params.chat_template, chat);
    }

    const int id_task = ctx_server->queue_tasks.get_new_id();
    ctx_server->queue_results.add_waiting_task_id(id_task);
    ctx_server->request_completion(id_task, -1, json_params, infill, false);

    return id_task;
}

JNIEXPORT jobject JNICALL Java_de_kherud_llama_LlamaModel_receiveCompletion(JNIEnv *env, jobject obj, jint id_task)
{
    jlong server_handle = env->GetLongField(obj, f_model_pointer);
    auto *ctx_server = reinterpret_cast<server_context *>(server_handle); // NOLINT(*-no-int-to-ptr)

    server_task_result result = ctx_server->queue_results.recv(id_task);

    if (result.error)
    {
        std::string response = result.data["message"].get<std::string>();
        ctx_server->queue_results.remove_waiting_task_id(id_task);
        env->ThrowNew(c_llama_error, response.c_str());
        return nullptr;
    }

    std::string response = result.data["content"].get<std::string>();
    if (result.stop)
    {
        ctx_server->queue_results.remove_waiting_task_id(id_task);
    }

    jobject o_probabilities = env->NewObject(c_hash_map, cc_hash_map);
    if (result.data.contains("completion_probabilities"))
    {
        auto completion_probabilities = result.data["completion_probabilities"];
        for (const auto &entry : completion_probabilities)
        {
            auto probs = entry["probs"];
            for (const auto &tp : probs)
            {
                std::string tok_str = tp["tok_str"];
                jstring jtok_str = env->NewStringUTF(tok_str.c_str());
                float prob = tp["prob"];
                jobject jprob = env->NewObject(c_float, cc_float, prob);
                env->CallObjectMethod(o_probabilities, m_map_put, jtok_str, jprob);
                env->DeleteLocalRef(jtok_str);
                env->DeleteLocalRef(jprob);
            }
        }
    }

    jbyteArray jbytes = parse_jbytes(env, response);
    return env->NewObject(c_output, cc_output, jbytes, o_probabilities, result.stop);
}

JNIEXPORT jfloatArray JNICALL Java_de_kherud_llama_LlamaModel_embed(JNIEnv *env, jobject obj, jstring jprompt)
{
    jlong server_handle = env->GetLongField(obj, f_model_pointer);
    auto *ctx_server = reinterpret_cast<server_context *>(server_handle); // NOLINT(*-no-int-to-ptr)

    if (!ctx_server->params.embedding)
    {
        env->ThrowNew(c_llama_error,
                      "model was not loaded with embedding support (see ModelParameters#setEmbedding(boolean))");
        return nullptr;
    }

    const std::string prompt = parse_jstring(env, jprompt);

    const int id_task = ctx_server->queue_tasks.get_new_id();
    ctx_server->queue_results.add_waiting_task_id(id_task);
    ctx_server->request_completion(id_task, -1, {{"prompt", prompt}}, false, true);

    server_task_result result = ctx_server->queue_results.recv(id_task);
    ctx_server->queue_results.remove_waiting_task_id(id_task);
    if (result.error)
    {
        std::string response = result.data["message"].get<std::string>();
        env->ThrowNew(c_llama_error, response.c_str());
        return nullptr;
    }

    std::vector<float> embedding = result.data["embedding"].get<std::vector<float>>();
    jsize embedding_size = embedding.size(); // NOLINT(*-narrowing-conversions)

    jfloatArray j_embedding = env->NewFloatArray(embedding_size);
    if (j_embedding == nullptr)
    {
        env->ThrowNew(c_error_oom, "could not allocate embedding");
        return nullptr;
    }

    env->SetFloatArrayRegion(j_embedding, 0, embedding_size, reinterpret_cast<const jfloat *>(embedding.data()));

    return j_embedding;
}

JNIEXPORT jintArray JNICALL Java_de_kherud_llama_LlamaModel_encode(JNIEnv *env, jobject obj, jstring jprompt)
{
    jlong server_handle = env->GetLongField(obj, f_model_pointer);
    auto *ctx_server = reinterpret_cast<server_context *>(server_handle); // NOLINT(*-no-int-to-ptr)

    const std::string c_prompt = parse_jstring(env, jprompt);
    std::vector<llama_token> tokens = ctx_server->tokenize(c_prompt, false);
    jsize token_size = tokens.size(); // NOLINT(*-narrowing-conversions)

    jintArray java_tokens = env->NewIntArray(token_size);
    if (java_tokens == nullptr)
    {
        env->ThrowNew(c_error_oom, "could not allocate token memory");
        return nullptr;
    }

    env->SetIntArrayRegion(java_tokens, 0, token_size, reinterpret_cast<const jint *>(tokens.data()));

    return java_tokens;
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_decodeBytes(JNIEnv *env, jobject obj,
                                                                         jintArray java_tokens)
{
    jlong server_handle = env->GetLongField(obj, f_model_pointer);
    auto *ctx_server = reinterpret_cast<server_context *>(server_handle); // NOLINT(*-no-int-to-ptr)

    jsize length = env->GetArrayLength(java_tokens);
    jint *elements = env->GetIntArrayElements(java_tokens, nullptr);
    std::vector<llama_token> tokens(elements, elements + length);
    std::string text = tokens_to_str(ctx_server->ctx, tokens.cbegin(), tokens.cend());

    env->ReleaseIntArrayElements(java_tokens, elements, 0);

    return parse_jbytes(env, text);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_delete(JNIEnv *env, jobject obj)
{
    jlong server_handle = env->GetLongField(obj, f_model_pointer);
    auto *ctx_server = reinterpret_cast<server_context *>(server_handle); // NOLINT(*-no-int-to-ptr)
    ctx_server->queue_tasks.terminate();
    delete ctx_server;
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_cancelCompletion(JNIEnv *env, jobject obj, jint id_task)
{
    jlong server_handle = env->GetLongField(obj, f_model_pointer);
    auto *ctx_server = reinterpret_cast<server_context *>(server_handle); // NOLINT(*-no-int-to-ptr)
    ctx_server->request_cancel(id_task);
    ctx_server->queue_results.remove_waiting_task_id(id_task);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setLogger(JNIEnv *env, jclass clazz, jobject log_format,
                                                                 jobject jcallback)
{
    if (o_log_callback != nullptr)
    {
        env->DeleteGlobalRef(o_log_callback);
    }

    log_json = env->IsSameObject(log_format, o_log_format_json);

    if (jcallback == nullptr)
    {
        log_callback = nullptr;
        llama_log_set(nullptr, nullptr);
    }
    else
    {
        o_log_callback = env->NewGlobalRef(jcallback);
        log_callback = [](enum ggml_log_level level, const char *text, void *user_data) {
            JNIEnv *env = get_jni_env();
            jstring message = env->NewStringUTF(text);
            jobject log_level = log_level_to_jobject(level);
            env->CallVoidMethod(o_log_callback, m_biconsumer_accept, log_level, message);
            env->DeleteLocalRef(message);
        };
        if (!log_json)
        {
            llama_log_set(log_callback_trampoline, nullptr);
        }
    }
}
