#include "jllama.h"

#include "json.hpp"
#include "llama.h"
#include "server.hpp"
#include "utils.hpp"

// We store some references to Java classes and their fields/methods here to speed up things for later and to fail
// early on if anything can't be found. This happens when the JVM loads the shared library (see `JNI_OnLoad`).
// The references remain valid throughout the whole life of the shared library, on `JNI_OnUnload` they are released.

JavaVM *g_vm = nullptr;

// classes
static jclass c_llama_model = 0;
static jclass c_llama_iterator = 0;
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
static jfieldID f_utf_8 = 0;
// iterator
static jfieldID f_iter_has_next = 0;
static jfieldID f_iter_n_generated = 0;
static jfieldID f_iter_token_index = 0;

// objects
static jobject o_utf_8 = 0;

/**
 * Convert a Java string to a std::string
 */
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

/**
 * Since Java expects utf16 but std::strings are utf8, we can't directly use `env->NewString` or `env-NewString`,
 * but we directly send the bytes and do the conversion in Java. Unfortunately, there isn't a nice/standardized way to
 * do this conversion in C++
 */
static jbyteArray parse_jbytes(JNIEnv *env, std::string string)
{
    jsize len = string.size();
    jbyteArray bytes = env->NewByteArray(len);
    env->SetByteArrayRegion(bytes, 0, len, reinterpret_cast<const jbyte *>(string.c_str()));
    return bytes;
}

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
    JNIEnv *env = 0;

    if (JNI_OK != vm->GetEnv((void **)&env, JNI_VERSION_1_1))
    {
        goto error;
    }

    // find classes
    c_llama_model = env->FindClass("de/kherud/llama/LlamaModel");
    c_llama_iterator = env->FindClass("de/kherud/llama/LlamaModel$LlamaIterator");
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
    c_biconsumer = env->FindClass("java/util/function/BiConsumer");
    c_llama_error = env->FindClass("de/kherud/llama/LlamaException");
    c_error_oom = env->FindClass("java/lang/OutOfMemoryError");

    if (!(c_llama_model && c_llama_iterator && c_standard_charsets && c_output && c_string && c_hash_map && c_map &&
          c_set && c_entry && c_iterator && c_integer && c_float && c_biconsumer && c_llama_error && c_error_oom))
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
    f_utf_8 = env->GetStaticFieldID(c_standard_charsets, "UTF_8", "Ljava/nio/charset/Charset;");
    f_iter_has_next = env->GetFieldID(c_llama_iterator, "hasNext", "Z");
    f_iter_n_generated = env->GetFieldID(c_llama_iterator, "generatedCount", "J");
    f_iter_token_index = env->GetFieldID(c_llama_iterator, "tokenIndex", "J");

    if (!(f_model_pointer && f_utf_8 && f_iter_has_next && f_iter_n_generated && f_iter_token_index))
    {
        goto error;
    }

    o_utf_8 = env->NewStringUTF("UTF-8");

    if (!(o_utf_8))
    {
        goto error;
    }

    o_utf_8 = (jclass)env->NewGlobalRef(o_utf_8);

    if (env->ExceptionCheck())
    {
        env->ExceptionDescribe();
        goto error;
    }

    goto success;

error:
    return JNI_ERR;

success:
    return JNI_VERSION_1_2;
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
    JNIEnv *env = 0;

    if (JNI_OK != vm->GetEnv((void **)&env, JNI_VERSION_1_1))
        return;

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
    env->DeleteGlobalRef(c_error_oom);

    env->DeleteGlobalRef(o_utf_8);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_loadModel(JNIEnv *env, jobject obj, jstring jparams)
{
    gpt_params params;
    server_params sparams;

    server_context *ctx_server = new server_context();

    std::string c_params = parse_jstring(env, jparams);
    json json_params = json::parse(c_params);
    server_params_parse(json_params, sparams, params);

    if (!sparams.system_prompt.empty())
    {
        ctx_server->system_prompt_set(sparams.system_prompt);
    }

    if (params.model_alias == "unknown")
    {
        params.model_alias = params.model;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INFO("build info", {{"build", LLAMA_BUILD_NUMBER}, {"commit", LLAMA_COMMIT}});

    LOG_INFO("system info", {
                                {"n_threads", params.n_threads},
                                {"n_threads_batch", params.n_threads_batch},
                                {"total_threads", std::thread::hardware_concurrency()},
                                {"system_info", llama_print_system_info()},
                            });

    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    // load the model
    if (!ctx_server->load_model(params))
    {
        state.store(SERVER_STATE_ERROR);
        env->ThrowNew(c_llama_error, "could not load model from given file path");
        return;
    }
    else
    {
        ctx_server->init();
        state.store(SERVER_STATE_READY);
    }

    LOG_INFO("model loaded", {});

    const auto model_meta = ctx_server->model_meta();

    // if a custom chat template is not supplied, we will use the one that comes with the model (if any)
    if (sparams.chat_template.empty())
    {
        if (!ctx_server->validate_model_chat_template())
        {
            LOG_ERROR("The chat template that comes with this model is not yet supported, falling back to chatml. This "
                      "may cause the model to output suboptimal responses",
                      {});
            sparams.chat_template = "chatml";
        }
    }

    // print sample chat example to make it clear which template is used
    {
        json chat;
        chat.push_back({{"role", "system"}, {"content", "You are a helpful assistant"}});
        chat.push_back({{"role", "user"}, {"content", "Hello"}});
        chat.push_back({{"role", "assistant"}, {"content", "Hi there"}});
        chat.push_back({{"role", "user"}, {"content", "How are you?"}});

        const std::string chat_example = format_chat(ctx_server->model, sparams.chat_template, chat);

        LOG_INFO("chat template", {
                                      {"chat_example", chat_example},
                                      {"built_in", sparams.chat_template.empty()},
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

    std::thread t([ctx_server]() { ctx_server->queue_tasks.start_loop(); });
    t.detach();

    env->SetLongField(obj, f_model_pointer, reinterpret_cast<jlong>(ctx_server));
}

// JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_newAnswerIterator(JNIEnv *env, jobject obj, jstring prompt,
//                                                                          jobject params)
//{
//     jlong llama_handle = env->GetLongField(obj, f_model_pointer);
//     jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
//
//     //    auto lock = llama->lock();
//
//     llama->rewind();
//
//     llama_reset_timings(llama->ctx);
//
//     setup_answering(env, llama, prompt, params);
//
//     llama->loadPrompt();
//     llama->beginCompletion();
// }
//
// JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_newInfillIterator(JNIEnv *env, jobject obj, jstring prefix,
//                                                                          jstring suffix, jobject params)
//{
//     jlong llama_handle = env->GetLongField(obj, f_model_pointer);
//     jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
//
//     //    auto lock = llama->lock();
//
//     llama->rewind();
//
//     llama_reset_timings(llama->ctx);
//
//     setup_infilling(env, llama, prefix, suffix, params);
//
//     llama->loadInfill();
//     llama->beginCompletion();
// }
//
// JNIEXPORT jobject JNICALL Java_de_kherud_llama_LlamaModel_getNext(JNIEnv *env, jobject obj, jobject iter)
//{
//     jlong llama_handle = env->GetLongField(obj, f_model_pointer);
//     jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
//
//     size_t sent_count = env->GetLongField(iter, f_iter_n_generated);
//     size_t sent_token_probs_index = env->GetLongField(iter, f_iter_token_index);
//
//     completion_token_output token_with_probs;
//     while (llama->has_next_token)
//     {
//         token_with_probs = llama->doCompletion();
//         if (token_with_probs.tok >= 0 && llama->multibyte_pending <= 0)
//         {
//             break;
//         }
//     }
//     const std::string token_text = llama_token_to_piece(llama->ctx, token_with_probs.tok);
//
//     size_t pos = std::min(sent_count, llama->generated_text.size());
//
//     const std::string str_test = llama->generated_text.substr(pos);
//     bool is_stop_full = false;
//     size_t stop_pos = llama->findStoppingStrings(str_test, token_text.size(), STOP_FULL);
//     if (stop_pos != std::string::npos)
//     {
//         is_stop_full = true;
//         llama->generated_text.erase(llama->generated_text.begin() + pos + stop_pos, llama->generated_text.end());
//         pos = std::min(sent_count, llama->generated_text.size());
//     }
//     else
//     {
//         is_stop_full = false;
//         stop_pos = llama->findStoppingStrings(str_test, token_text.size(), STOP_PARTIAL);
//     }
//
//     std::string to_send;
//     if (stop_pos == std::string::npos ||
//         // Send rest of the text if we are at the end of the generation
//         (!llama->has_next_token && !is_stop_full && stop_pos > 0))
//     {
//         to_send = llama->generated_text.substr(pos, std::string::npos);
//
//         sent_count += to_send.size();
//         env->SetLongField(iter, f_iter_n_generated, sent_count);
//
//         std::vector<completion_token_output> probs_output = {};
//
//         if (llama->params.sparams.n_probs > 0)
//         {
//             const std::vector<llama_token> to_send_toks =
//                 llama_tokenize(llama->ctx, to_send, false, llama->tokenize_special);
//             size_t probs_pos = std::min(sent_token_probs_index, llama->generated_token_probs.size());
//             size_t probs_stop_pos =
//                 std::min(sent_token_probs_index + to_send_toks.size(), llama->generated_token_probs.size());
//             if (probs_pos < probs_stop_pos)
//             {
//                 probs_output =
//                     std::vector<completion_token_output>(llama->generated_token_probs.begin() + probs_pos,
//                                                          llama->generated_token_probs.begin() + probs_stop_pos);
//             }
//             sent_token_probs_index = probs_stop_pos;
//             env->SetLongField(iter, f_iter_token_index, sent_token_probs_index);
//         }
//     }
//     else
//     {
//         to_send = "";
//     }
//
//     if (!llama->has_next_token)
//     {
//         env->SetBooleanField(iter, f_iter_has_next, false);
//         // llama.mutex.unlock();
//         // lock.release();
//     }
//
//     jobject o_probabilities = env->NewObject(c_hash_map, cc_hash_map);
//     for (const auto &tp : token_with_probs.probs)
//     {
//         jobject jtoken = env->NewObject(c_integer, cc_integer, tp.tok);
//         jobject jprob = env->NewObject(c_float, cc_float, tp.prob);
//         env->CallObjectMethod(o_probabilities, m_map_put, jtoken, jprob);
//     }
//     jbyteArray jbytes = parse_jbytes(env, to_send);
//     return env->NewObject(c_output, cc_output, token_with_probs.tok, jbytes, o_probabilities);
// }
//
JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_getAnswer(JNIEnv *env, jobject obj, jstring jprompt,
                                                                       jstring jparams)
{
    jlong server_handle = env->GetLongField(obj, f_model_pointer);
    server_context *ctx_server = reinterpret_cast<server_context *>(server_handle);

    std::string c_params = parse_jstring(env, jparams);
    json json_params = json::parse(c_params);
    json_params["prompt"] = parse_jstring(env, jprompt);

    const int id_task = ctx_server->queue_tasks.get_new_id();
    ctx_server->queue_results.add_waiting_task_id(id_task);
    ctx_server->request_completion(id_task, -1, json_params, false, false);

    server_task_result result = ctx_server->queue_results.recv(id_task);

    if (!result.error && result.stop)
    {
        std::string response = result.data["content"].get<std::string>();
        ctx_server->queue_results.remove_waiting_task_id(id_task);
        return parse_jbytes(env, response);
    }
    else
    {
        std::string response = result.data["message"].get<std::string>();
        env->ThrowNew(c_llama_error, response.c_str());
        return nullptr;
    }
}
//
// JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_getInfill(JNIEnv *env, jobject obj, jstring prefix,
//                                                                        jstring suffix, jobject params)
//{
//     jlong llama_handle = env->GetLongField(obj, f_model_pointer);
//     jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
//
//     //	auto lock = llama->lock();
//
//     llama->rewind();
//
//     llama_reset_timings(llama->ctx);
//
//     setup_infilling(env, llama, prefix, suffix, params);
//
//     llama->loadInfill();
//     llama->beginCompletion();
//
//     size_t stop_pos = std::string::npos;
//
//     while (llama->has_next_token)
//     {
//         const completion_token_output token_with_probs = llama->doCompletion();
//         const std::string token_text =
//             token_with_probs.tok == -1 ? "" : llama_token_to_piece(llama->ctx, token_with_probs.tok);
//
//         stop_pos = llama->findStoppingStrings(llama->generated_text, token_text.size(), STOP_FULL);
//     }
//
//     if (stop_pos == std::string::npos)
//     {
//         stop_pos = llama->findStoppingStrings(llama->generated_text, 0, STOP_PARTIAL);
//     }
//     if (stop_pos != std::string::npos)
//     {
//         llama->generated_text.erase(llama->generated_text.begin() + stop_pos, llama->generated_text.end());
//     }
//
//     //	llama->lock().release();
//     //	llama->mutex.unlock();
//
//     return parse_jbytes(env, llama->generated_text);
// }
//
// JNIEXPORT jfloatArray JNICALL Java_de_kherud_llama_LlamaModel_embed(JNIEnv *env, jobject obj, jstring java_prompt)
//{
//     jlong llama_handle = env->GetLongField(obj, f_model_pointer);
//     jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
//
//     //	auto lock = llama->lock();
//
//     llama->rewind();
//     llama_reset_timings(llama->ctx);
//     llama->prompt = parse_jstring(env, java_prompt);
//     llama->params.n_predict = 0;
//     llama->loadPrompt();
//     llama->beginCompletion();
//     llama->doCompletion();
//
//     static const int n_embd = llama_n_embd(llama->model);
//     const float *data = llama_get_embeddings(llama->ctx);
//     std::vector<float> embedding(data, data + n_embd);
//
//     jfloatArray java_embedding = env->NewFloatArray(embedding.size());
//     if (java_embedding == nullptr)
//     {
//         env->ThrowNew(c_error_oom, "could not allocate embedding");
//         return nullptr;
//     }
//
//     env->SetFloatArrayRegion(java_embedding, 0, embedding.size(), reinterpret_cast<const jfloat
//     *>(embedding.data()));
//
//     return java_embedding;
// }
//
// JNIEXPORT jintArray JNICALL Java_de_kherud_llama_LlamaModel_encode(JNIEnv *env, jobject obj, jstring jprompt)
//{
//     jlong llama_handle = env->GetLongField(obj, f_model_pointer);
//     jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
//
//     //	auto lock = llama->lock();
//
//     std::string prompt = parse_jstring(env, jprompt);
//     std::vector<llama_token> tokens = llama->tokenize(prompt, false);
//
//     jintArray java_tokens = env->NewIntArray(tokens.size());
//     if (java_tokens == nullptr)
//     {
//         env->ThrowNew(c_error_oom, "could not allocate tokens");
//         return nullptr;
//     }
//
//     env->SetIntArrayRegion(java_tokens, 0, tokens.size(), reinterpret_cast<const jint *>(tokens.data()));
//
//     //	lock.release();
//     return java_tokens;
// }
//
// JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_decodeBytes(JNIEnv *env, jobject obj,
//                                                                          jintArray java_tokens)
//{
//     jlong llama_handle = env->GetLongField(obj, f_model_pointer);
//     jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
//
//     //    auto lock = llama->lock();
//
//     jsize length = env->GetArrayLength(java_tokens);
//     jint *elements = env->GetIntArrayElements(java_tokens, nullptr);
//     std::vector<llama_token> tokens(elements, elements + length);
//     std::string text = tokens_to_str(llama->ctx, tokens.cbegin(), tokens.cend());
//
//     env->ReleaseIntArrayElements(java_tokens, elements, 0);
//
//     //	lock.release();
//     return parse_jbytes(env, text);
// }
//
// JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setLogger(JNIEnv *env, jclass clazz, jobject callback)
//{
//     env->GetJavaVM(&g_vm);
//
//     if (g_log_callback != nullptr)
//     {
//         env->DeleteGlobalRef(g_log_callback);
//     }
//
//     if (callback == nullptr)
//     {
//         llama_log_set(nullptr, nullptr);
//     }
//     else
//     {
//         g_log_callback = env->NewGlobalRef(callback);
//         llama_log_set(jllama_log_callback, nullptr);
//     }
// }
//
// JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_delete(JNIEnv *env, jobject obj)
//{
//     jlong llama_handle = env->GetLongField(obj, f_model_pointer);
//     jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
//     delete llama;
// }
