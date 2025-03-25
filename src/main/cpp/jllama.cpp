#include "jllama.h"
#include "arg.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "log.h"
#include "nlohmann/json.hpp"
#include "server.hpp"
#include <functional>
#include <iostream>
#include <stdexcept>

// We store some references to Java classes and their fields/methods here to speed up things for later and to fail
// early on if anything can't be found. This happens when the JVM loads the shared library (see `JNI_OnLoad`).
// The references remain valid throughout the whole life of the shared library, on `JNI_OnUnload` they are released.

namespace {
  JavaVM * g_vm = nullptr;

  // classes
  jclass c_llama_model = nullptr;
  jclass c_standard_charsets = nullptr;
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
  std::string parse_jstring(JNIEnv * env, jstring java_string) {
    auto *
      const string_bytes = (jbyteArray) env -> CallObjectMethod(java_string, m_get_bytes, o_utf_8);

    auto length = (size_t) env -> GetArrayLength(string_bytes);
    jbyte * byte_elements = env -> GetByteArrayElements(string_bytes, nullptr);

    std::string string = std::string((char * ) byte_elements, length);

    env -> ReleaseByteArrayElements(string_bytes, byte_elements, JNI_ABORT);
    env -> DeleteLocalRef(string_bytes);

    return string;
  }

  char ** parse_string_array(JNIEnv * env,
    const jobjectArray string_array,
      const jsize length) {
    auto *
      const result = static_cast < char ** > (malloc(length * sizeof(char * )));

    if (result == nullptr) {
      return nullptr;
    }

    for (jsize i = 0; i < length; i++) {
      auto *
        const javaString = static_cast < jstring > (env -> GetObjectArrayElement(string_array, i));
      const char * cString = env -> GetStringUTFChars(javaString, nullptr);
      result[i] = strdup(cString);
      env -> ReleaseStringUTFChars(javaString, cString);
    }

    return result;
  }

  void free_string_array(char ** array, jsize length) {
    if (array != nullptr) {
      for (jsize i = 0; i < length; i++) {
        free(array[i]);
      }
      free(array);
    }
  }

  /**
   * Since Java expects utf16 but std::strings are utf8, we can't directly use `env->NewString` or `env-NewString`,
   * but we directly send the bytes and do the conversion in Java. Unfortunately, there isn't a nice/standardized way to
   * do this conversion in C++
   */
  jbyteArray parse_jbytes(JNIEnv * env,
    const std::string & string) {
    jsize length = string.size(); // NOLINT(*-narrowing-conversions)
    jbyteArray bytes = env -> NewByteArray(length);
    env -> SetByteArrayRegion(bytes, 0, length, reinterpret_cast <
      const jbyte * > (string.c_str()));
    return bytes;
  }

  /**
   * Map a llama.cpp log level to its Java enumeration option.
   */
  jobject log_level_to_jobject(ggml_log_level level) {
    switch (level) {
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
  JNIEnv * get_jni_env() {
    JNIEnv * env = nullptr;
    if (g_vm == nullptr || g_vm -> GetEnv(reinterpret_cast < void ** > ( & env), JNI_VERSION_1_6) != JNI_OK) {
      throw std::runtime_error("Thread is not attached to the JVM");
    }
    return env;
  }

  bool log_json;
  std:: function < void(ggml_log_level,
    const char * , void * ) > log_callback;

	/**
 		* Format a log message as JSON
 	*/
	std::string format_log_as_json(ggml_log_level level, const char* text) {
    	std::string level_str;
    	switch (level) {
        	case GGML_LOG_LEVEL_ERROR: level_str = "ERROR"; break;
        	case GGML_LOG_LEVEL_WARN:  level_str = "WARN";  break;
        	case GGML_LOG_LEVEL_INFO:  level_str = "INFO";  break;
        	default:
        	case GGML_LOG_LEVEL_DEBUG: level_str = "DEBUG"; break;
    	}
    
    	// Create a JSON object with timestamp, level, and message
    	nlohmann::json log_json = {
        	{"timestamp", std::time(nullptr)},
        	{"level", level_str},
        	{"message", text}
    	};
    
    	return log_json.dump();
	}
  /**
   * Invoke the log callback if there is any.
   */
/**
 * Invoke the log callback if there is any.
 */
	void log_callback_trampoline(ggml_log_level level, const char* text, void* user_data) {
    	if (log_callback != nullptr) {
        	if (log_json) {
            	// Format the message as JSON before passing to callback
            	std::string json_text = format_log_as_json(level, text);
            	log_callback(level, json_text.c_str(), user_data);
        	} else {
            // Pass the original text
            log_callback(level, text, user_data);
        	}
    	}	
	}
} // namespace

/**
 * The VM calls JNI_OnLoad when the native library is loaded (for example, through `System.loadLibrary`).
 * `JNI_OnLoad` must return the JNI version needed by the native library.
 * In order to use any of the new JNI functions, a native library must export a `JNI_OnLoad` function that returns
 * `JNI_VERSION_1_2`. If the native library does not export a JNI_OnLoad function, the VM assumes that the library
 * only requires JNI version `JNI_VERSION_1_1`. If the VM does not recognize the version number returned by
 `JNI_OnLoad`, the VM will unload the library and act as if the library was never loaded.
 */
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM * vm, void * reserved) {
  g_vm = vm;
  JNIEnv * env = nullptr;

  if (JNI_OK != vm -> GetEnv((void ** ) & env, JNI_VERSION_1_1)) {
    goto error;
  }

  // find classes
  c_llama_model = env -> FindClass("de/kherud/llama/LlamaModel");
  c_standard_charsets = env -> FindClass("java/nio/charset/StandardCharsets");
  c_string = env -> FindClass("java/lang/String");
  c_hash_map = env -> FindClass("java/util/HashMap");
  c_map = env -> FindClass("java/util/Map");
  c_set = env -> FindClass("java/util/Set");
  c_entry = env -> FindClass("java/util/Map$Entry");
  c_iterator = env -> FindClass("java/util/Iterator");
  c_integer = env -> FindClass("java/lang/Integer");
  c_float = env -> FindClass("java/lang/Float");
  c_biconsumer = env -> FindClass("java/util/function/BiConsumer");
  c_llama_error = env -> FindClass("de/kherud/llama/LlamaException");
  c_log_level = env -> FindClass("de/kherud/llama/LogLevel");
  c_log_format = env -> FindClass("de/kherud/llama/args/LogFormat");
  c_error_oom = env -> FindClass("java/lang/OutOfMemoryError");

  if (!(c_llama_model &&  c_standard_charsets  && c_string && c_hash_map && c_map &&
      c_set && c_entry && c_iterator && c_integer && c_float && c_biconsumer && c_llama_error && c_log_level &&
      c_log_format && c_error_oom)) {
    goto error;
  }

  // create references
  c_llama_model = (jclass) env -> NewGlobalRef(c_llama_model);
  c_string = (jclass) env -> NewGlobalRef(c_string);
  c_hash_map = (jclass) env -> NewGlobalRef(c_hash_map);
  c_map = (jclass) env -> NewGlobalRef(c_map);
  c_set = (jclass) env -> NewGlobalRef(c_set);
  c_entry = (jclass) env -> NewGlobalRef(c_entry);
  c_iterator = (jclass) env -> NewGlobalRef(c_iterator);
  c_integer = (jclass) env -> NewGlobalRef(c_integer);
  c_float = (jclass) env -> NewGlobalRef(c_float);
  c_biconsumer = (jclass) env -> NewGlobalRef(c_biconsumer);
  c_llama_error = (jclass) env -> NewGlobalRef(c_llama_error);
  c_log_level = (jclass) env -> NewGlobalRef(c_log_level);
  c_log_format = (jclass) env -> NewGlobalRef(c_log_format);
  c_error_oom = (jclass) env -> NewGlobalRef(c_error_oom);

  // find constructors
  cc_hash_map = env -> GetMethodID(c_hash_map, "<init>", "()V");
  cc_integer = env -> GetMethodID(c_integer, "<init>", "(I)V");
  cc_float = env -> GetMethodID(c_float, "<init>", "(F)V");

  if (!(cc_hash_map && cc_integer && cc_float)) {
    goto error;
  }

  // find methods
  m_get_bytes = env -> GetMethodID(c_string, "getBytes", "(Ljava/lang/String;)[B");
  m_entry_set = env -> GetMethodID(c_map, "entrySet", "()Ljava/util/Set;");
  m_set_iterator = env -> GetMethodID(c_set, "iterator", "()Ljava/util/Iterator;");
  m_iterator_has_next = env -> GetMethodID(c_iterator, "hasNext", "()Z");
  m_iterator_next = env -> GetMethodID(c_iterator, "next", "()Ljava/lang/Object;");
  m_entry_key = env -> GetMethodID(c_entry, "getKey", "()Ljava/lang/Object;");
  m_entry_value = env -> GetMethodID(c_entry, "getValue", "()Ljava/lang/Object;");
  m_map_put = env -> GetMethodID(c_map, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
  m_int_value = env -> GetMethodID(c_integer, "intValue", "()I");
  m_float_value = env -> GetMethodID(c_float, "floatValue", "()F");
  m_biconsumer_accept = env -> GetMethodID(c_biconsumer, "accept", "(Ljava/lang/Object;Ljava/lang/Object;)V");

  if (!(m_get_bytes && m_entry_set && m_set_iterator && m_iterator_has_next && m_iterator_next && m_entry_key &&
      m_entry_value && m_map_put && m_int_value && m_float_value && m_biconsumer_accept)) {
    goto error;
  }

  // find fields
  f_model_pointer = env -> GetFieldID(c_llama_model, "ctx", "J");
  f_utf_8 = env -> GetStaticFieldID(c_standard_charsets, "UTF_8", "Ljava/nio/charset/Charset;");
  f_log_level_debug = env -> GetStaticFieldID(c_log_level, "DEBUG", "Lde/kherud/llama/LogLevel;");
  f_log_level_info = env -> GetStaticFieldID(c_log_level, "INFO", "Lde/kherud/llama/LogLevel;");
  f_log_level_warn = env -> GetStaticFieldID(c_log_level, "WARN", "Lde/kherud/llama/LogLevel;");
  f_log_level_error = env -> GetStaticFieldID(c_log_level, "ERROR", "Lde/kherud/llama/LogLevel;");
  f_log_format_json = env -> GetStaticFieldID(c_log_format, "JSON", "Lde/kherud/llama/args/LogFormat;");
  f_log_format_text = env -> GetStaticFieldID(c_log_format, "TEXT", "Lde/kherud/llama/args/LogFormat;");

  if (!(f_model_pointer  && f_utf_8  && f_log_level_debug && f_log_level_info &&
      f_log_level_warn && f_log_level_error && f_log_format_json && f_log_format_text)) {
    goto error;
  }

  o_utf_8 = env -> NewStringUTF("UTF-8");
  o_log_level_debug = env -> GetStaticObjectField(c_log_level, f_log_level_debug);
  o_log_level_info = env -> GetStaticObjectField(c_log_level, f_log_level_info);
  o_log_level_warn = env -> GetStaticObjectField(c_log_level, f_log_level_warn);
  o_log_level_error = env -> GetStaticObjectField(c_log_level, f_log_level_error);
  o_log_format_json = env -> GetStaticObjectField(c_log_format, f_log_format_json);
  o_log_format_text = env -> GetStaticObjectField(c_log_format, f_log_format_text);

  if (!(o_utf_8 && o_log_level_debug && o_log_level_info && o_log_level_warn && o_log_level_error &&
      o_log_format_json && o_log_format_text)) {
    goto error;
  }

  o_utf_8 = env -> NewGlobalRef(o_utf_8);
  o_log_level_debug = env -> NewGlobalRef(o_log_level_debug);
  o_log_level_info = env -> NewGlobalRef(o_log_level_info);
  o_log_level_warn = env -> NewGlobalRef(o_log_level_warn);
  o_log_level_error = env -> NewGlobalRef(o_log_level_error);
  o_log_format_json = env -> NewGlobalRef(o_log_format_json);
  o_log_format_text = env -> NewGlobalRef(o_log_format_text);

  if (env -> ExceptionCheck()) {
    env -> ExceptionDescribe();
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
JNIEXPORT void JNICALL JNI_OnUnload(JavaVM * vm, void * reserved) {
  JNIEnv * env = nullptr;

  if (JNI_OK != vm -> GetEnv((void ** ) & env, JNI_VERSION_1_6)) {
    return;
  }

  env -> DeleteGlobalRef(c_llama_model);
  env -> DeleteGlobalRef(c_string);
  env -> DeleteGlobalRef(c_hash_map);
  env -> DeleteGlobalRef(c_map);
  env -> DeleteGlobalRef(c_set);
  env -> DeleteGlobalRef(c_entry);
  env -> DeleteGlobalRef(c_iterator);
  env -> DeleteGlobalRef(c_integer);
  env -> DeleteGlobalRef(c_float);
  env -> DeleteGlobalRef(c_biconsumer);
  env -> DeleteGlobalRef(c_llama_error);
  env -> DeleteGlobalRef(c_log_level);
  env -> DeleteGlobalRef(c_log_level);
  env -> DeleteGlobalRef(c_error_oom);

  env -> DeleteGlobalRef(o_utf_8);
  env -> DeleteGlobalRef(o_log_level_debug);
  env -> DeleteGlobalRef(o_log_level_info);
  env -> DeleteGlobalRef(o_log_level_warn);
  env -> DeleteGlobalRef(o_log_level_error);
  env -> DeleteGlobalRef(o_log_format_json);
  env -> DeleteGlobalRef(o_log_format_text);

  if (o_log_callback != nullptr) {
    env -> DeleteGlobalRef(o_log_callback);
  }

  llama_backend_free();
}

/**
 * Load a model with the given parameters.
 * This function initializes the server context and loads the language model.
 */
JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_loadModel(JNIEnv* env, jobject obj, jobjectArray jparams) {
    common_params params;

    const jsize argc = env->GetArrayLength(jparams);
    char** argv = parse_string_array(env, jparams, argc);
    if (argv == nullptr) {
        env->ThrowNew(c_error_oom, "Failed to allocate memory for parameters");
        return;
    }

    const auto parsed_params = common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER);
    free_string_array(argv, argc);
    if (!parsed_params) {
        env->ThrowNew(c_llama_error, "Failed to parse parameters");
        return;
    }

    SRV_INF("loading model '%s'\n", params.model.c_str());

    common_init();

    // Create server context structure that contains llama context and inference
    auto* ctx_server = new server_context();

    // Initialize NUMA if configured
    llama_numa_init(params.numa);

    // Log system information
    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", 
            params.cpuparams.n_threads, params.cpuparams_batch.n_threads, 
            std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    // Initialize server state
    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    // Set prompt similarity threshold for slot selection
    ctx_server->slot_prompt_similarity = params.slot_prompt_similarity;

    LOG_INF("%s: loading model\n", __func__);

    // Load the model
    if (!ctx_server->load_model(params)) {
        delete ctx_server;
        llama_backend_free();
        env->ThrowNew(c_llama_error, "Could not load model from given file path");
        return;
    }

    // Initialize the server context
    ctx_server->init();
    state.store(SERVER_STATE_READY);

    LOG_INF("%s: model loaded\n", __func__);

    // Load draft model if configured (for speculative decoding)
    if (!params.speculative.model.empty() || !params.speculative.hf_repo.empty()) {
        SRV_INF("loading draft model '%s'\n", params.speculative.model.c_str());
        auto params_dft = params;

        params_dft.devices = params.speculative.devices;
        params_dft.hf_file = params.speculative.hf_file;
        params_dft.hf_repo = params.speculative.hf_repo;
        params_dft.model = params.speculative.model;
        params_dft.model_url = params.speculative.model_url;
        params_dft.n_ctx = params.speculative.n_ctx == 0 ? params.n_ctx / params.n_parallel : params.speculative.n_ctx;
        params_dft.n_gpu_layers = params.speculative.n_gpu_layers;
        params_dft.n_parallel = 1;

        common_init_result llama_init_dft = common_init_from_params(params_dft);
        llama_model* model_dft = llama_init_dft.model.get();

        if (model_dft == nullptr) {
            SRV_ERR("failed to load draft model, '%s'\n", params.speculative.model.c_str());
        } else {
            if (!common_speculative_are_compatible(ctx_server->ctx, llama_init_dft.context.get())) {
                SRV_ERR("the draft model '%s' is not compatible with the target model '%s'\n",
                         params.speculative.model.c_str(), params.model.c_str());
            } else {
                const int n_ctx_dft = llama_n_ctx(llama_init_dft.context.get());
                ctx_server->cparams_dft = common_context_params_to_llama(params_dft);
                ctx_server->cparams_dft.n_batch = n_ctx_dft;
                
                // force F16 KV cache for the draft model for extra performance
                ctx_server->cparams_dft.type_k = GGML_TYPE_F16;
                ctx_server->cparams_dft.type_v = GGML_TYPE_F16;
                
                // the context is not needed - we will create one for each slot
                llama_init_dft.context.reset();
            }
        }
    }

    // Initialize chat templates
    ctx_server->chat_templates = common_chat_templates_init(ctx_server->model, params.chat_template);
    try {
        common_chat_format_example(ctx_server->chat_templates.get(), params.use_jinja);
    } catch (const std::exception& e) {
        SRV_WRN("%s: The chat template that comes with this model is not yet supported, falling back to chatml. This "
          "may cause the model to output suboptimal responses\n", __func__);
        ctx_server->chat_templates = common_chat_templates_init(ctx_server->model, "chatml");
    }

    // Print sample chat example to make it clear which template is used
    LOG_INF("%s: chat template, chat_template: %s, example_format: '%s'\n", __func__,
        common_chat_templates_source(ctx_server->chat_templates.get()),
        common_chat_format_example(ctx_server->chat_templates.get(), ctx_server->params_base.use_jinja).c_str());

    // Set up task handlers
    ctx_server->queue_tasks.on_new_task(
        std::bind(&server_context::process_single_task, ctx_server, std::placeholders::_1));
    ctx_server->queue_tasks.on_update_slots(std::bind(&server_context::update_slots, ctx_server));

    // Start task processing thread
    std::thread t([ctx_server]() {
        JNIEnv* env;
        jint res = g_vm->GetEnv((void**)&env, JNI_VERSION_1_6);
        if (res == JNI_EDETACHED) {
            res = g_vm->AttachCurrentThread((void**)&env, nullptr);
            if (res != JNI_OK) {
                throw std::runtime_error("Failed to attach thread to JVM");
            }
        }
        ctx_server->queue_tasks.start_loop();
    });
    t.detach();

    // Store server context pointer in Java object
    env->SetLongField(obj, f_model_pointer, reinterpret_cast<jlong>(ctx_server));
}

/**
 * Clean up resources and delete the model.
 * This function shuts down the server context and frees memory.
 */
JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_delete(JNIEnv* env, jobject obj) {
    try {
        jlong server_handle = env->GetLongField(obj, f_model_pointer);
        if (server_handle == 0) {
            return;  // Already deleted or not initialized
        }

        auto* ctx_server = reinterpret_cast<server_context*>(server_handle);
        
        // Log shutdown
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        
        // Cancel all pending tasks
        ctx_server->queue_tasks.terminate();
        
        // Wait for a brief moment to allow tasks to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Delete the server context
        delete ctx_server;
        
        // Clear the pointer in Java
        env->SetLongField(obj, f_model_pointer, 0);
        
        SRV_INF("%s: cleanup complete\n", __func__);
    } catch (const std::exception& e) {
        SRV_ERR("Exception during shutdown: %s\n", e.what());
        // We don't throw here, as this would prevent proper cleanup during JVM shutdown
    }
}

/**
 * Set a logger for llama.cpp logs.
 * This function configures the logging system to forward messages to Java.
 */
JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setLogger(JNIEnv* env, jclass clazz, jobject log_format, jobject jcallback) {
    if (o_log_callback != nullptr) {
        env->DeleteGlobalRef(o_log_callback);
        o_log_callback = nullptr;
    }

    log_json = env->IsSameObject(log_format, o_log_format_json);

    if (jcallback == nullptr) {
        // Disable logging if callback is null
        log_callback = nullptr;
        llama_log_set(nullptr, nullptr);
    } else {
        // Store a global reference to the callback object
        o_log_callback = env->NewGlobalRef(jcallback);
        
        // Create a C++ callback function that forwards to Java
        log_callback = [](enum ggml_log_level level, const char* text, void* user_data) {
            JNIEnv* env = get_jni_env();
            jstring message = env->NewStringUTF(text);
            jobject log_level = log_level_to_jobject(level);
            env->CallVoidMethod(o_log_callback, m_biconsumer_accept, log_level, message);
            env->DeleteLocalRef(message);
        };
        
        // Always set the logger, regardless of JSON format
        llama_log_set(log_callback_trampoline, nullptr);
        
        // For debugging, send an initial log message
        LOG_INF("Logger initialized (JSON format: %s)\n", log_json ? "true" : "false");

    }
}

/**
 * Handle standard completions request.
 * Equivalent to POST /completions endpoint.
 */
JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleCompletions(JNIEnv* env, jobject obj, jstring jrequestData, jboolean jstream) {
    try {
        // Get server context pointer from Java object
        jlong server_handle = env->GetLongField(obj, f_model_pointer);
        if (server_handle == 0) {
            env->ThrowNew(c_llama_error, "Model is not loaded");
            return nullptr;
        }

        auto* ctx_server = reinterpret_cast<server_context*>(server_handle);

        // Check if embeddings mode is active (which would prevent completions)
        if (ctx_server->params_base.embedding) {
            env->ThrowNew(c_llama_error, "This server does not support completions. Start it without `--embeddings`");
            return nullptr;
        }

        // Parse request data from JSON
        std::string request_str = parse_jstring(env, jrequestData);
        json data = json::parse(request_str);

        // Set streaming flag
        bool stream = jstream;
        data["stream"] = stream;

        // Create a completion ID
        auto completion_id = gen_chatcmplid();
        std::vector<server_task> tasks;

        try {
            // Extract prompt from request data
            const auto& prompt = data.at("prompt");
            
            // Tokenize prompt
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server->vocab, prompt, true, true);
            
            // Create tasks for each tokenized prompt
            tasks.reserve(tokenized_prompts.size());
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task(SERVER_TASK_TYPE_COMPLETION);

                task.id = ctx_server->queue_tasks.get_new_id();
                task.index = i;

                task.prompt_tokens = std::move(tokenized_prompts[i]);
                task.params = server_task::params_from_json_cmpl(
                    ctx_server->ctx, ctx_server->params_base, data);
                
                task.id_selected_slot = json_value(data, "id_slot", -1);
                
                // Set completion ID (but not OAI compatibility for standard completion)
                task.params.oaicompat = OAICOMPAT_TYPE_NONE;
                task.params.oaicompat_cmpl_id = completion_id;

                tasks.push_back(task);
            }
        } catch (const std::exception& e) {
            const auto& err = format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST);
            env->ThrowNew(c_llama_error, err.dump().c_str());
            return nullptr;
        }

        // Add tasks to waiting queue and post them for processing
        ctx_server->queue_results.add_waiting_tasks(tasks);
        ctx_server->queue_tasks.post(tasks);

        // Get task IDs
        const auto task_ids = server_task::get_list_id(tasks);

        // Create response JSON
        json response;

        if (!stream) {
            // For non-streaming, collect all results
            std::vector<server_task_result_ptr> results;
            results.reserve(tasks.size());

            for (size_t i = 0; i < tasks.size(); i++) {
                server_task_result_ptr result = ctx_server->queue_results.recv(task_ids);

                if (result->is_error()) {
                    // Clean up and throw error
                    ctx_server->queue_results.remove_waiting_task_ids(task_ids);
                    std::string error_msg = result->to_json()["message"].get<std::string>();
                    env->ThrowNew(c_llama_error, error_msg.c_str());
                    return nullptr;
                }

                results.push_back(std::move(result));
            }

            // Format the response
            response["type"] = "completion";
            response["streaming"] = false;
            response["completion_id"] = completion_id;

            if (results.size() == 1) {
                // Single result - preserve all the data including token probabilities
                auto result_json = results[0]->to_json();

                // Check if this is a final completion result that might have probabilities
                auto* cmpl_final = dynamic_cast<server_task_result_cmpl_final*>(results[0].get());

                if (cmpl_final != nullptr && !cmpl_final->probs_output.empty() && cmpl_final->post_sampling_probs) {
                    // Make sure the token probabilities are included
                    result_json["completion_probabilities"] =
                        completion_token_output::probs_vector_to_json(cmpl_final->probs_output,
                            cmpl_final->post_sampling_probs);
                }

                response["result"] = result_json;
            } else {
                // Multiple results
                json results_array = json::array();
                for (auto& res: results) {
                    auto result_json = res->to_json();

                    // Check for token probabilities in each result
                    auto* cmpl_final = dynamic_cast<server_task_result_cmpl_final*>(res.get());

                    if (cmpl_final != nullptr && !cmpl_final->probs_output.empty() && cmpl_final->post_sampling_probs) {
                        // Make sure the token probabilities are included
                        result_json["completion_probabilities"] =
                            completion_token_output::probs_vector_to_json(cmpl_final->probs_output,
                                cmpl_final->post_sampling_probs);
                    }

                    results_array.push_back(result_json);
                }
                response["results"] = results_array;
            }

            // Clean up
            ctx_server->queue_results.remove_waiting_task_ids(task_ids);
        } else {
            // For streaming, return the task IDs
            response["type"] = "stream_init";
            response["streaming"] = true;
            response["completion_id"] = completion_id;

            // Convert set to array
            json task_ids_array = json::array();
            for (const auto& id: task_ids) {
                task_ids_array.push_back(id);
            }
            response["task_ids"] = task_ids_array;

            SRV_INF("Started streaming completion with %zu task(s)\n", task_ids.size());
        }

        // Return the response as a JSON string
        std::string response_str = response.dump();
        jstring result = env->NewStringUTF(response_str.c_str());

        return result;
    } catch (const std::exception& e) {
        SRV_ERR("Exception in handleCompletions: %s\n", e.what());
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

/**
 * Handle OpenAI compatible completions request.
 * Equivalent to POST /v1/completions endpoint.
 */
JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleCompletionsOai(JNIEnv* env, jobject obj, jstring jrequestData, jboolean jstream) {
    try {
        // Get server context pointer from Java object
        jlong server_handle = env->GetLongField(obj, f_model_pointer);
        if (server_handle == 0) {
            env->ThrowNew(c_llama_error, "Model is not loaded");
            return nullptr;
        }

        auto* ctx_server = reinterpret_cast<server_context*>(server_handle);

        // Check if embeddings mode is active (which would prevent completions)
        if (ctx_server->params_base.embedding) {
            env->ThrowNew(c_llama_error, "This server does not support completions. Start it without `--embeddings`");
            return nullptr;
        }

        // Parse request data from JSON
        std::string request_str = parse_jstring(env, jrequestData);
        json body = json::parse(request_str);

        // Set streaming flag
        bool stream = jstream;
        body["stream"] = stream;

        // Parse the OpenAI-compatible parameters
        json data = oaicompat_completion_params_parse(body);

        // Create a completion ID
        auto completion_id = gen_chatcmplid();
        std::vector<server_task> tasks;

        try {
            // Extract prompt from request data
            const auto& prompt = data.at("prompt");
            
            // Tokenize prompt
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server->vocab, prompt, true, true);
            
            // Create tasks for each tokenized prompt
            tasks.reserve(tokenized_prompts.size());
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task(SERVER_TASK_TYPE_COMPLETION);

                task.id = ctx_server->queue_tasks.get_new_id();
                task.index = i;

                task.prompt_tokens = std::move(tokenized_prompts[i]);
                task.params = server_task::params_from_json_cmpl(
                    ctx_server->ctx, ctx_server->params_base, data);
                
                task.id_selected_slot = json_value(data, "id_slot", -1);
                
                // Set OAI compatibility mode
                task.params.oaicompat = OAICOMPAT_TYPE_COMPLETION;
                task.params.oaicompat_cmpl_id = completion_id;

                tasks.push_back(task);
            }
        } catch (const std::exception& e) {
            const auto& err = format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST);
            env->ThrowNew(c_llama_error, err.dump().c_str());
            return nullptr;
        }

        // Add tasks to waiting queue and post them for processing
        ctx_server->queue_results.add_waiting_tasks(tasks);
        ctx_server->queue_tasks.post(tasks);

        // Get task IDs
        const auto task_ids = server_task::get_list_id(tasks);

        // Create response JSON
        json response;

        if (!stream) {
            // For non-streaming, collect all results
            std::vector<server_task_result_ptr> results;
            results.reserve(tasks.size());

            for (size_t i = 0; i < tasks.size(); i++) {
                server_task_result_ptr result = ctx_server->queue_results.recv(task_ids);

                if (result->is_error()) {
                    // Clean up and throw error
                    ctx_server->queue_results.remove_waiting_task_ids(task_ids);
                    std::string error_msg = result->to_json()["message"].get<std::string>();
                    env->ThrowNew(c_llama_error, error_msg.c_str());
                    return nullptr;
                }

                results.push_back(std::move(result));
            }

            // Format the response
            response["type"] = "oai_completion";
            response["streaming"] = false;
            response["completion_id"] = completion_id;

            if (results.size() == 1) {
                // Single result
                response["result"] = results[0]->to_json();
            } else {
                // Multiple results
                json results_array = json::array();
                for (auto& res: results) {
                    results_array.push_back(res->to_json());
                }
                response["results"] = results_array;
            }

            // Clean up
            ctx_server->queue_results.remove_waiting_task_ids(task_ids);
        } else {
            // For streaming, return the task IDs
            response["type"] = "oai_stream_init";
            response["streaming"] = true;
            response["completion_id"] = completion_id;

            // Convert set to array
            json task_ids_array = json::array();
            for (const auto& id: task_ids) {
                task_ids_array.push_back(id);
            }
            response["task_ids"] = task_ids_array;

            SRV_INF("Started streaming OAI completion with %zu task(s)\n", task_ids.size());
        }

        // Return the response as a JSON string
        std::string response_str = response.dump();
        jstring result = env->NewStringUTF(response_str.c_str());

        return result;
    } catch (const std::exception& e) {
        SRV_ERR("Exception in handleCompletionsOai: %s\n", e.what());
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

/**
 * Handle chat completions request.
 * Equivalent to POST /chat/completions or POST /v1/chat/completions endpoints.
 */
JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleChatCompletions(JNIEnv* env, jobject obj, jstring jrequestData, jboolean jstream) {
    try {
        // Get server context pointer from Java object
        jlong server_handle = env->GetLongField(obj, f_model_pointer);
        if (server_handle == 0) {
            env->ThrowNew(c_llama_error, "Model is not loaded");
            return nullptr;
        }

        auto* ctx_server = reinterpret_cast<server_context*>(server_handle);

        // Check if embeddings mode is active (which would prevent completions)
        if (ctx_server->params_base.embedding) {
            env->ThrowNew(c_llama_error, "This server does not support completions. Start it without `--embeddings`");
            return nullptr;
        }

        // Parse request data from JSON
        std::string request_str = parse_jstring(env, jrequestData);
        json body = json::parse(request_str);
        
        // Log debug information
        LOG_DBG("Chat request: %s\n", request_str.c_str());

        // Set streaming flag
        bool stream = jstream;
        body["stream"] = stream;

        // Parse the OAI-compatible parameters with chat template application
        json data = oaicompat_completion_params_parse(
            body,
            ctx_server->params_base.use_jinja,
            ctx_server->params_base.reasoning_format,
            ctx_server->chat_templates.get());

        // Create a completion ID
        auto completion_id = gen_chatcmplid();
        std::vector<server_task> tasks;

        try {
            // Extract prompt from processed data
            const auto& prompt = data.at("prompt");
            
            // Tokenize prompt
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(
                ctx_server->vocab, prompt, true, true);

            // Create tasks for each tokenized prompt
            tasks.reserve(tokenized_prompts.size());
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task(SERVER_TASK_TYPE_COMPLETION);

                task.id = ctx_server->queue_tasks.get_new_id();
                task.index = i;

                task.prompt_tokens = std::move(tokenized_prompts[i]);
                task.params = server_task::params_from_json_cmpl(
                    ctx_server->ctx, ctx_server->params_base, data);

                task.id_selected_slot = json_value(data, "id_slot", -1);

                // Set OAI chat compatibility mode
                task.params.oaicompat = OAICOMPAT_TYPE_CHAT;
                task.params.oaicompat_cmpl_id = completion_id;

                tasks.push_back(task);
            }
        } catch (const std::exception& e) {
            const auto& err = format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST);
            env->ThrowNew(c_llama_error, err.dump().c_str());
            return nullptr;
        }

        // Add tasks to waiting queue and post them for processing
        ctx_server->queue_results.add_waiting_tasks(tasks);
        ctx_server->queue_tasks.post(tasks);

        // Get task IDs
        const auto task_ids = server_task::get_list_id(tasks);

        // Create response JSON
        json response;

        if (!stream) {
            // For non-streaming, collect all results
            std::vector<server_task_result_ptr> results;
            results.reserve(tasks.size());

            for (size_t i = 0; i < tasks.size(); i++) {
                server_task_result_ptr result = ctx_server->queue_results.recv(task_ids);

                if (result->is_error()) {
                    // Clean up and throw error
                    ctx_server->queue_results.remove_waiting_task_ids(task_ids);
                    std::string error_msg = result->to_json()["message"].get<std::string>();
                    env->ThrowNew(c_llama_error, error_msg.c_str());
                    return nullptr;
                }

                results.push_back(std::move(result));
            }

            // Format the response
            response["type"] = "oai_chat";
            response["streaming"] = false;
            response["completion_id"] = completion_id;

            if (results.size() == 1) {
                // Single result
                response["result"] = results[0]->to_json();
            } else {
                // Multiple results
                json results_array = json::array();
                for (auto& res: results) {
                    results_array.push_back(res->to_json());
                }
                response["results"] = results_array;
            }

            // Clean up
            ctx_server->queue_results.remove_waiting_task_ids(task_ids);
        } else {
            // For streaming, return the task IDs
            response["type"] = "oai_chat_stream_init";
            response["streaming"] = true;
            response["completion_id"] = completion_id;

            // Convert set to array
            json task_ids_array = json::array();
            for (const auto& id: task_ids) {
                task_ids_array.push_back(id);
            }
            response["task_ids"] = task_ids_array;

            SRV_INF("Started streaming OAI chat completion with %zu task(s)\n", task_ids.size());
        }

        // Return the response as a JSON string
        std::string response_str = response.dump();
        jstring result = env->NewStringUTF(response_str.c_str());

        return result;
    } catch (const std::exception& e) {
        SRV_ERR("Exception in handleChatCompletions: %s\n", e.what());
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

/**
 * Handle text infill request (completing text with given prefix and suffix).
 * Equivalent to POST /infill endpoint.
 */
JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleInfill(JNIEnv* env, jobject obj, jstring jrequestData, jboolean jstream) {
    try {
        // Get server context pointer from Java object
        jlong server_handle = env->GetLongField(obj, f_model_pointer);
        if (server_handle == 0) {
            env->ThrowNew(c_llama_error, "Model is not loaded");
            return nullptr;
        }

        auto* ctx_server = reinterpret_cast<server_context*>(server_handle);

        // Check if embeddings mode is active (which would prevent infill)
        if (ctx_server->params_base.embedding) {
            env->ThrowNew(c_llama_error, "This server does not support infill. Start it without `--embeddings`");
            return nullptr;
        }

        // Check model compatibility for infill
        std::string err;
        if (llama_vocab_fim_pre(ctx_server->vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server->vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server->vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            env->ThrowNew(c_llama_error, ("Infill is not supported by this model: " + err).c_str());
            return nullptr;
        }

        // Parse request data from JSON
        std::string request_str = parse_jstring(env, jrequestData);
        json data = json::parse(request_str);

        // Validate input
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            env->ThrowNew(c_llama_error, "\"prompt\" must be a string");
            return nullptr;
        }

        if (!data.contains("input_prefix")) {
            env->ThrowNew(c_llama_error, "\"input_prefix\" is required");
            return nullptr;
        }

        if (!data.contains("input_suffix")) {
            env->ThrowNew(c_llama_error, "\"input_suffix\" is required");
            return nullptr;
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            env->ThrowNew(c_llama_error, "\"input_extra\" must be an array of {\"filename\": string, \"text\": string}");
            return nullptr;
        }

        // Set streaming flag
        bool stream = jstream;
        data["stream"] = stream;

        // Process input_extra (context chunks)
        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto& chunk : input_extra) {
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                env->ThrowNew(c_llama_error, "extra_context chunk must contain a \"text\" field with a string value");
                return nullptr;
            }
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                env->ThrowNew(c_llama_error, "extra_context chunk's \"filename\" field must be a string");
                return nullptr;
            }
        }
        data["input_extra"] = input_extra;

        // Format the infill prompt
        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server->vocab, prompt, false, true);
        
        data["prompt"] = format_infill(
            ctx_server->vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            ctx_server->params_base.n_batch,
            ctx_server->params_base.n_predict,
            ctx_server->slots[0].n_ctx,
            ctx_server->params_base.spm_infill,
            tokenized_prompts.empty() ? std::vector<llama_token>() : tokenized_prompts[0]
        );

        // Create a completion ID
        auto completion_id = gen_chatcmplid();
        std::vector<server_task> tasks;

        try {
            // Process formatted prompt
            std::vector<llama_tokens> infill_prompts = tokenize_input_prompts(
                ctx_server->vocab, data.at("prompt"), true, true);

            tasks.reserve(infill_prompts.size());
            for (size_t i = 0; i < infill_prompts.size(); i++) {
                server_task task(SERVER_TASK_TYPE_INFILL);

                task.id = ctx_server->queue_tasks.get_new_id();
                task.index = i;

                task.prompt_tokens = std::move(infill_prompts[i]);
                task.params = server_task::params_from_json_cmpl(
                    ctx_server->ctx, ctx_server->params_base, data);

                task.id_selected_slot = json_value(data, "id_slot", -1);
                
                // Infill is not OAI compatible, but we still set the completion ID
                task.params.oaicompat = OAICOMPAT_TYPE_NONE;
                task.params.oaicompat_cmpl_id = completion_id;

                tasks.push_back(task);
            }
        } catch (const std::exception& e) {
            const auto& err = format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST);
            env->ThrowNew(c_llama_error, err.dump().c_str());
            return nullptr;
        }

        // Add tasks to waiting queue and post them for processing
        ctx_server->queue_results.add_waiting_tasks(tasks);
        ctx_server->queue_tasks.post(tasks);

        // Get task IDs
        const auto task_ids = server_task::get_list_id(tasks);

        // Create response JSON
        json response;

        if (!stream) {
            // For non-streaming, collect all results
            std::vector<server_task_result_ptr> results;
            results.reserve(tasks.size());

            for (size_t i = 0; i < tasks.size(); i++) {
                server_task_result_ptr result = ctx_server->queue_results.recv(task_ids);

                if (result->is_error()) {
                    // Clean up and throw error
                    ctx_server->queue_results.remove_waiting_task_ids(task_ids);
                    std::string error_msg = result->to_json()["message"].get<std::string>();
                    env->ThrowNew(c_llama_error, error_msg.c_str());
                    return nullptr;
                }

                results.push_back(std::move(result));
            }

            // Format the response
            response["type"] = "infill";
            response["streaming"] = false;
            response["completion_id"] = completion_id;

            if (results.size() == 1) {
                // Single result
                response["result"] = results[0]->to_json();
            } else {
                // Multiple results
                json results_array = json::array();
                for (auto& res : results) {
                    results_array.push_back(res->to_json());
                }
                response["results"] = results_array;
            }

            // Clean up
            ctx_server->queue_results.remove_waiting_task_ids(task_ids);
        } else {
            // For streaming, return the task IDs
            response["type"] = "infill_stream_init";
            response["streaming"] = true;
            response["completion_id"] = completion_id;

            // Convert set to array
            json task_ids_array = json::array();
            for (const auto& id : task_ids) {
                task_ids_array.push_back(id);
            }
            response["task_ids"] = task_ids_array;

            SRV_INF("Started streaming infill with %zu task(s)\n", task_ids.size());
        }

        // Return the response as a JSON string
        std::string response_str = response.dump();
        jstring result = env->NewStringUTF(response_str.c_str());

        return result;
    } catch (const std::exception& e) {
        SRV_ERR("Exception in handleInfill: %s\n", e.what());
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

/**
 * Get the next chunk of streaming results for a completion task.
 * Used to retrieve results during streaming.
 */

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getNextStreamResult(JNIEnv* env, jobject obj, jint taskId) {
    auto* ctx_server = static_cast<server_context*>(nullptr);
    try {
        // Get server context pointer from Java object
        jlong server_handle = env->GetLongField(obj, f_model_pointer);
        if (server_handle == 0) {
            env->ThrowNew(c_llama_error, "Model is not loaded");
            return nullptr;
        }

        ctx_server = reinterpret_cast<server_context*>(server_handle);

        // Get next result chunk from the result queue
        server_task_result_ptr result = ctx_server->queue_results.recv(taskId);

        if (result->is_error()) {
            // If there's an error, clean up and throw
            ctx_server->queue_results.remove_waiting_task_id(taskId);
            std::string error_msg = result->to_json()["message"].get<std::string>();
            env->ThrowNew(c_llama_error, error_msg.c_str());
            return nullptr;
        }
        
        // Try to parse the result JSON (check for UTF-8 validity)
        json resultJson;
        try {
            resultJson = result->to_json();
        } catch (const json::exception& e) {
            // If parsing fails, create a basic error response instead
            SRV_WRN("JSON parsing error: %s\n", e.what());
            resultJson = {
                {"content", "[Content contains invalid characters]"},
                {"error", e.what()}
            };
        }

        // Create response JSON with metadata
        json response = {
            {"type", "stream_chunk"},
            {"task_id", taskId},
            {"result", resultJson},
            {"is_final", result->is_stop()}
        };

        // If this is the final result, remove the task from the queue
        if (result->is_stop()) {
            ctx_server->queue_results.remove_waiting_task_id(taskId);
        }

        // Create JSON string with extra safety measures
        std::string response_str;
        try {
            response_str = response.dump();
            
            // Verify JSON is parseable (double-check)
            json::parse(response_str);
        } catch (const json::exception& e) {
            // If still failing, create a minimal valid JSON response
            SRV_ERR("Failed to create valid JSON response: %s\n", e.what());
            json fallback = {
                {"type", "stream_chunk"},
                {"task_id", taskId},
                {"result", {{"content", "[INVALID CONTENT]"}}},
                {"is_final", result->is_stop()},
                {"error", "Failed to generate valid JSON"}
            };
            response_str = fallback.dump();
        }
        
        // Check for invalid UTF-8 characters
        if (!is_valid_utf8(response_str)) {
            SRV_WRN("Response contains invalid UTF-8, sanitizing\n", "");
            response_str = sanitize_utf8(response_str);
        }
        
        // Create Java string
        jstring result_str = env->NewStringUTF(response_str.c_str());
        
        // Check if string creation succeeded
        if (result_str == nullptr) {
            // If NewStringUTF failed (due to invalid UTF-8), create a fallback response
            SRV_ERR("Failed to create Java string from response\n","");
            
            // Create a minimal ASCII-only response
            json ascii_fallback = {
                {"type", "stream_chunk"},
                {"task_id", taskId},
                {"result", {{"content", "[CONTENT CONTAINS INVALID CHARACTERS]"}}},
                {"is_final", result->is_stop()},
                {"error", "Invalid UTF-8 characters in response"}
            };
            
            // Use the ASCII-only fallback
            result_str = env->NewStringUTF(ascii_fallback.dump().c_str());
            
            // If still failing, something is very wrong
            if (result_str == nullptr) {
                env->ThrowNew(c_llama_error, "Critical error: Unable to create response string");
                return nullptr;
            }
        }
        
        return result_str;
    } catch (const std::exception& e) {
        SRV_ERR("Exception in getNextStreamResult: %s\n", e.what());
        env->ThrowNew(c_llama_error, e.what());
        if (ctx_server != nullptr) {
            ctx_server->queue_results.remove_waiting_task_id(taskId);
        }
        return nullptr;
    }
}

/**
 * Release resources associated with a task.
 * Used to clean up after a task is complete.
 */
JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_releaseTask(JNIEnv* env, jobject obj, jint taskId) {
    try {
        // Get server context pointer from Java object
        jlong server_handle = env->GetLongField(obj, f_model_pointer);
        if (server_handle == 0) {
            env->ThrowNew(c_llama_error, "Model is not loaded");
            return;
        }

        auto* ctx_server = reinterpret_cast<server_context*>(server_handle);

        // Remove the task from the waiting tasks queue
        ctx_server->queue_results.remove_waiting_task_id(taskId);
        
        SRV_INF("Task %d released\n", taskId);
    } catch (const std::exception& e) {
        SRV_ERR("Exception in releaseTask: %s\n", e.what());
        env->ThrowNew(c_llama_error, e.what());
    }
}

/**
 * Cancel an ongoing completion.
 * Stops generation and cleans up resources.
 */
JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_cancelCompletion(JNIEnv* env, jobject obj, jint taskId) {
    try {
        // Get server context pointer from Java object
        jlong server_handle = env->GetLongField(obj, f_model_pointer);
        if (server_handle == 0) {
            env->ThrowNew(c_llama_error, "Model is not loaded");
            return;
        }

        auto* ctx_server = reinterpret_cast<server_context*>(server_handle);

        // Create a set with the task ID
        std::unordered_set<int> task_ids = {taskId};
        
        // Cancel the tasks in the server context
        ctx_server->cancel_tasks(task_ids);
        
        // Remove the task from the waiting tasks queue
        ctx_server->queue_results.remove_waiting_task_id(taskId);
        
        SRV_INF("Task %d canceled\n", taskId);
    } catch (const std::exception& e) {
        SRV_ERR("Exception in cancelCompletion: %s\n", e.what());
        env->ThrowNew(c_llama_error, e.what());
    }
}


/**
 * Handle embeddings request.
 * Equivalent to POST /embeddings endpoint.
 */
JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleEmbeddings(JNIEnv* env, jobject obj, jstring jrequestData, jboolean joaiCompat) {
    try {
        // Get server context pointer from Java object
        jlong server_handle = env->GetLongField(obj, f_model_pointer);
        if (server_handle == 0) {
            env->ThrowNew(c_llama_error, "Model is not loaded");
            return nullptr;
        }

        auto* ctx_server = reinterpret_cast<server_context*>(server_handle);
        
        // Check if embeddings mode is enabled
        if (!ctx_server->params_base.embedding) {
            env->ThrowNew(c_llama_error, "Model was not loaded with embedding support (see ModelParameters#setEmbedding(boolean))");
            return nullptr;
        }

        // Set compatibility mode
        oaicompat_type oaicompat = joaiCompat ? OAICOMPAT_TYPE_EMBEDDING : OAICOMPAT_TYPE_NONE;
        
        // Check if pooling type is compatible with OAI mode
        if (oaicompat != OAICOMPAT_TYPE_NONE && llama_pooling_type(ctx_server->ctx) == LLAMA_POOLING_TYPE_NONE) {
            env->ThrowNew(c_llama_error, "Pooling type 'none' is not OAI compatible. Please use a different pooling type");
            return nullptr;
        }
        
        // Parse request data from JSON
        std::string request_str = parse_jstring(env, jrequestData);
        json body = json::parse(request_str);
        
        // Check for input field
        json prompt;
        if (body.count("input") != 0) {
            prompt = body.at("input");
        } else if (body.contains("content")) {
            // "content" field is not OAI compatible
            oaicompat = OAICOMPAT_TYPE_NONE;
            prompt = body.at("content");
        } else {
            env->ThrowNew(c_llama_error, "\"input\" or \"content\" must be provided");
            return nullptr;
        }
        
        // Check encoding format
        bool use_base64 = false;
        if (body.count("encoding_format") != 0) {
            const std::string& format = body.at("encoding_format");
            if (format == "base64") {
                use_base64 = true;
            } else if (format != "float") {
                env->ThrowNew(c_llama_error, "The format to return the embeddings in. Can be either float or base64");
                return nullptr;
            }
        }
        
        // Tokenize the prompts
        std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server->vocab, prompt, true, true);
        
        // Check for empty input
        for (const auto& tokens : tokenized_prompts) {
            if (tokens.empty()) {
                env->ThrowNew(c_llama_error, "Input content cannot be empty");
                return nullptr;
            }
        }
        
        // Create embedding tasks
        json responses = json::array();
        std::vector<server_task> tasks;
        tasks.reserve(tokenized_prompts.size());
        
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id = ctx_server->queue_tasks.get_new_id();
            task.index = i;
            task.prompt_tokens = std::move(tokenized_prompts[i]);
            task.params.oaicompat = oaicompat;

            tasks.push_back(task);
        }
        
        // Submit tasks for processing
        ctx_server->queue_results.add_waiting_tasks(tasks);
        ctx_server->queue_tasks.post(tasks);
        
        // Get task IDs
        std::unordered_set<int> task_ids = server_task::get_list_id(tasks);
        
        // Get task results
        for (size_t i = 0; i < tasks.size(); i++) {
            server_task_result_ptr result = ctx_server->queue_results.recv(task_ids);
            
            if (result->is_error()) {
                ctx_server->queue_results.remove_waiting_task_ids(task_ids);
                std::string error_msg = result->to_json()["message"].get<std::string>();
                env->ThrowNew(c_llama_error, error_msg.c_str());
                return nullptr;
            }
            
            responses.push_back(result->to_json());
        }
        
        // Clean up
        ctx_server->queue_results.remove_waiting_task_ids(task_ids);
        
        // Format response based on compatibility mode
        json root = oaicompat == OAICOMPAT_TYPE_EMBEDDING
            ? format_embeddings_response_oaicompat(body, responses, use_base64)
            : json(responses);
        
        // Return the response as a JSON string
        std::string response_str = root.dump(2);
        jstring result = env->NewStringUTF(response_str.c_str());
        
        return result;
    } catch (const std::exception& e) {
        SRV_ERR("Exception in handleEmbeddings: %s\n", e.what());
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

/**
 * Handle reranking request.
 * Equivalent to POST /rerank endpoint.
 */
JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleRerank(JNIEnv* env, jobject obj, jstring jrequestData) {
    try {
        // Get server context pointer from Java object
        jlong server_handle = env->GetLongField(obj, f_model_pointer);
        if (server_handle == 0) {
            env->ThrowNew(c_llama_error, "Model is not loaded");
            return nullptr;
        }

        auto* ctx_server = reinterpret_cast<server_context*>(server_handle);
        
        // Check if reranking mode is enabled and embedding mode is disabled
        if (!ctx_server->params_base.reranking || ctx_server->params_base.embedding) {
            env->ThrowNew(c_llama_error, 
                "This server does not support reranking. Start it with `--reranking` and without `--embedding`");
            return nullptr;
        }
        
        // Parse request data from JSON
        std::string request_str = parse_jstring(env, jrequestData);
        json body = json::parse(request_str);
        
        // Check if using TEI or Jina API format
        bool is_tei_format = body.contains("texts");
        
        // Validate and get query
        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                env->ThrowNew(c_llama_error, "\"query\" must be a string");
                return nullptr;
            }
        } else {
            env->ThrowNew(c_llama_error, "\"query\" must be provided");
            return nullptr;
        }
        
        // Get documents/texts
        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            env->ThrowNew(c_llama_error, "\"documents\" must be a non-empty string array");
            return nullptr;
        }
        
        // Tokenize query
        llama_tokens tokenized_query = tokenize_input_prompts(ctx_server->vocab, query, false, true)[0];
        
        // Create rerank tasks
        json responses = json::array();
        std::vector<server_task> tasks;
        std::vector<llama_tokens> tokenized_docs = tokenize_input_prompts(ctx_server->vocab, documents, false, true);
        
        tasks.reserve(tokenized_docs.size());
        for (size_t i = 0; i < tokenized_docs.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_RERANK);
            task.id = ctx_server->queue_tasks.get_new_id();
            task.index = i;
            task.prompt_tokens = format_rerank(ctx_server->vocab, tokenized_query, tokenized_docs[i]);
            tasks.push_back(task);
        }
        
        // Submit tasks for processing
        ctx_server->queue_results.add_waiting_tasks(tasks);
        ctx_server->queue_tasks.post(tasks);
        
        // Get task IDs
        std::unordered_set<int> task_ids = server_task::get_list_id(tasks);
        
        // Get task results
        for (size_t i = 0; i < tasks.size(); i++) {
            server_task_result_ptr result = ctx_server->queue_results.recv(task_ids);
            
            if (result->is_error()) {
                ctx_server->queue_results.remove_waiting_task_ids(task_ids);
                std::string error_msg = result->to_json()["message"].get<std::string>();
                env->ThrowNew(c_llama_error, error_msg.c_str());
                return nullptr;
            }
            
            responses.push_back(result->to_json());
        }
        
        // Clean up
        ctx_server->queue_results.remove_waiting_task_ids(task_ids);
        
        // Format the rerank response
        json root = format_response_rerank(
            body,
            responses,
            is_tei_format,
            documents);
        
        // Return the response as a JSON string
        std::string response_str = root.dump(2);
        jstring result = env->NewStringUTF(response_str.c_str());
        
        return result;
    } catch (const std::exception& e) {
        SRV_ERR("Exception in handleRerank: %s\n", e.what());
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}


/**
 * Handle tokenization request.
 * Equivalent to POST /tokenize endpoint.
 */
JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleTokenize(JNIEnv* env, jobject obj, jstring jcontent, jboolean jaddSpecial, jboolean jwithPieces) {
    try {
        // Get server context pointer from Java object
        jlong server_handle = env->GetLongField(obj, f_model_pointer);
        if (server_handle == 0) {
            env->ThrowNew(c_llama_error, "Model is not loaded");
            return nullptr;
        }

        auto* ctx_server = reinterpret_cast<server_context*>(server_handle);
        
        // Parse parameters
        const std::string content = parse_jstring(env, jcontent);
        const bool add_special = jaddSpecial;
        const bool with_pieces = jwithPieces;
        
        // Tokenize the content
        llama_tokens tokens = tokenize_mixed(ctx_server->vocab, content, add_special, true);
        
        // Create response JSON
        json tokens_response = json::array();
        
        if (with_pieces) {
            // If detailed token info is requested, include token pieces
            for (const auto& token : tokens) {
                std::string piece = common_token_to_piece(ctx_server->ctx, token);
                json piece_json;
                
                // Check if the piece is valid UTF-8
                if (is_valid_utf8(piece)) {
                    piece_json = piece;
                } else {
                    // If not valid UTF-8, store as array of byte values
                    piece_json = json::array();
                    for (unsigned char c : piece) {
                        piece_json.push_back(static_cast<int>(c));
                    }
                }
                
                tokens_response.push_back({
                    {"id", token},
                    {"piece", piece_json}
                });
            }
        } else {
            // Otherwise just include token IDs
            tokens_response = tokens;
        }
        
        // Format the response
        json data = format_tokenizer_response(tokens_response);
        
        // Return as JSON string
        std::string response_str = data.dump(2);
        jstring result = env->NewStringUTF(response_str.c_str());
        
        return result;
    } catch (const std::exception& e) {
        SRV_ERR("Exception in handleTokenize: %s\n", e.what());
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

/**
 * Handle detokenization request.
 * Equivalent to POST /detokenize endpoint.
 */
JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleDetokenize(JNIEnv* env, jobject obj, jintArray jtokens) {
    try {
        // Get server context pointer from Java object
        jlong server_handle = env->GetLongField(obj, f_model_pointer);
        if (server_handle == 0) {
            env->ThrowNew(c_llama_error, "Model is not loaded");
            return nullptr;
        }

        auto* ctx_server = reinterpret_cast<server_context*>(server_handle);
        
        // Convert Java tokens to C++ vector
        jsize length = env->GetArrayLength(jtokens);
        jint* elements = env->GetIntArrayElements(jtokens, nullptr);
        std::vector<llama_token> tokens(elements, elements + length);
        
        // Convert tokens to string
        std::string content = tokens_to_str(ctx_server->ctx, tokens.cbegin(), tokens.cend());
        
        // Release Java array elements
        env->ReleaseIntArrayElements(jtokens, elements, JNI_ABORT);
        
        // Format the response
        json data = format_detokenized_response(content);
        
        // Return as JSON string
        std::string response_str = data.dump(2);
        jstring result = env->NewStringUTF(response_str.c_str());
        
        return result;
    } catch (const std::exception& e) {
        SRV_ERR("Exception in handleDetokenize: %s\n", e.what());
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

/**
 * Apply chat template to messages.
 * Equivalent to POST /apply-template endpoint.
 */
JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_applyTemplate(JNIEnv* env, jobject obj, jstring jrequestData) {
    try {
        // Get server context pointer from Java object
        jlong server_handle = env->GetLongField(obj, f_model_pointer);
        if (server_handle == 0) {
            env->ThrowNew(c_llama_error, "Model is not loaded");
            return nullptr;
        }

        auto* ctx_server = reinterpret_cast<server_context*>(server_handle);
        
        // Parse request data
        std::string request_str = parse_jstring(env, jrequestData);
        json body = json::parse(request_str);
        
        // Apply the template using the OpenAI parameter parsing function
        // This function processes the messages using the model's chat template
        json templateData = oaicompat_completion_params_parse(
            body,
            ctx_server->params_base.use_jinja,
            ctx_server->params_base.reasoning_format,
            ctx_server->chat_templates.get()
        );
        
        // Extract the formatted prompt
        std::string formatted_prompt = templateData.at("prompt");
        
        // Create response JSON
        json response = {
            {"prompt", formatted_prompt}
        };
        
        // Return as JSON string
        std::string response_str = response.dump(2);
        jstring result = env->NewStringUTF(response_str.c_str());
        
        return result;
    } catch (const std::exception& e) {
        SRV_ERR("Exception in applyTemplate: %s\n", e.what());
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

/**
 * Handle slot management operations.
 * Consolidates GET /slots and POST /slots/:id_slot endpoints.
 * 
 * @param env JNI environment
 * @param obj Java object
 * @param action Action to perform: 0=GET (list), 1=SAVE, 2=RESTORE, 3=ERASE
 * @param slotId Slot ID (ignored for GET action)
 * @param jfilename Filename for save/restore (ignored for GET and ERASE actions)
 * @return JSON string for GET action, true/false for other actions
 */
JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleSlotAction(JNIEnv* env, jobject obj, jint action, jint slotId, jstring jfilename) {
    try {
        // Get server context pointer from Java object
        jlong server_handle = env->GetLongField(obj, f_model_pointer);
        if (server_handle == 0) {
            env->ThrowNew(c_llama_error, "Model is not loaded");
            return nullptr;
        }

        auto* ctx_server = reinterpret_cast<server_context*>(server_handle);
        
        // Process based on action type
        switch (action) {
            case 0: { // GET - List slots
                // Check if slots endpoint is enabled
                if (!ctx_server->params_base.endpoint_slots) {
                    env->ThrowNew(c_llama_error, "This server does not support slots endpoint. Start it with `--slots`");
                    return nullptr;
                }
                
                // Request slots data using task queue
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = ctx_server->queue_tasks.get_new_id();
                ctx_server->queue_results.add_waiting_task_id(task.id);
                ctx_server->queue_tasks.post(task, true); // high-priority task

                // Get the result
                server_task_result_ptr result = ctx_server->queue_results.recv(task.id);
                ctx_server->queue_results.remove_waiting_task_id(task.id);

                if (result->is_error()) {
                    std::string error_msg = result->to_json()["message"].get<std::string>();
                    env->ThrowNew(c_llama_error, error_msg.c_str());
                    return nullptr;
                }
                
                // Parse metrics result
                auto res_metrics = dynamic_cast<server_task_result_metrics*>(result.get());
                if (res_metrics == nullptr) {
                    env->ThrowNew(c_llama_error, "Invalid metrics result");
                    return nullptr;
                }
                
                // Create JSON response with slots data
                json response = {
                    {"slots", res_metrics->slots_data},
                    {"n_idle_slots", res_metrics->n_idle_slots},
                    {"success", true}
                };
                
                // Return as JSON string
                std::string response_str = response.dump(2);
                return env->NewStringUTF(response_str.c_str());
            }
            
            case 1: { // SAVE - Save slot state
                // Check if slot save is enabled
                if (ctx_server->params_base.slot_save_path.empty()) {
                    env->ThrowNew(c_llama_error, "This server does not support slot save. Start it with `--slot-save-path`");
                    return nullptr;
                }
                
                // Get the filename
                std::string filename = parse_jstring(env, jfilename);
                if (!fs_validate_filename(filename)) {
                    env->ThrowNew(c_llama_error, "Invalid filename");
                    return nullptr;
                }
                
                std::string filepath = ctx_server->params_base.slot_save_path + filename;
                
                // Create the save task
                server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
                task.id = ctx_server->queue_tasks.get_new_id();
                task.slot_action.slot_id = slotId;
                task.slot_action.filename = filename;
                task.slot_action.filepath = filepath;

                ctx_server->queue_results.add_waiting_task_id(task.id);
                ctx_server->queue_tasks.post(task);

                server_task_result_ptr result = ctx_server->queue_results.recv(task.id);
                ctx_server->queue_results.remove_waiting_task_id(task.id);

                if (result->is_error()) {
                    std::string error_msg = result->to_json()["message"].get<std::string>();
                    env->ThrowNew(c_llama_error, error_msg.c_str());
                    return nullptr;
                }
                
                // Create JSON response indicating success
                json response = {
                    {"action", "save"},
                    {"slot_id", slotId},
                    {"filename", filename},
                    {"success", true}
                };
                
                SRV_INF("Slot %d saved to file %s\n", slotId, filename.c_str());
                
                // Return as JSON string
                std::string response_str = response.dump(2);
                return env->NewStringUTF(response_str.c_str());
            }
            
            case 2: { // RESTORE - Restore slot state
                // Check if slot save is enabled
                if (ctx_server->params_base.slot_save_path.empty()) {
                    env->ThrowNew(c_llama_error, "This server does not support slot restore. Start it with `--slot-save-path`");
                    return nullptr;
                }
                
                // Get the filename
                std::string filename = parse_jstring(env, jfilename);
                if (!fs_validate_filename(filename)) {
                    env->ThrowNew(c_llama_error, "Invalid filename");
                    return nullptr;
                }
                
                std::string filepath = ctx_server->params_base.slot_save_path + filename;
                
                // Create the restore task
                server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
                task.id = ctx_server->queue_tasks.get_new_id();
                task.slot_action.slot_id = slotId;
                task.slot_action.filename = filename;
                task.slot_action.filepath = filepath;

                ctx_server->queue_results.add_waiting_task_id(task.id);
                ctx_server->queue_tasks.post(task);

                server_task_result_ptr result = ctx_server->queue_results.recv(task.id);
                ctx_server->queue_results.remove_waiting_task_id(task.id);

                if (result->is_error()) {
                    std::string error_msg = result->to_json()["message"].get<std::string>();
                    env->ThrowNew(c_llama_error, error_msg.c_str());
                    return nullptr;
                }
                
                // Create JSON response indicating success
                json response = {
                    {"action", "restore"},
                    {"slot_id", slotId},
                    {"filename", filename},
                    {"success", true}
                };
                
                SRV_INF("Slot %d restored from file %s\n", slotId, filename.c_str());
                
                // Return as JSON string
                std::string response_str = response.dump(2);
                return env->NewStringUTF(response_str.c_str());
            }
            
            case 3: { // ERASE - Erase slot state
                // Create the erase task
                server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
                task.id = ctx_server->queue_tasks.get_new_id();
                task.slot_action.slot_id = slotId;

                ctx_server->queue_results.add_waiting_task_id(task.id);
                ctx_server->queue_tasks.post(task);

                server_task_result_ptr result = ctx_server->queue_results.recv(task.id);
                ctx_server->queue_results.remove_waiting_task_id(task.id);

                if (result->is_error()) {
                    std::string error_msg = result->to_json()["message"].get<std::string>();
                    env->ThrowNew(c_llama_error, error_msg.c_str());
                    return nullptr;
                }
                
                // Create JSON response indicating success
                json response = {
                    {"action", "erase"},
                    {"slot_id", slotId},
                    {"success", true}
                };
                
                SRV_INF("Slot %d erased\n", slotId);
                
                // Return as JSON string
                std::string response_str = response.dump(2);
                return env->NewStringUTF(response_str.c_str());
            }
            
            default:
                env->ThrowNew(c_llama_error, "Invalid slot action");
                return nullptr;
        }
    } catch (const std::exception& e) {
        SRV_ERR("Exception in handleSlotAction: %s\n", e.what());
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

/**
 * Convert a JSON schema to a grammar.
 * Utility method for generating grammar rules from JSON schema definitions.
 */
JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_jsonSchemaToGrammarBytes(JNIEnv* env, jclass clazz, jstring j_schema) {
    try {
        // Parse the JSON schema string
        const std::string c_schema = parse_jstring(env, j_schema);
        
        // Parse the schema as ordered JSON (to maintain property order)
        nlohmann::ordered_json c_schema_json;
        try {
            c_schema_json = nlohmann::ordered_json::parse(c_schema);
        } catch (const nlohmann::json::exception& e) {
            env->ThrowNew(c_llama_error, ("Failed to parse JSON schema: " + std::string(e.what())).c_str());
            return nullptr;
        }
        
        // Convert JSON schema to grammar
        std::string c_grammar;
        try {
            c_grammar = json_schema_to_grammar(c_schema_json);
        } catch (const std::exception& e) {
            env->ThrowNew(c_llama_error, ("Failed to convert schema to grammar: " + std::string(e.what())).c_str());
            return nullptr;
        }
        
        // Convert the grammar string to a byte array
        jbyteArray result = parse_jbytes(env, c_grammar);
        
        SRV_INF("JSON schema converted to grammar (%zu bytes)\n", c_grammar.size());
        return result;
    } catch (const std::exception& e) {
        SRV_ERR("Exception in jsonSchemaToGrammarBytes: %s\n", e.what());
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

JNIEXPORT jintArray JNICALL Java_de_kherud_llama_LlamaModel_encode(JNIEnv * env, jobject obj, jstring jprompt) {
  jlong server_handle = env -> GetLongField(obj, f_model_pointer);
  auto * ctx_server = reinterpret_cast < server_context * > (server_handle); // NOLINT(*-no-int-to-ptr)

  const std::string c_prompt = parse_jstring(env, jprompt);

  llama_tokens tokens = tokenize_mixed(ctx_server -> vocab, c_prompt, false, true);
  jsize token_size = tokens.size(); // NOLINT(*-narrowing-conversions)

  jintArray java_tokens = env -> NewIntArray(token_size);
  if (java_tokens == nullptr) {
    env -> ThrowNew(c_error_oom, "could not allocate token memory");
    return nullptr;
  }

  env -> SetIntArrayRegion(java_tokens, 0, token_size, reinterpret_cast <
    const jint * > (tokens.data()));

  return java_tokens;
}