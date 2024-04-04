#include "jllama.h"

#include "common.h"
#include "json.hpp"

using json = nlohmann::json;

JavaVM *g_vm = nullptr;
jobject g_log_callback = nullptr;

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
static jclass c_split_mode = 0;
static jclass c_log_format = 0;
static jclass c_miro_stat = 0;
static jclass c_numa_strategy = 0;
static jclass c_pooling_type = 0;
static jclass c_rope_scaling = 0;
static jclass c_sampler = 0;

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
static jfieldID f_cache_prompt = 0;
static jfieldID f_n_predict = 0;
static jfieldID f_top_k = 0;
static jfieldID f_top_p = 0;
static jfieldID f_min_p = 0;
static jfieldID f_tfs_z = 0;
static jfieldID f_typical_p = 0;
static jfieldID f_temp = 0;
static jfieldID f_dynatemp_range = 0;
static jfieldID f_dynatemp_exponent = 0;
static jfieldID f_penalty_last_n = 0;
static jfieldID f_penalty_repeat = 0;
static jfieldID f_penalty_freq = 0;
static jfieldID f_penalty_present = 0;
static jfieldID f_mirostat = 0;
static jfieldID f_mirostat_tau = 0;
static jfieldID f_mirostat_eta = 0;
static jfieldID f_penalize_nl = 0;
static jfieldID f_n_keep = 0;
static jfieldID f_n_discard = 0;
static jfieldID f_infer_seed = 0;
static jfieldID f_n_probs = 0;
static jfieldID f_min_keep = 0;
static jfieldID f_grammar = 0;
static jfieldID f_ignore_eos = 0;
static jfieldID f_logit_bias = 0;
static jfieldID f_antiprompt = 0;
// model parameters
static jfieldID f_model_seed = 0;
static jfieldID f_model_path = 0;
static jfieldID f_model_url = 0;
static jfieldID f_model_hf_repo = 0;
static jfieldID f_model_hf_file = 0;
static jfieldID f_model_alias = 0;
static jfieldID f_n_ctx = 0;
static jfieldID f_rope_scaling_type = 0;
static jfieldID f_rope_freq_base = 0;
static jfieldID f_rope_freq_scale = 0;
static jfieldID f_yarn_ext_factor = 0;
static jfieldID f_yarn_attn_factor = 0;
static jfieldID f_yarn_beta_fast = 0;
static jfieldID f_yarn_beta_slow = 0;
static jfieldID f_pooling_type = 0;
static jfieldID f_defrag_thold = 0;
static jfieldID f_n_threads = 0;
static jfieldID f_grp_attn_n = 0;
static jfieldID f_grp_attn_w = 0;
static jfieldID f_n_threads_batch = 0;
static jfieldID f_n_batch = 0;
static jfieldID f_n_ubatch = 0;
static jfieldID f_n_gpu_layers = 0;
static jfieldID f_no_kv_offload = 0;
static jfieldID f_split_mode = 0;
static jfieldID f_tensor_split = 0;
static jfieldID f_main_gpu = 0;
static jfieldID f_verbose = 0;
static jfieldID f_use_mlock = 0;
static jfieldID f_use_mmap = 0;
static jfieldID f_numa_strategy = 0;
static jfieldID f_embedding = 0;
static jfieldID f_cont_batching = 0;
static jfieldID f_n_parallel = 0;
static jfieldID f_n_predict = 0;
static jfieldID f_system_prompt_file = 0;
static jfieldID f_log_format = 0;
// enum fields
static jfieldID f_utf_8 = 0;
static jfieldID f_log_level_debug = 0;
static jfieldID f_log_level_info = 0;
static jfieldID f_log_level_warn = 0;
static jfieldID f_log_level_error = 0;
static jfieldID f_rope_scaling_none = 0;
static jfieldID f_rope_scaling_linear = 0;
static jfieldID f_rope_scaling_yarn = 0;
static jfieldID f_pooling_type_none = 0;
static jfieldID f_pooling_type_mean = 0;
static jfieldID f_pooling_type_cls = 0;
static jfieldID f_split_mode_none = 0;
static jfieldID f_split_mode_layer = 0;
static jfieldID f_split_mode_row = 0;
static jfieldID f_numa_strategy_distribute = 0;
static jfieldID f_numa_strategy_isolate = 0;
static jfieldID f_numa_strategy_numactl = 0;
static jfieldID f_log_format_json = 0;
static jfieldID f_log_format_text = 0;
static jfieldID f_mirostat_v1 = 0;
static jfieldID f_mirostat_v2 = 0;
// objects
static jobject o_utf_8 = 0;
static jobject o_log_level_debug = 0;
static jobject o_log_level_info = 0;
static jobject o_log_level_warn = 0;
static jobject o_log_level_error = 0;
static jobject o_rope_scaling_none = 0;
static jobject o_rope_scaling_linear = 0;
static jobject o_rope_scaling_yarn = 0;
static jobject o_pooling_type_none = 0;
static jobject o_pooling_type_mean = 0;
static jobject o_pooling_type_cls = 0;
static jobject o_split_mode_none = 0;
static jobject o_split_mode_layer = 0;
static jobject o_split_mode_row = 0;
static jobject o_numa_strategy_distribute = 0;
static jobject o_numa_strategy_isolate = 0;
static jobject o_numa_strategy_numactl = 0;
static jobject o_log_format_json = 0;
static jobject o_log_format_text = 0;
static jobject o_mirostat_v1 = 0;
static jobject o_mirostat_v2 = 0;

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

// Since Java expects utf16 but std::strings are utf8, we can't directly use
// `env->NewString` or `env-NewString`, but we simply send the bytes directly
// and do the conversion in Java. Unfortunately, there isn't a
// nice/standardized way to do this conversion in C++
static jbyteArray parse_jbytes(JNIEnv *env, std::string string)
{
    jsize len = string.size();
    jbyteArray bytes = env->NewByteArray(len);
    env->SetByteArrayRegion(bytes, 0, len, reinterpret_cast<const jbyte *>(string.c_str()));
    return bytes;
}

// this method
static void load_server_params(JNIEnv *env, jobject jparams, server_params &sparams, gpt_params &params)
{
    gpt_params default_params;
    server_params default_sparams;

    bool invalid_param = false;

    params.seed = env->GetIntField(jparams, f_model_seed);
    params.model = get_string_field(env, jparams, f_model_path);
    params.model_url = get_string_field(env, jparams, f_model_url);
    params.hf_repo = get_string_field(env, jparams, f_model_hf_repo);
    params.hf_file = get_string_field(env, jparams, f_model_hf_file);
    params.model_alias = get_string_field(env, jparams, f_model_alias);
    params.n_ctx = env->GetIntField(jparams, f_n_ctx);

    jobject value = env->GetObjectField(jparams, f_rope_scaling_type);
    if (value == o_rope_scaling_none)
    {
        params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
    }
    else if (value == o_rope_scaling_linear)
    {
        params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
    }
    else if (value == o_rope_scaling_yarn)
    {
        params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
    }

    params.rope_freq_base = env->GetFloatField(jparams, f_rope_freq_base);
    params.rope_freq_scale = env->GetFloatField(jparams, f_rope_freq_scale);
    params.yarn_ext_factor = env->GetFloatField(jparams, f_yarn_ext_factor);
    params.yarn_attn_factor = env->GetFloatField(jparams, f_yarn_attn_factor);
    params.yarn_beta_fast = env->GetFloatField(jparams, f_yarn_beta_fast);
    params.yarn_beta_slow = env->GetFloatField(jparams, f_yarn_beta_slow);

    value = env->GetObjectField(jparams, f_pooling_type);
    if (value == o_pooling_type_none)
    {
        params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    }
    else if (value == o_pooling_type_mean)
    {
        params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    }
    else if (value == o_pooling_type_cls)
    {
        params.pooling_type = LLAMA_POOLING_TYPE_CLS;
    }

    params.defrag_thold = env->GetFloatField(jparams, f_defrag_thold);
    params.n_threads = env->GetIntField(jparams, f_n_threads);
    params.grp_attn_n = env->GetIntField(jparams, f_grp_attn_n);
    params.grp_attn_w = env->GetIntField(jparams, f_grp_attn_w);
    params.n_threads_batch = env->GetIntField(jparams, f_n_threads_batch);
    params.n_batch = env->GetIntField(jparams, f_n_batch);
    params.n_ubatch = env->GetIntField(jparams, f_n_ubatch);

    if (llama_supports_gpu_offload())
    {
        params.n_gpu_layers = env->GetIntField(jparams, f_n_gpu_layers);
    }
    else
    {
        LOG_WARNING("Not compiled with GPU offload support, --n-gpu-layers option will be ignored. "
                    "See main README.md for information on enabling GPU BLAS support",
                    {{"n_gpu_layers", params.n_gpu_layers}});
    }

    params.no_kv_offload = env->GetBooleanField(jparams, f_no_kv_offload);

    value = env->GetObjectField(jparams, f_split_mode);
    if (value == o_split_mode_none)
    {
        params.split_mode = LLAMA_SPLIT_MODE_NONE;
    }
    else if (value == o_split_mode_layer)
    {
        params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    }
    else if (value == o_split_mode_row)
    {
        params.split_mode = LLAMA_SPLIT_MODE_ROW;
    }

#ifndef GGML_USE_CUDA
    if (value != o_split_mode_none)
    {
        fprintf(stderr, "warning: llama.cpp was compiled without CUDA. Setting the split mode has no effect.\n");
    }
#endif

    jintArray j_tensor_split = env->GetObjectField(jparams, f_tensor_split);
    jsize j_tensor_split_size = env->GetArrayLength(j_tensor_split);
    jfloat *j_tensor_split_elements = env->GetFloatArrayElements(j_tensor_split, 0);

#if defined(GGML_USE_CUDA) || defined(GGML_USE_SYCL)
    GGML_ASSERT(j_tensor_split_size <= llama_max_devices());

    for (size_t i_device = 0; i_device < llama_max_devices(); ++i_device)
    {
        if (i_device < j_tensor_split_size)
        {
            params.tensor_split[i_device] = j_tensor_split_elements[i_device];
        }
        else
        {
            params.tensor_split[i_device] = 0.0f;
        }
    }
#else
    LOG_WARNING("llama.cpp was compiled without CUDA. It is not possible to set a tensor split.\n", {});
#endif

    params.main_gpu = env->GetIntField(jparams, f_main_gpu);
#if defined(GGML_USE_CUDA) || defined(GGML_USE_SYCL)
#else
    LOG_WARNING("llama.cpp was compiled without CUDA. It is not possible to set a main GPU.", {});
#endif

    //	// todo: there can be multiple lora adapters
    //	value = env->GetObjectField(jparams, f_lora_adapter);
    //	if (value != nullptr) {
    //		auto adapter = parse_jstring(env, (jstring) value);
    //		params.lora_adapter.emplace_back(adapter, 1.0f);
    //		params.use_mmap = false;
    //	}

    //	else if (arg == "--lora-scaled") {
    //		if (++i >= argc) {
    //			invalid_param = true;
    //			break;
    //		}
    //		const char * lora_adapter = argv[i];
    //		if (++i >= argc) {
    //			invalid_param = true;
    //			break;
    //		}
    //		params.lora_adapter.emplace_back(lora_adapter, std::stof(argv[i]));
    //		params.use_mmap = false;
    //	}

    //	params.lora_base = get_string_field(env, jparams, f_lora_base);

    sparams.verbose = env->GetBooleanField(jparams, f_verbose);
#if SERVER_VERBOSE != 1
    if (sparams.verbose)
    {
        LOG_WARNING("server.cpp is not built with verbose logging.", {});
    }
#else
    server_verbose = true;
#endif

    params.use_mlock = env->GetBooleanField(jparams, f_use_mlock);
    params.use_mmap = env->GetBooleanField(jparams, f_use_mmap);

    value = env->GetObjectField(jparams, f_numa_strategy);
    if (value == o_numa_strategy_distribute)
    {
        params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
    }
    else if (value == o_numa_strategy_isolate)
    {
        params.numa = GGML_NUMA_STRATEGY_ISOLATE;
    }
    else if (value == o_numa_strategy_numactl)
    {
        params.numa = GGML_NUMA_STRATEGY_NUMACTL;
    }

    params.embedding = env->GetBooleanField(jparams, f_embedding);
    params.cont_batching = env->GetBooleanField(jparams, f_cont_batching);
    params.n_parallel = env->GetIntField(jparams, f_n_parallel);
    params.n_predict = env->GetIntField(jparams, f_n_predict);

    auto system_prompt_file = get_string_field(env, jparams, f_system_prompt_file);
    if (system_prompt_file.length() > 0)
    {
        std::ifstream file(system_prompt_file);
        if (!file)
        {
            fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
            invalid_param = true;
            break;
        }
        std::string system_prompt;
        std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(),
                  std::back_inserter(system_prompt));
        sparams.system_prompt = system_prompt;
    }

    value = env->GetObjectField(jparams, f_log_format);
    if (value == o_log_format_json)
    {
        server_log_json = true;
    }
    else if (value == o_log_format_text)
    {
        server_log_json = false;
    }
    else
    {
        log_set_target(stdout);
        LOG_INFO("logging to file is disabled.", {});
    }

    //	auto system_prompt_file = get_string_field(env, jparams, f_system_prompt_file);
    //
    //	else if (arg == "--chat-template") {
    //            if (++i >= argc) {
    //                invalid_param = true;
    //                break;
    //            }
    //            if (!verify_custom_template(argv[i])) {
    //                fprintf(stderr, "error: the supplied chat template is not supported: %s\n", argv[i]);
    //                fprintf(stderr, "note: llama.cpp does not use jinja parser, we only support commonly used
    //                templates\n"); invalid_param = true; break;
    //            }
    //            sparams.chat_template = argv[i];
    //        } else if (arg == "--override-kv") {
    //            if (++i >= argc) {
    //                invalid_param = true;
    //                break;
    //            }
    //            char * sep = strchr(argv[i], '=');
    //            if (sep == nullptr || sep - argv[i] >= 128) {
    //                fprintf(stderr, "error: Malformed KV override: %s\n", argv[i]);
    //                invalid_param = true;
    //                break;
    //            }
    //
    //            struct llama_model_kv_override kvo;
    //            std::strncpy(kvo.key, argv[i], sep - argv[i]);
    //            kvo.key[sep - argv[i]] = 0;
    //            sep++;
    //            if (strncmp(sep, "int:", 4) == 0) {
    //                sep += 4;
    //                kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    //                kvo.int_value = std::atol(sep);
    //            } else if (strncmp(sep, "float:", 6) == 0) {
    //                sep += 6;
    //                kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
    //                kvo.float_value = std::atof(sep);
    //            } else if (strncmp(sep, "bool:", 5) == 0) {
    //                sep += 5;
    //                kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
    //                if (std::strcmp(sep, "true") == 0) {
    //                    kvo.bool_value = true;
    //                } else if (std::strcmp(sep, "false") == 0) {
    //                    kvo.bool_value = false;
    //                } else {
    //                    fprintf(stderr, "error: Invalid boolean value for KV override: %s\n", argv[i]);
    //                    invalid_param = true;
    //                    break;
    //                }
    //            } else {
    //                fprintf(stderr, "error: Invalid type for KV override: %s\n", argv[i]);
    //                invalid_param = true;
    //                break;
    //            }
    //            params.kv_overrides.push_back(kvo);
    //        }
    //    }
    //
    //    if (!params.kv_overrides.empty()) {
    //        params.kv_overrides.emplace_back();
    //        params.kv_overrides.back().key[0] = 0;
    //    }
}

//
static bool launch_slot(server_slot &slot, const server_task &task)
{
    slot_params default_params;
    llama_sampling_params default_sparams;
    auto &data = task.data;

    slot.oaicompat = false;
    slot.oaicompat_model = "";

    slot.params.stream = task.stream;
    slot.params.cache_prompt = env->GetBooleanField(jparams, f_cache_prompt);
    slot.params.n_predict = env->GetIntField(jparams, f_n_predict);
    slot.sparams.top_k = env->GetIntField(jparams, f_top_k);
    slot.sparams.top_p = env->GetFloatField(jparams, f_top_p);
    slot.sparams.min_p = env->GetFloatField(jparams, f_min_p);
    slot.sparams.tfs_z = env->GetFloatField(jparams, f_tfs_z);
    slot.sparams.typical_p = env->GetFloatField(jparams, f_typical_p);
    slot.sparams.temp = env->GetFloatField(jparams, f_temp);
    slot.sparams.dynatemp_range = env->GetFloatField(jparams, f_dynatemp_range);
    slot.sparams.dynatemp_exponent = env->GetFloatField(jparams, f_dynatemp_exponent);
    slot.sparams.penalty_last_n = env->GetIntField(jparams, f_penalty_last_n);
    slot.sparams.penalty_repeat = env->GetFloatField(jparams, f_penalty_repeat);
    slot.sparams.penalty_freq = env->GetFloatField(jparams, f_penalty_freq);
    slot.sparams.penalty_present = env->GetFloatField(jparams, f_penalty_present);

    auto mirostat = env->GetObjectField(jparams, f_mirostat);
    if (mirostat == o_mirostat_v1)
    {
        slot.sparams.mirostat = 1;
    }
    else if (mirostat == o_mirostat_v2)
    {
        slot.sparams.mirostat = 2;
    }
    else
    {
        slot.sparams.mirostat = 0;
    }
    slot.sparams.mirostat_tau = env->GetFloatField(jparams, f_mirostat_tau);
    slot.sparams.mirostat_eta = env->GetFloatField(jparams, f_mirostat_eta);
    slot.sparams.penalize_nl = env->GetBooleanField(jparams, f_penalize_nl);
    slot.params.n_keep = env->GetIntField(jparams, f_n_keep);
    slot.params.n_discard = env->GetIntField(jparams, f_n_discard);
    slot.params.seed = env->GetIntField(jparams, f_infer_seed);
    slot.sparams.n_probs = env->GetIntField(jparams, f_n_probs);
    slot.sparams.min_keep = env->GetIntField(jparams, f_min_keep);

    jstring j_grammar = (jstring)env->GetObjectField(jparams, f_grammar);
    if (j_grammar != nullptr)
    {
        slot.sparams.grammar = parse_jstring(env, j_grammar);
    }

    if (slot.params.cache_prompt && slot.ga_n != 1)
    {
        LOG_WARNING("cache_prompt is not supported with group-attention", {});
        slot.params.cache_prompt = false;
    }

    if (slot.n_predict > 0 && slot.params.n_predict > slot.n_predict)
    {
        // Might be better to reject the request with a 400 ?
        LOG_WARNING("Max tokens to predict exceeds server configuration",
                    {
                        {"params.n_predict", slot.params.n_predict},
                        {"slot.n_predict", slot.n_predict},
                    });
        slot.params.n_predict = slot.n_predict;
    }

    slot.prompt = task.prompt;
    slot.params.input_prefix = task.input_prefix;
    slot.params.input_suffix = task.input_suffix;

    // penalize user-provided tokens
    //	{
    //		slot.sparams.penalty_prompt_tokens.clear();
    //		slot.sparams.use_penalty_prompt_tokens = false;
    //
    //		const auto & penalty_prompt = data.find("penalty_prompt");
    //
    //		if (penalty_prompt != data.end()) {
    //			if (penalty_prompt->is_string()) {
    //				const auto penalty_prompt_string = penalty_prompt->get<std::string>();
    //				slot.sparams.penalty_prompt_tokens = llama_tokenize(model, penalty_prompt_string, false);
    //
    //				if (slot.params.n_predict > 0) {
    //					slot.sparams.penalty_prompt_tokens.reserve(slot.sparams.penalty_prompt_tokens.size() +
    // slot.params.n_predict);
    //				}
    //				slot.sparams.use_penalty_prompt_tokens = true;
    //
    //				LOG_VERBOSE("penalty_prompt_tokens", {
    //					{"id_slot", slot.id},
    //					{"tokens",  slot.sparams.penalty_prompt_tokens},
    //				});
    //			}
    //			else if (penalty_prompt->is_array()) {
    //				const auto n_tokens = penalty_prompt->size();
    //				slot.sparams.penalty_prompt_tokens.reserve(n_tokens + std::max(0, slot.params.n_predict));
    //
    //				const int n_vocab = llama_n_vocab(model);
    //				for (const auto & penalty_token : *penalty_prompt) {
    //					if (penalty_token.is_number_integer()) {
    //						const auto tok = penalty_token.get<llama_token>();
    //						if (tok >= 0 && tok < n_vocab) {
    //							slot.sparams.penalty_prompt_tokens.push_back(tok);
    //						}
    //					}
    //				}
    //				slot.sparams.use_penalty_prompt_tokens = true;
    //
    //				LOG_VERBOSE("penalty_prompt_tokens", {
    //					{"id_slot", slot.id},
    //					{"tokens",  slot.sparams.penalty_prompt_tokens},
    //				});
    //			}
    //		}
    //	}

    sparams.logit_bias.clear();
    jboolean ignore_eos = env->GetBooleanField(jparams, f_ignore_eos);
    if (ignore_eos)
    {
        slot.sparams.logit_bias[llama_token_eos(llama->model)] = -INFINITY;
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
            slot.sparams.logit_bias[tok] = bias;

            env->DeleteLocalRef(entry);
            env->DeleteLocalRef(key);
            env->DeleteLocalRef(value);
        }
    }

    slot.params.antiprompt.clear();
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
                slot.params.antiprompt.push_back(string);
                env->DeleteLocalRef(java_string);
            }
        }
    }

    //	{
    //		const auto & samplers_sequence = data.find("samplers");
    //		if (samplers_sequence != data.end() && samplers_sequence->is_array()) {
    //			std::vector<std::string> sampler_names;
    //			for (const auto & sampler_name : *samplers_sequence) {
    //				if (sampler_name.is_string()) {
    //					sampler_names.emplace_back(sampler_name);
    //				}
    //			}
    //			slot.sparams.samplers_sequence = sampler_types_from_names(sampler_names, false);
    //		} else {
    //			slot.sparams.samplers_sequence = default_sparams.samplers_sequence;
    //		}
    //	}

    //	{
    //		if (slot.ctx_sampling != nullptr) {
    //			llama_sampling_free(slot.ctx_sampling);
    //		}
    //		slot.ctx_sampling = llama_sampling_init(slot.sparams);
    //		if (slot.ctx_sampling == nullptr) {
    //			// for now, the only error that may happen here is invalid grammar
    //			send_error(task, "Failed to parse grammar", ERROR_TYPE_INVALID_REQUEST);
    //			return false;
    //		}
    //		llama_set_rng_seed(ctx, slot.params.seed);
    //	}

    slot.command = SLOT_COMMAND_LOAD_PROMPT;
    slot.prompt_tokens.clear();
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
    c_split_mode = env->FindClass("de/kherud/llama/args/GpuSplitMode");
    c_log_format = env->FindClass("de/kherud/llama/args/LogFormat");
    c_miro_stat = env->FindClass("de/kherud/llama/args/MiroStat");
    c_numa_strategy = env->FindClass("de/kherud/llama/args/NumaStrategy");
    c_pooling_type = env->FindClass("de/kherud/llama/args/PoolingType");
    c_rope_scaling = env->FindClass("de/kherud/llama/args/RopeScalingType");
    c_sampler = env->FindClass("de/kherud/llama/args/Sampler");

    if (!(c_llama_model && c_llama_iterator && c_infer_params && c_model_params && c_standard_charsets && c_output &&
          c_string && c_hash_map && c_map && c_set && c_entry && c_iterator && c_integer && c_float && c_log_level &&
          c_biconsumer && c_llama_error && c_error_oom && c_split_mode && c_log_format && c_miro_stat &&
          c_numa_strategy && c_pooling_type && c_rope_scaling && c_sampler))
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
    c_split_mode = (jclass)env->NewGlobalRef(c_split_mode);
    c_log_format = (jclass)env->NewGlobalRef(c_log_format);
    c_miro_stat = (jclass)env->NewGlobalRef(c_miro_stat);
    c_numa_strategy = (jclass)env->NewGlobalRef(c_numa_strategy);
    c_pooling_type = (jclass)env->NewGlobalRef(c_pooling_type);
    c_rope_scaling = (jclass)env->NewGlobalRef(c_rope_scaling);
    c_sampler = (jclass)env->NewGlobalRef(c_sampler);

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
    f_iter_has_next = env->GetFieldID(c_llama_iterator, "hasNext", "Z");
    f_iter_n_generated = env->GetFieldID(c_llama_iterator, "generatedCount", "J");
    f_iter_token_index = env->GetFieldID(c_llama_iterator, "tokenIndex", "J");

    if (!(f_model_pointer && f_iter_has_next && f_iter_n_generated && f_iter_token_index))
    {
        goto error;
    }

    // find inference parameters fields
    f_cache_prompt = env->GetFieldID(c_infer_params, "cachePrompt", "I");
    f_n_predict = env->GetFieldID(c_infer_params, "nPredict", "I");
    f_top_k = env->GetFieldID(c_infer_params, "topK", "I");
    f_top_p = env->GetFieldID(c_infer_params, "topP", "F");
    f_min_p = env->GetFieldID(c_infer_params, "minP", "F");
    f_tfs_z = env->GetFieldID(c_infer_params, "tfsZ", "F");
    f_typical_p = env->GetFieldID(c_infer_params, "typicalP", "F");
    f_temp = env->GetFieldID(c_infer_params, "temperature", "F");
    f_dynatemp_range = env->GetFieldID(c_infer_params, "dynamicTemperatureRange", "F");
    f_dynatemp_exponent = env->GetFieldID(c_infer_params, "dynamicTemperatureExponent", "F");
    f_penalty_last_n = env->GetFieldID(c_infer_params, "repeatLastN", "I");
    f_penalty_repeat = env->GetFieldID(c_infer_params, "repeatPenalty", "F");
    f_penalty_freq = env->GetFieldID(c_infer_params, "frequencyPenalty", "F");
    f_penalty_present = env->GetFieldID(c_infer_params, "presencePenalty", "F");
    f_mirostat = env->GetFieldID(c_infer_params, "mirostat", "Lde/kherud/llama/args/MiroStat;");
    f_mirostat_tau = env->GetFieldID(c_infer_params, "mirostatTau", "F");
    f_mirostat_eta = env->GetFieldID(c_infer_params, "mirostatEta", "F");
    f_penalize_nl = env->GetFieldID(c_infer_params, "penalizeNl", "Z");
    f_n_keep = env->GetFieldID(c_infer_params, "nKeep", "I");
    f_n_discard = env->GetFieldID(c_infer_params, "nDiscard", "I");
    f_infer_seed = env->GetFieldID(c_infer_params, "seed", "I");
    f_n_probs = env->GetFieldID(c_infer_params, "nProbs", "I");
    f_min_keep = env->GetFieldID(c_infer_params, "minKeep", "I");
    f_grammar = env->GetFieldID(c_infer_params, "grammar", "Ljava/lang/String;");
    f_ignore_eos = env->GetFieldID(c_infer_params, "ignoreEos", "Z");
    f_logit_bias = env->GetFieldID(c_infer_params, "logitBias", "Ljava/util/Map;");
    f_antiprompt = env->GetFieldID(c_infer_params, "stopStrings", "[Ljava/lang/String;");

    if (!(f_cache_prompt && f_n_predict && f_top_k && f_top_p && f_min_p && f_tfs_z && f_typical_p && f_temp &&
          f_dynatemp_range && f_dynatemp_exponent && f_penalty_last_n && f_penalty_repeat && f_penalty_freq &&
          f_penalty_present && f_mirostat && f_mirostat_tau && f_mirostat_eta && f_penalize_nl && f_n_keep &&
          f_n_discard && f_infer_seed && f_n_probs && f_min_keep && f_grammar && f_ignore_eos && f_logit_bias &&
          f_antiprompt))
    {
        goto error;
    }

	// find model parameters fields
	f_model_seed = env->GetFieldID(c_model_params, "seed", "I");
    f_model_path = env->GetFieldID(c_model_params, "modelFilePath", "Ljava/lang/String;");
    f_model_url = env->GetFieldID(c_model_params, "modelUrl", "Ljava/lang/String;");
    f_model_hf_repo = env->GetFieldID(c_model_params, "huggingFaceRepository", "Ljava/lang/String;");
    f_model_hf_file = env->GetFieldID(c_model_params, "huggingFaceFile", "Ljava/lang/String;");
    f_model_alias = env->GetFieldID(c_model_params, "modelAlias", "Ljava/lang/String;");
    f_n_ctx = env->GetFieldID(c_model_params, "nCtx", "I");
    f_rope_scaling_type = env->GetFieldID(c_model_params, "ropeScalingType", "Lde/kherud/llama/args/RopeScalingType;");
    f_rope_freq_base = env->GetFieldID(c_model_params, "ropeFreqBase", "F");
    f_rope_freq_scale = env->GetFieldID(c_model_params, "ropeFreqScale", "F");
    f_yarn_ext_factor = env->GetFieldID(c_model_params, "yarnExtFactor", "F");
    f_yarn_attn_factor = env->GetFieldID(c_model_params, "yarnAttnFactor", "F");
    f_yarn_beta_fast = env->GetFieldID(c_model_params, "yarnBetaFast", "F");
    f_yarn_beta_slow = env->GetFieldID(c_model_params, "yarnBetaSlow", "F");
    f_pooling_type = env->GetFieldID(c_model_params, "poolingType", "Lde/kherud/llama/args/PoolingType;");
    f_defrag_thold = env->GetFieldID(c_model_params, "defragmentationThreshold", "F");
    f_n_threads = env->GetFieldID(c_model_params, "nThreads", "I");
    f_grp_attn_n = env->GetFieldID(c_model_params, "groupAttnN", "I");
    f_grp_attn_w = env->GetFieldID(c_model_params, "groupAttnW", "I");
    f_n_threads_batch = env->GetFieldID(c_model_params, "nThreadsBatch", "I");
    f_n_batch = env->GetFieldID(c_model_params, "nBatch", "I");
    f_n_ubatch = env->GetFieldID(c_model_params, "nUBatch", "I");
    f_n_gpu_layers = env->GetFieldID(c_model_params, "nGpuLayers", "I");
    f_no_kv_offload = env->GetFieldID(c_model_params, "noKVOffload", "Z");
    f_split_mode = env->GetFieldID(c_model_params, "gpuSplitMode", "Lde/kherud/llama/args/GpuSplitMode;");
    f_tensor_split = env->GetFieldID(c_model_params, "tensorSplit", "[F;");
    f_main_gpu = env->GetFieldID(c_model_params, "mainGpu", "I");
    f_verbose = env->GetFieldID(c_model_params, "verbose", "Z");
    f_use_mlock = env->GetFieldID(c_model_params, "useMlock", "Z");
    f_use_mmap = env->GetFieldID(c_model_params, "useMmap", "Z");
    f_numa_strategy = env->GetFieldID(c_model_params, "numa", "Lde/kherud/llama/args/NumaStrategy;");
    f_embedding = env->GetFieldID(c_model_params, "embedding", "Z");
    f_cont_batching = env->GetFieldID(c_model_params, "continuousBatching", "Z");
    f_n_parallel = env->GetFieldID(c_model_params, "nParallel", "I");
    f_n_predict = env->GetFieldID(c_model_params, "nPredict", "I");
    f_system_prompt_file = env->GetFieldID(c_model_params, "systemPromptFile", "Ljava/lang/String;");
    f_log_format = env->GetFieldID(c_model_params, "logFormat", "Lde/kherud/llama/args/LogFormat;");

    if (!(f_model_seed && f_model_path && f_model_url && f_model_hf_repo && f_model_hf_file && f_model_alias &&
          f_n_ctx && f_rope_scaling_type && f_rope_freq_base && f_rope_freq_scale && f_yarn_ext_factor &&
          f_yarn_attn_factor && f_yarn_beta_fast && f_yarn_beta_slow && f_pooling_type && f_defrag_thold &&
          f_n_threads && f_grp_attn_n && f_grp_attn_w && f_n_threads_batch && f_n_batch && f_n_ubatch &&
          f_n_gpu_layers && f_no_kv_offload && f_split_mode && f_tensor_split && f_main_gpu && f_verbose &&
          f_use_mlock && f_use_mmap && f_numa_strategy && f_embedding && f_cont_batching && f_n_parallel &&
          f_n_predict && f_system_prompt_file && f_log_format))
    {
        goto error;
    }

    f_utf_8 = env->GetStaticFieldID(c_standard_charsets, "UTF_8", "Ljava/nio/charset/Charset;");

    f_log_level_debug = env->GetStaticFieldID(c_log_level, "DEBUG", "Lde/kherud/llama/LogLevel;");
    f_log_level_info = env->GetStaticFieldID(c_log_level, "INFO", "Lde/kherud/llama/LogLevel;");
    f_log_level_warn = env->GetStaticFieldID(c_log_level, "WARN", "Lde/kherud/llama/LogLevel;");
    f_log_level_error = env->GetStaticFieldID(c_log_level, "ERROR", "Lde/kherud/llama/LogLevel;");

    f_rope_scaling_none = env->GetStaticFieldID(c_log_level, "UNSPECIFIED", "Lde/kherud/llama/args/RopeScalingType;");
    f_rope_scaling_linear = env->GetStaticFieldID(c_log_level, "LINEAR", "Lde/kherud/llama/args/RopeScalingType;");
    f_rope_scaling_yarn = env->GetStaticFieldID(c_log_level, "YARN", "Lde/kherud/llama/args/RopeScalingType;");

    f_pooling_type_none = env->GetStaticFieldID(c_log_level, "UNSPECIFIED", "Lde/kherud/llama/args/PoolingType;");
    f_pooling_type_mean = env->GetStaticFieldID(c_log_level, "MEAN", "Lde/kherud/llama/args/PoolingType;");
    f_pooling_type_cls = env->GetStaticFieldID(c_log_level, "CLS", "Lde/kherud/llama/args/PoolingType;");

    f_split_mode_none = env->GetStaticFieldID(c_log_level, "NONE", "Lde/kherud/llama/args/GpuSplitMode;");
    f_split_mode_layer = env->GetStaticFieldID(c_log_level, "LAYER", "Lde/kherud/llama/args/GpuSplitMode;");
    f_split_mode_row = env->GetStaticFieldID(c_log_level, "ROW", "Lde/kherud/llama/args/GpuSplitMode;");

    f_numa_strategy_distribute =
        env->GetStaticFieldID(c_log_level, "DISTRIBUTE", "Lde/kherud/llama/args/NumaStrategy;");
    f_numa_strategy_isolate = env->GetStaticFieldID(c_log_level, "ISOLATE", "Lde/kherud/llama/args/NumaStrategy;");
    f_numa_strategy_numactl = env->GetStaticFieldID(c_log_level, "NUMA_CTL", "Lde/kherud/llama/args/NumaStrategy;");

    f_log_format_json = env->GetStaticFieldID(c_log_level, "JSON", "Lde/kherud/llama/args/LogFormat;");
    f_log_format_text = env->GetStaticFieldID(c_log_level, "TEXT", "Lde/kherud/llama/args/LogFormat;");

    f_mirostat_v1 = env->GetStaticFieldID(c_log_level, "V1", "Lde/kherud/llama/args/MiroStat;");
    f_mirostat_v2 = env->GetStaticFieldID(c_log_level, "V2", "Lde/kherud/llama/args/MiroStat;");

    if (!(f_utf_8 && f_log_level_debug && f_log_level_info && f_log_level_warn && f_log_level_error &&
          f_rope_scaling_none && f_rope_scaling_linear && f_rope_scaling_yarn && f_pooling_type_none &&
          f_pooling_type_mean && f_pooling_type_cls && f_split_mode_none && f_split_mode_layer && f_split_mode_row &&
          f_numa_strategy_distribute && f_numa_strategy_isolate && f_numa_strategy_numactl && f_log_format_json &&
          f_log_format_text && f_mirostat_v1 && f_mirostat_v2))
    {
        goto error;
    }

    //    o_utf_8 = env->GetStaticObjectField(c_standard_charsets, f_utf_8);
    o_utf_8 = env->NewStringUTF("UTF-8");
    o_utf_8 = (jclass)env->NewGlobalRef(o_utf_8);

    o_log_level_debug = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_log_level, f_log_level_debug));
    o_log_level_info = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_log_level, f_log_level_info));
    o_log_level_warn = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_log_level, f_log_level_warn));
    o_log_level_error = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_log_level, f_log_level_error));

    o_rope_scaling_none = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_rope_scaling, f_rope_scaling_none));
    o_rope_scaling_linear = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_rope_scaling, f_rope_scaling_linear));
    o_rope_scaling_yarn = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_rope_scaling, f_rope_scaling_yarn));

    o_pooling_type_none = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_pooling_type, f_pooling_type_none));
    o_pooling_type_mean = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_pooling_type, f_pooling_type_mean));
    o_pooling_type_cls = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_pooling_type, f_pooling_type_cls));

    o_split_mode_none = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_split_mode, f_split_mode_none));
    o_split_mode_layer = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_split_mode, f_split_mode_layer));
    o_split_mode_row = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_split_mode, f_split_mode_row));

    o_numa_strategy_distribute = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_numa_strategy, f_numa_strategy_distribute));
    o_numa_strategy_isolate = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_numa_strategy, f_numa_strategy_isolate));
    o_numa_strategy_numactl = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_numa_strategy, f_numa_strategy_numactl));

    o_log_format_json = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_log_format, f_log_format_json));
    o_log_format_text = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_log_format, f_log_format_text));

    o_mirostat_v1 = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_miro_stat, f_mirostat_v1));
    o_mirostat_v2 = (jobject)env->NewGlobalRef(env->GetStaticObjectField(c_miro_stat, f_mirostat_v2));

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
    env->DeleteGlobalRef(c_split_mode);
    env->DeleteGlobalRef(c_log_format);
    env->DeleteGlobalRef(c_miro_stat);
    env->DeleteGlobalRef(c_numa_strategy);
    env->DeleteGlobalRef(c_pooling_type);
    env->DeleteGlobalRef(c_rope_scaling);
    env->DeleteGlobalRef(c_sampler);

    env->DeleteGlobalRef(o_utf_8);
    env->DeleteGlobalRef(o_log_level_debug);
    env->DeleteGlobalRef(o_log_level_info);
    env->DeleteGlobalRef(o_log_level_warn);
    env->DeleteGlobalRef(o_log_level_error);
    env->DeleteGlobalRef(o_rope_scaling_none);
    env->DeleteGlobalRef(o_rope_scaling_linear);
    env->DeleteGlobalRef(o_rope_scaling_yarn);
    env->DeleteGlobalRef(o_pooling_type_none);
    env->DeleteGlobalRef(o_pooling_type_mean);
    env->DeleteGlobalRef(o_pooling_type_cls);
    env->DeleteGlobalRef(o_split_mode_none);
    env->DeleteGlobalRef(o_split_mode_layer);
    env->DeleteGlobalRef(o_split_mode_row);
    env->DeleteGlobalRef(o_numa_strategy_distribute);
    env->DeleteGlobalRef(o_numa_strategy_isolate);
    env->DeleteGlobalRef(o_numa_strategy_numactl);
    env->DeleteGlobalRef(o_log_format_json);
    env->DeleteGlobalRef(o_log_format_text);
    env->DeleteGlobalRef(o_mirostat_v1);
    env->DeleteGlobalRef(o_mirostat_v2);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_loadModel(JNIEnv *env, jobject obj, jstring file_path,
                                                                 jobject jparams)
{
    gpt_params params;
    server_params sparams;

    server_context ctx_server;

    server_params_parse(env, jparams, sparams, params);

    if (!sparams.system_prompt.empty())
    {
        ctx_server.system_prompt_set(sparams.system_prompt);
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
    if (!ctx_server.load_model(params))
    {
        state.store(SERVER_STATE_ERROR);
        env->ThrowNew(c_llama_error, "could not load model from given file path");
        return;
    }
    else
    {
        ctx_server.init();
        state.store(SERVER_STATE_READY);
    }

    LOG_INFO("model loaded", {});

    const auto model_meta = ctx_server.model_meta();

    // if a custom chat template is not supplied, we will use the one that comes with the model (if any)
    if (sparams.chat_template.empty())
    {
        if (!ctx_server.validate_model_chat_template())
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

        const std::string chat_example = format_chat(ctx_server.model, sparams.chat_template, chat);

        LOG_INFO("chat template", {
                                      {"chat_example", chat_example},
                                      {"built_in", sparams.chat_template.empty()},
                                  });
    }

    env->SetLongField(obj, f_model_pointer, reinterpret_cast<jlong>(llama));
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
// JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_getAnswer(JNIEnv *env, jobject obj, jstring prompt,
//                                                                        jobject params)
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
//     setup_answering(env, llama, prompt, params);
//
//     llama->loadPrompt();
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
