# Model : zai-org/GLM-5-FP8

### Trail 1 : 

#### Docker Image : rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260218
#### Transformers : 4.57.1

#### Command : 
```
python3 -m sglang.launch_server --attention-backend triton --model-path zai-org/GLM-5-FP8 --tp-size 8
```

#### Error Log : 

<details>
<summary>Click to view Error Log</summary>
  
```bash
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 04:55:25] INFO core.py:501: import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
Traceback (most recent call last):
  File "/opt/venv/lib/python3.10/site-packages/transformers/models/auto/configuration_auto.py", line 1360, in from_pretrained
    config_class = CONFIG_MAPPING[config_dict["model_type"]]
  File "/opt/venv/lib/python3.10/site-packages/transformers/models/auto/configuration_auto.py", line 1048, in __getitem__
    raise KeyError(key)
KeyError: 'glm_moe_dsa'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/sgl-workspace/sglang/python/sglang/launch_server.py", line 32, in <module>
    server_args = prepare_server_args(sys.argv[1:])
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 5563, in prepare_server_args
    return ServerArgs.from_cli_args(raw_args)
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 5049, in from_cli_args
    return cls(**{attr: getattr(args, attr) for attr in attrs})
  File "<string>", line 330, in __init__
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 730, in __post_init__
    self._handle_gpu_memory_settings(gpu_mem)
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 1006, in _handle_gpu_memory_settings
    if not self.use_mla_backend():
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 5082, in use_mla_backend
    model_config = self.get_model_config()
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 5063, in get_model_config
    self.model_config = ModelConfig.from_server_args(self)
  File "/sgl-workspace/sglang/python/sglang/srt/configs/model_config.py", line 250, in from_server_args
    return ModelConfig(
  File "/sgl-workspace/sglang/python/sglang/srt/configs/model_config.py", line 127, in __init__
    self.hf_config = get_config(
  File "/sgl-workspace/sglang/python/sglang/srt/utils/common.py", line 3475, in wrapper
    result = func(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/utils/hf_transformers_utils.py", line 320, in get_config
    raise e
  File "/sgl-workspace/sglang/python/sglang/srt/utils/hf_transformers_utils.py", line 315, in get_config
    config = AutoConfig.from_pretrained(
  File "/opt/venv/lib/python3.10/site-packages/transformers/models/auto/configuration_auto.py", line 1362, in from_pretrained
    raise ValueError(
ValueError: The checkpoint you are trying to load has model type `glm_moe_dsa` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

You can update Transformers with the command `pip install --upgrade transformers`. If this does not work, and the checkpoint is very new, then there may not be a release version that supports this model yet. In this case, you can get the most up-to-date code by installing Transformers from source with the command `pip install git+https://github.com/huggingface/transformers.git`

```
</details>

#### Resolution : 
```
pip install git+https://github.com/huggingface/transformers.git
```
#### Output : 
```
Successfully installed huggingface-hub-1.4.1 markdown-it-py-4.0.0 mdurl-0.1.2 rich-14.3.3 shellingham-1.5.4 transformers-5.3.0.dev0 typer-0.24.1 typer-slim-0.24.0
```
#### Command : 
```
python3 -m sglang.launch_server --attention-backend triton --model-path zai-org/GLM-5-FP8 --tp-size 8 --kv-cache-dtype fp8_e4m3
```

#### Error Log : 

<details>
<summary>Click to view Error Log</summary>
  
```bash
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 05:19:56] INFO core.py:501: import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 05:19:58] INFO _client.py:1025: HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:19:58] INFO _client.py:1025: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:19:58] INFO _client.py:1025: HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:19:58] INFO _client.py:1025: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/generation_config.json "HTTP/1.1 200 OK"
The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
[2026-02-26 05:19:58] WARNING model_config.py:1025: Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:19:58] WARNING server_args.py:1245: DSA with TP mode is active, dp_size=1, tp_size=8, attn_tp_size=8, attention weights will be sharded across 8 ranks.
[2026-02-26 05:19:58] WARNING server_args.py:1252: Setting page size to 1 for DeepSeek DSA on ROCm.
[2026-02-26 05:19:58] WARNING server_args.py:1177: Set NSA backends for fp8_e4m3 KV Cache: prefill=flashmla_auto, decode=flashmla_kv.
[2026-02-26 05:19:58] server_args=ServerArgs(model_path='zai-org/GLM-5-FP8', tokenizer_path='zai-org/GLM-5-FP8', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='fp8_e4m3', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.908, max_running_requests=None, max_queued_requests=None, max_total_tokens=None, chunked_prefill_size=16384, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=8, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, random_seed=77808626, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=None, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='info', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='zai-org/GLM-5-FP8', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='triton', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='pytorch', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend='flashmla_auto', nsa_decode_backend='flashmla_kv', disable_flashinfer_autotune=False, speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method='AMXINT4', kt_cpuinfer=None, kt_threadpool_count=2, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=512, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, enable_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=2048, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=16, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_decode_tp=None, disaggregation_decode_dp=None, disaggregation_prefill_pp=1, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)
[2026-02-26 05:19:58] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:19:58] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:19:58] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:19:59] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:19:59] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/tokenizer_config.json "HTTP/1.1 200 OK"
[2026-02-26 05:19:59] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
[2026-02-26 05:19:59] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
[2026-02-26 05:20:00] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8 "HTTP/1.1 200 OK"
[2026-02-26 05:20:00] Using default HuggingFace chat template with detected content format: openai
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 05:20:02] INFO core.py:501: import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 05:20:02] INFO core.py:501: import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 05:20:02] INFO core.py:501: import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 05:20:02] INFO core.py:501: import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 05:20:03] INFO core.py:501: import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 05:20:03] INFO core.py:501: import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 05:20:03] INFO core.py:501: import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 05:20:03] INFO core.py:501: import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 05:20:03] INFO core.py:501: import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 05:20:04 TP7] Process 5142 gpu_id 7 is running on CPUs: [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
[2026-02-26 05:20:04 TP7] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:04 TP7] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:04 TP3] Process 5138 gpu_id 3 is running on CPUs: [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
[2026-02-26 05:20:04 TP7] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:04 TP7] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/generation_config.json "HTTP/1.1 200 OK"
The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
[2026-02-26 05:20:04 TP7] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:04 TP7] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP7] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP3] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP3] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP2] Process 5137 gpu_id 2 is running on CPUs: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
[2026-02-26 05:20:05 TP7] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP7] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/tokenizer_config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP3] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP3] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/generation_config.json "HTTP/1.1 200 OK"
The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
[2026-02-26 05:20:05 TP3] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:05 TP7] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
[2026-02-26 05:20:05 TP6] Process 5141 gpu_id 6 is running on CPUs: [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
[2026-02-26 05:20:05 TP3] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP3] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP2] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP2] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP7] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP3] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP2] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[2026-02-26 05:20:05 TP2] Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[2026-02-26 05:20:05 TP3] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/tokenizer_config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP2] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/generation_config.json "HTTP/1.1 200 OK"
The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
[2026-02-26 05:20:05 TP2] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:05 TP6] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP0] Process 5135 gpu_id 0 is running on CPUs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[2026-02-26 05:20:05 TP6] Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[2026-02-26 05:20:05 TP6] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP3] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
[2026-02-26 05:20:05 TP2] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP2] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP6] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP6] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/generation_config.json "HTTP/1.1 200 OK"
The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
[2026-02-26 05:20:05 TP6] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:05 TP5] Process 5140 gpu_id 5 is running on CPUs: [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
[2026-02-26 05:20:05 TP3] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP2] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP1] Process 5136 gpu_id 1 is running on CPUs: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
[2026-02-26 05:20:05 TP2] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/tokenizer_config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP0] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP0] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP6] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP6] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP2] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
[2026-02-26 05:20:05 TP0] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP4] Process 5139 gpu_id 4 is running on CPUs: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
[2026-02-26 05:20:05 TP0] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/generation_config.json "HTTP/1.1 200 OK"
The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
[2026-02-26 05:20:05 TP0] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:05 TP5] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP5] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP6] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP1] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[2026-02-26 05:20:05 TP1] Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[2026-02-26 05:20:05 TP6] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/tokenizer_config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP1] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP2] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP0] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP0] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP5] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP5] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/generation_config.json "HTTP/1.1 200 OK"
The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
[2026-02-26 05:20:05 TP5] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:05 TP6] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
[2026-02-26 05:20:05 TP1] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP1] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/generation_config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP4] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
[2026-02-26 05:20:05 TP1] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:05 TP4] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP0] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP0] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/tokenizer_config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP5] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP5] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP1] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP6] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP4] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP1] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP4] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/generation_config.json "HTTP/1.1 200 OK"
The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
[2026-02-26 05:20:05 TP4] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:05 TP0] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
[2026-02-26 05:20:05 TP5] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP4] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP4] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP5] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/tokenizer_config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP1] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP1] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/tokenizer_config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP0] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
[2026-02-26 05:20:05] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP4] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05 TP4] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/tokenizer_config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP5] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
[2026-02-26 05:20:05 TP1] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
[2026-02-26 05:20:05] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
[2026-02-26 05:20:05] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/zai-org/GLM-5-FP8/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1/tokenizer_config.json "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP4] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
[2026-02-26 05:20:05 TP5] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
[2026-02-26 05:20:05 TP1] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
[2026-02-26 05:20:05] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
[2026-02-26 05:20:05 TP4] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
[2026-02-26 05:20:05] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
[2026-02-26 05:20:06 TP7] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8 "HTTP/1.1 200 OK"
[2026-02-26 05:20:06 TP7] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:06 TP7] Init torch distributed begin.
[2026-02-26 05:20:06 TP7] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 05:20:06 TP3] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8 "HTTP/1.1 200 OK"
[2026-02-26 05:20:06 TP3] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:06 TP3] Init torch distributed begin.
[2026-02-26 05:20:06 TP3] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 05:20:06 TP6] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8 "HTTP/1.1 200 OK"
[2026-02-26 05:20:06 TP6] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:06 TP6] Init torch distributed begin.
[2026-02-26 05:20:06 TP6] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 05:20:06 TP2] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8 "HTTP/1.1 200 OK"
[2026-02-26 05:20:06 TP2] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:06 TP2] Init torch distributed begin.
[2026-02-26 05:20:06 TP2] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 05:20:06 TP0] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8 "HTTP/1.1 200 OK"
[2026-02-26 05:20:06 TP0] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:06 TP5] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8 "HTTP/1.1 200 OK"
[2026-02-26 05:20:06 TP1] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8 "HTTP/1.1 200 OK"
[2026-02-26 05:20:06 TP0] Init torch distributed begin.
[2026-02-26 05:20:06 TP0] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 05:20:06 TP5] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:06 TP1] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:06 TP5] Init torch distributed begin.
[2026-02-26 05:20:06 TP5] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 05:20:06 TP1] Init torch distributed begin.
[2026-02-26 05:20:06 TP1] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 05:20:06 TP4] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8 "HTTP/1.1 200 OK"
[2026-02-26 05:20:06] HTTP Request: GET https://huggingface.co/api/models/zai-org/GLM-5-FP8 "HTTP/1.1 200 OK"
[2026-02-26 05:20:06 TP4] Transformers version 5.3.0.dev0 is used for model type glm_moe_dsa. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
[2026-02-26 05:20:06 TP4] Init torch distributed begin.
[2026-02-26 05:20:06 TP4] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[Gloo] Rank 0 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 1 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 4 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 5 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 2 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 3 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 7 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 6 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[2026-02-26 05:20:07 TP7] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 05:20:07 TP0] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 05:20:07 TP4] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 05:20:07 TP2] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 05:20:07 TP1] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 05:20:07 TP5] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 05:20:07 TP3] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 05:20:07 TP6] [AR] All-reduce call path: NCCL (custom AR disabled)
[Gloo] Rank 0 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 2 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 1 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 5 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 3 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 6 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 4 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 7 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[2026-02-26 05:20:07 TP0] sglang is using nccl==2.27.7
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 05:20:19 TP7] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 05:20:19 TP3] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 05:20:19 TP6] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 05:20:19 TP1] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 05:20:19 TP2] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 05:20:19 TP0] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 05:20:19 TP3] [AR] Using AiterCustomAllreduce (AMD default)
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 05:20:19 TP4] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 05:20:19 TP6] [AR] Using AiterCustomAllreduce (AMD default)
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 05:20:19 TP5] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 05:20:19 TP1] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 05:20:19 TP2] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 05:20:19 TP0] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 05:20:19 TP4] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 05:20:19 TP5] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 05:20:19 TP7] [AR] Using AiterCustomAllreduce (AMD default)
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 05:20:19 TP0] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 05:20:19 TP7] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 05:20:19 TP1] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 05:20:19 TP5] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 05:20:19 TP3] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 05:20:19 TP6] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 05:20:19 TP2] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 05:20:19 TP4] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 05:20:19 TP7] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 05:20:19 TP3] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 05:20:19 TP0] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 05:20:19 TP1] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 05:20:19 TP2] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 05:20:19 TP4] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 05:20:19 TP5] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 05:20:19 TP6] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP7] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP3] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP0] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP1] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP2] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP5] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP4] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP6] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP7] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP0] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP3] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP4] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP2] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP1] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP5] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP6] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 05:20:19 TP0] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 05:20:19 TP2] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 05:20:19 TP4] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 05:20:19 TP1] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 05:20:19 TP3] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 05:20:19 TP7] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 05:20:19 TP6] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 05:20:19 TP5] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[2026-02-26 05:20:19 TP0] Init torch distributed ends. elapsed=13.00 s, mem usage=3.81 GB
[2026-02-26 05:20:19 TP7] Init torch distributed ends. elapsed=13.46 s, mem usage=3.69 GB
[2026-02-26 05:20:19 TP6] Init torch distributed ends. elapsed=13.13 s, mem usage=3.70 GB
[2026-02-26 05:20:19 TP5] Init torch distributed ends. elapsed=12.97 s, mem usage=3.76 GB
[2026-02-26 05:20:19 TP4] Init torch distributed ends. elapsed=12.85 s, mem usage=3.69 GB
[2026-02-26 05:20:19 TP3] Init torch distributed ends. elapsed=13.28 s, mem usage=3.83 GB
[2026-02-26 05:20:19 TP2] Init torch distributed ends. elapsed=13.06 s, mem usage=3.83 GB
[2026-02-26 05:20:19 TP1] Init torch distributed ends. elapsed=12.97 s, mem usage=3.41 GB
/opt/venv/lib/python3.10/site-packages/apex/transformer/functional/fused_rope.py:49: UserWarning: Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0
  warnings.warn("Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0", UserWarning)
/opt/venv/lib/python3.10/site-packages/apex/transformer/functional/fused_rope.py:49: UserWarning: Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0
  warnings.warn("Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0", UserWarning)
/opt/venv/lib/python3.10/site-packages/apex/transformer/functional/fused_rope.py:49: UserWarning: Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0
  warnings.warn("Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0", UserWarning)
/opt/venv/lib/python3.10/site-packages/apex/transformer/functional/fused_rope.py:49: UserWarning: Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0
  warnings.warn("Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0", UserWarning)
/opt/venv/lib/python3.10/site-packages/apex/transformer/functional/fused_rope.py:49: UserWarning: Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0
  warnings.warn("Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0", UserWarning)
/opt/venv/lib/python3.10/site-packages/apex/transformer/functional/fused_rope.py:49: UserWarning: Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0
  warnings.warn("Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0", UserWarning)
/opt/venv/lib/python3.10/site-packages/apex/transformer/functional/fused_rope.py:49: UserWarning: Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0
  warnings.warn("Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0", UserWarning)
/opt/venv/lib/python3.10/site-packages/apex/transformer/functional/fused_rope.py:49: UserWarning: Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0
  warnings.warn("Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0", UserWarning)
[2026-02-26 05:20:20 TP7] Load weight begin. avail mem=283.62 GB
[2026-02-26 05:20:20 TP3] Load weight begin. avail mem=283.48 GB
[2026-02-26 05:20:20 TP1] Load weight begin. avail mem=283.90 GB
[2026-02-26 05:20:20 TP6] Load weight begin. avail mem=283.61 GB
[2026-02-26 05:20:20 TP5] Load weight begin. avail mem=283.55 GB
[2026-02-26 05:20:21 TP0] Load weight begin. avail mem=283.50 GB
[2026-02-26 05:20:21 TP0] Detected fp8 checkpoint.
[2026-02-26 05:20:21 TP4] Load weight begin. avail mem=283.62 GB
[2026-02-26 05:20:21 TP2] Load weight begin. avail mem=283.48 GB
[2026-02-26 05:20:21 TP7] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/model.safetensors.index.json "HTTP/1.1 302 Found"
[2026-02-26 05:20:21 TP0] Shared experts fusion optimization enabled.
[2026-02-26 05:20:21 TP5] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/model.safetensors.index.json "HTTP/1.1 302 Found"
[2026-02-26 05:20:21 TP0] Found local HF snapshot for zai-org/GLM-5-FP8 at /home/models/models--zai-org--GLM-5-FP8/snapshots/7ca2d2f1f1703aa0b189977fe3c126caf18b70e1; skipping download.
[2026-02-26 05:20:21 TP0] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/model.safetensors.index.json "HTTP/1.1 302 Found"

Loading safetensors checkpoint shards:   0% Completed | 0/142 [00:00<?, ?it/s]
[2026-02-26 05:20:21 TP4] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/model.safetensors.index.json "HTTP/1.1 302 Found"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[2026-02-26 05:20:21 TP4] Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[2026-02-26 05:20:21 TP2] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/model.safetensors.index.json "HTTP/1.1 302 Found"

Loading safetensors checkpoint shards:   3% Completed | 4/142 [00:00<00:04, 33.56it/s]
[2026-02-26 05:20:21 TP1] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/model.safetensors.index.json "HTTP/1.1 302 Found"

Loading safetensors checkpoint shards:   6% Completed | 8/142 [00:00<00:04, 30.77it/s]
[2026-02-26 05:20:21 TP6] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/model.safetensors.index.json "HTTP/1.1 302 Found"

Loading safetensors checkpoint shards:   8% Completed | 12/142 [00:00<00:04, 29.73it/s]
[2026-02-26 05:20:21 TP3] HTTP Request: HEAD https://huggingface.co/zai-org/GLM-5-FP8/resolve/main/model.safetensors.index.json "HTTP/1.1 302 Found"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[2026-02-26 05:20:21 TP3] Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Loading safetensors checkpoint shards:  11% Completed | 15/142 [00:01<00:11, 11.14it/s]

Loading safetensors checkpoint shards:  13% Completed | 18/142 [00:01<00:09, 13.73it/s]

Loading safetensors checkpoint shards:  15% Completed | 21/142 [00:01<00:07, 16.13it/s]

Loading safetensors checkpoint shards:  17% Completed | 24/142 [00:01<00:06, 18.34it/s]

Loading safetensors checkpoint shards:  19% Completed | 27/142 [00:01<00:05, 19.67it/s]

Loading safetensors checkpoint shards:  21% Completed | 30/142 [00:01<00:05, 20.95it/s]

Loading safetensors checkpoint shards:  23% Completed | 33/142 [00:01<00:05, 20.74it/s]

Loading safetensors checkpoint shards:  25% Completed | 36/142 [00:02<00:11,  9.49it/s]

Loading safetensors checkpoint shards:  28% Completed | 40/142 [00:02<00:08, 12.72it/s]

Loading safetensors checkpoint shards:  30% Completed | 43/142 [00:02<00:06, 15.01it/s]

Loading safetensors checkpoint shards:  33% Completed | 47/142 [00:02<00:05, 17.80it/s]

Loading safetensors checkpoint shards:  35% Completed | 50/142 [00:02<00:04, 19.46it/s]

Loading safetensors checkpoint shards:  37% Completed | 53/142 [00:03<00:04, 20.80it/s]

Loading safetensors checkpoint shards:  39% Completed | 56/142 [00:03<00:03, 22.18it/s]

Loading safetensors checkpoint shards:  42% Completed | 59/142 [00:04<00:09,  8.71it/s]

Loading safetensors checkpoint shards:  44% Completed | 62/142 [00:04<00:07, 10.87it/s]

Loading safetensors checkpoint shards:  46% Completed | 65/142 [00:04<00:05, 13.20it/s]

Loading safetensors checkpoint shards:  48% Completed | 68/142 [00:04<00:04, 15.54it/s]

Loading safetensors checkpoint shards:  50% Completed | 71/142 [00:04<00:04, 17.15it/s]

Loading safetensors checkpoint shards:  52% Completed | 74/142 [00:04<00:03, 19.06it/s]

Loading safetensors checkpoint shards:  54% Completed | 77/142 [00:04<00:03, 19.60it/s]

Loading safetensors checkpoint shards:  56% Completed | 80/142 [00:04<00:03, 20.36it/s]

Loading safetensors checkpoint shards:  58% Completed | 83/142 [00:05<00:02, 20.39it/s]

Loading safetensors checkpoint shards:  61% Completed | 86/142 [00:05<00:02, 20.97it/s]

Loading safetensors checkpoint shards:  63% Completed | 89/142 [00:06<00:06,  7.89it/s]

Loading safetensors checkpoint shards:  65% Completed | 92/142 [00:06<00:05,  9.94it/s]

Loading safetensors checkpoint shards:  67% Completed | 95/142 [00:06<00:03, 12.32it/s]

Loading safetensors checkpoint shards:  69% Completed | 98/142 [00:06<00:03, 14.02it/s]

Loading safetensors checkpoint shards:  71% Completed | 101/142 [00:06<00:02, 15.83it/s]

Loading safetensors checkpoint shards:  73% Completed | 104/142 [00:06<00:02, 17.84it/s]

Loading safetensors checkpoint shards:  75% Completed | 107/142 [00:06<00:01, 19.41it/s]

Loading safetensors checkpoint shards:  77% Completed | 110/142 [00:07<00:01, 19.88it/s]

Loading safetensors checkpoint shards:  80% Completed | 113/142 [00:07<00:01, 21.16it/s]

Loading safetensors checkpoint shards:  82% Completed | 116/142 [00:07<00:01, 20.99it/s]

Loading safetensors checkpoint shards:  84% Completed | 119/142 [00:07<00:01, 22.24it/s]

Loading safetensors checkpoint shards:  86% Completed | 122/142 [00:07<00:00, 22.49it/s]

Loading safetensors checkpoint shards:  88% Completed | 125/142 [00:07<00:00, 22.19it/s]

Loading safetensors checkpoint shards:  90% Completed | 128/142 [00:08<00:02,  7.00it/s]

Loading safetensors checkpoint shards:  92% Completed | 131/142 [00:08<00:01,  9.00it/s]

Loading safetensors checkpoint shards:  94% Completed | 134/142 [00:09<00:00, 10.99it/s]

Loading safetensors checkpoint shards:  98% Completed | 139/142 [00:09<00:00, 15.95it/s]

Loading safetensors checkpoint shards: 100% Completed | 142/142 [00:09<00:00, 17.76it/s]

Loading safetensors checkpoint shards: 100% Completed | 142/142 [00:09<00:00, 15.36it/s]

[2026-02-26 05:24:09 TP4] Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0. This may lead to less accurate results!
[2026-02-26 05:24:09 TP4] Load weight end. elapsed=228.09 s, type=GlmMoeDsaForCausalLM, dtype=torch.bfloat16, avail mem=194.04 GB, mem usage=89.58 GB.
[2026-02-26 05:24:09 TP5] Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0. This may lead to less accurate results!
[2026-02-26 05:24:09 TP5] Load weight end. elapsed=228.53 s, type=GlmMoeDsaForCausalLM, dtype=torch.bfloat16, avail mem=193.96 GB, mem usage=89.58 GB.
[2026-02-26 05:24:12 TP6] Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0. This may lead to less accurate results!
[2026-02-26 05:24:12 TP6] Load weight end. elapsed=231.42 s, type=GlmMoeDsaForCausalLM, dtype=torch.bfloat16, avail mem=194.03 GB, mem usage=89.58 GB.
[2026-02-26 05:24:15 TP3] Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0. This may lead to less accurate results!
[2026-02-26 05:24:15 TP3] Load weight end. elapsed=234.49 s, type=GlmMoeDsaForCausalLM, dtype=torch.bfloat16, avail mem=193.90 GB, mem usage=89.58 GB.
[2026-02-26 05:24:17 TP1] Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0. This may lead to less accurate results!
[2026-02-26 05:24:17 TP1] Load weight end. elapsed=236.31 s, type=GlmMoeDsaForCausalLM, dtype=torch.bfloat16, avail mem=194.32 GB, mem usage=89.58 GB.
[2026-02-26 05:24:18 TP2] Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0. This may lead to less accurate results!
[2026-02-26 05:24:18 TP2] Load weight end. elapsed=237.02 s, type=GlmMoeDsaForCausalLM, dtype=torch.bfloat16, avail mem=193.90 GB, mem usage=89.58 GB.
[2026-02-26 05:33:19 TP0] Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0. This may lead to less accurate results!
[2026-02-26 05:33:19 TP0] Load weight end. elapsed=778.18 s, type=GlmMoeDsaForCausalLM, dtype=torch.bfloat16, avail mem=193.92 GB, mem usage=89.58 GB.
[2026-02-26 05:35:08 TP7] Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0. This may lead to less accurate results!
[2026-02-26 05:35:08 TP7] Load weight end. elapsed=887.93 s, type=GlmMoeDsaForCausalLM, dtype=torch.bfloat16, avail mem=194.04 GB, mem usage=89.58 GB.
[2026-02-26 05:35:08 TP0] Using KV cache dtype: torch.float8_e4m3fn
[2026-02-26 05:35:08 TP7] KV Cache is allocated. #tokens: 3262823, KV size: 186.77 GB
[2026-02-26 05:35:08 TP6] KV Cache is allocated. #tokens: 3262823, KV size: 186.77 GB
[2026-02-26 05:35:08 TP0] KV Cache is allocated. #tokens: 3262823, KV size: 186.77 GB
[2026-02-26 05:35:08 TP7] Memory pool end. avail mem=4.02 GB
[2026-02-26 05:35:08 TP6] Memory pool end. avail mem=4.00 GB
[2026-02-26 05:35:08 TP0] Memory pool end. avail mem=3.90 GB
[2026-02-26 05:35:08 TP2] KV Cache is allocated. #tokens: 3262823, KV size: 186.77 GB
[2026-02-26 05:35:08 TP2] Memory pool end. avail mem=3.88 GB
[2026-02-26 05:35:08 TP4] KV Cache is allocated. #tokens: 3262823, KV size: 186.77 GB
[2026-02-26 05:35:08 TP4] Memory pool end. avail mem=4.02 GB
[2026-02-26 05:35:08 TP1] KV Cache is allocated. #tokens: 3262823, KV size: 186.77 GB
[2026-02-26 05:35:08 TP5] KV Cache is allocated. #tokens: 3262823, KV size: 186.77 GB
[2026-02-26 05:35:08 TP1] Memory pool end. avail mem=4.30 GB
[2026-02-26 05:35:08 TP5] Memory pool end. avail mem=3.94 GB
[2026-02-26 05:35:08 TP3] KV Cache is allocated. #tokens: 3262823, KV size: 186.77 GB
[2026-02-26 05:35:08 TP3] Memory pool end. avail mem=3.88 GB
[2026-02-26 05:35:09 TP5] Capture cuda graph begin. This can take up to several minutes. avail mem=3.72 GB
[2026-02-26 05:35:09 TP4] Capture cuda graph begin. This can take up to several minutes. avail mem=3.80 GB
[2026-02-26 05:35:09 TP6] Capture cuda graph begin. This can take up to several minutes. avail mem=3.78 GB
[2026-02-26 05:35:09 TP1] Capture cuda graph begin. This can take up to several minutes. avail mem=4.07 GB
[2026-02-26 05:35:09 TP0] Capture cuda graph begin. This can take up to several minutes. avail mem=3.67 GB
[2026-02-26 05:35:09 TP0] Capture cuda graph bs [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]
[2026-02-26 05:35:09 TP2] Capture cuda graph begin. This can take up to several minutes. avail mem=3.66 GB
[2026-02-26 05:35:09 TP3] Capture cuda graph begin. This can take up to several minutes. avail mem=3.65 GB
[2026-02-26 05:35:09 TP7] Capture cuda graph begin. This can take up to several minutes. avail mem=3.79 GB

  0%|          | 0/52 [00:00<?, ?it/s]
Capturing batches (bs=512 avail_mem=2.60 GB):   0%|          | 0/52 [00:00<?, ?it/s][aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 05:35:10 TP7] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 05:35:10 TP7] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[2026-02-26 05:35:10 TP7] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[aiter] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[2026-02-26 05:35:10 TP7] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 05:35:10 TP5] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 05:35:10 TP5] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 05:35:10 TP6] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 05:35:10 TP6] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 05:35:10 TP0] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 05:35:10 TP0] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 05:35:10 TP1] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 05:35:10 TP2] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 05:35:10 TP1] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 05:35:10 TP2] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 05:35:10 TP3] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 05:35:10 TP3] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[2026-02-26 05:35:10 TP5] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[aiter] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[2026-02-26 05:35:10 TP5] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 05:35:10 TP4] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[2026-02-26 05:35:10 TP6] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 05:35:10 TP4] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[2026-02-26 05:35:10 TP6] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[aiter] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[2026-02-26 05:35:10 TP0] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[aiter] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[2026-02-26 05:35:10 TP0] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[aiter] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[2026-02-26 05:35:10 TP1] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[aiter] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[2026-02-26 05:35:10 TP1] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[aiter] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[2026-02-26 05:35:10 TP2] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[aiter] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[2026-02-26 05:35:10 TP2] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[aiter] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[2026-02-26 05:35:10 TP3] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[aiter] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[2026-02-26 05:35:10 TP3] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[aiter] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[2026-02-26 05:35:10 TP4] import [module_quant] under /sgl-workspace/aiter/aiter/jit/module_quant.so
[aiter] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[2026-02-26 05:35:10 TP4] type hints mismatch, override to --> dynamic_per_token_scaled_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, scale_ub: Optional[torch.Tensor] = None, shuffle_scale: bool = False, num_rows: Optional[torch.Tensor] = None, num_rows_factor: int | typing.SupportsIndex = 1) -> None
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 05:35:12 TP5] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 05:35:12 TP4] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]

Capturing batches (bs=512 avail_mem=2.60 GB):   0%|          | 0/52 [00:03<?, ?it/s]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 05:35:12 TP0] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 05:35:13 TP3] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 05:35:13 TP1] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 05:35:13 TP2] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 05:35:13 TP7] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 05:35:13 TP6] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] Registering 0 cuda graph addresses
[2026-02-26 05:35:13 TP7] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 05:35:13 TP6] Registering 0 cuda graph addresses
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 05:35:13 TP7] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 05:35:13 TP1] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 05:35:13 TP3] Registering 0 cuda graph addresses
[2026-02-26 05:35:13 TP2] Registering 0 cuda graph addresses
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 05:35:13 TP6] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 05:35:13 TP0] Registering 0 cuda graph addresses
[2026-02-26 05:35:13 TP4] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 05:35:13 TP5] Registering 0 cuda graph addresses
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 05:35:13 TP1] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 05:35:13 TP3] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 05:35:13 TP2] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 05:35:13 TP4] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 05:35:13 TP0] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 05:35:13 TP5] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 05:35:13 TP7] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 3096, in run_scheduler_process
    scheduler = Scheduler(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 360, in __init__
    self.init_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 553, in init_model_worker
    self.init_tp_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 511, in init_tp_model_worker
    self.tp_worker = TpModelWorker(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 247, in __init__
    self._init_model_runner()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 330, in _init_model_runner
    self._model_runner = ModelRunner(
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 412, in __init__
    self.initialize(min_per_gpu_memory)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 608, in initialize
    self.init_device_graphs()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 2145, in init_device_graphs
    self.graph_runner = graph_runners[self.device](self)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 370, in __init__
    self.capture()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 526, in capture
    _capture_one_stream()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 513, in _capture_one_stream
    ) = self.capture_one_batch_size(bs, forward, stream_idx)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 732, in capture_one_batch_size
    run_once()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 719, in run_once
    logits_output_or_pp_proxy_tensors = forward(
  File "/opt/venv/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2919, in forward
    hidden_states = self.model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2730, in forward
    hidden_states, residual = layer(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2395, in forward
    hidden_states = self.self_attn(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1366, in forward
    s = self.forward_prepare(
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1424, in forward_prepare
    inner_state = self.forward_absorb_fused_mla_rope_prepare(
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2010, in forward_absorb_fused_mla_rope_prepare
    attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
TypeError: cannot unpack non-iterable ForwardMetadata object

[2026-02-26 05:35:13] Received sigquit from a child process. It usually means the child failed.
[2026-02-26 05:35:13 TP6] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 3096, in run_scheduler_process
    scheduler = Scheduler(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 360, in __init__
    self.init_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 553, in init_model_worker
    self.init_tp_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 511, in init_tp_model_worker
    self.tp_worker = TpModelWorker(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 247, in __init__
    self._init_model_runner()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 330, in _init_model_runner
    self._model_runner = ModelRunner(
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 412, in __init__
    self.initialize(min_per_gpu_memory)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 608, in initialize
    self.init_device_graphs()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 2145, in init_device_graphs
    self.graph_runner = graph_runners[self.device](self)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 370, in __init__
    self.capture()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 526, in capture
    _capture_one_stream()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 513, in _capture_one_stream
    ) = self.capture_one_batch_size(bs, forward, stream_idx)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 732, in capture_one_batch_size
    run_once()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 719, in run_once
    logits_output_or_pp_proxy_tensors = forward(
  File "/opt/venv/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2919, in forward
    hidden_states = self.model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2730, in forward
    hidden_states, residual = layer(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2395, in forward
    hidden_states = self.self_attn(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1366, in forward
    s = self.forward_prepare(
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1424, in forward_prepare
    inner_state = self.forward_absorb_fused_mla_rope_prepare(
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2010, in forward_absorb_fused_mla_rope_prepare
    attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
TypeError: cannot unpack non-iterable ForwardMetadata object

[2026-02-26 05:35:13] Received sigquit from a child process. It usually means the child failed.
[2026-02-26 05:35:13 TP1] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 3096, in run_scheduler_process
    scheduler = Scheduler(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 360, in __init__
    self.init_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 553, in init_model_worker
    self.init_tp_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 511, in init_tp_model_worker
    self.tp_worker = TpModelWorker(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 247, in __init__
    self._init_model_runner()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 330, in _init_model_runner
    self._model_runner = ModelRunner(
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 412, in __init__
    self.initialize(min_per_gpu_memory)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 608, in initialize
    self.init_device_graphs()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 2145, in init_device_graphs
    self.graph_runner = graph_runners[self.device](self)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 370, in __init__
    self.capture()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 526, in capture
    _capture_one_stream()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 513, in _capture_one_stream
    ) = self.capture_one_batch_size(bs, forward, stream_idx)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 732, in capture_one_batch_size
    run_once()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 719, in run_once
    logits_output_or_pp_proxy_tensors = forward(
  File "/opt/venv/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2919, in forward
    hidden_states = self.model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2730, in forward
    hidden_states, residual = layer(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2395, in forward
    hidden_states = self.self_attn(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1366, in forward
    s = self.forward_prepare(
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1424, in forward_prepare
    inner_state = self.forward_absorb_fused_mla_rope_prepare(
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2010, in forward_absorb_fused_mla_rope_prepare
    attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
TypeError: cannot unpack non-iterable ForwardMetadata object

[2026-02-26 05:35:13] Received sigquit from a child process. It usually means the child failed.
[2026-02-26 05:35:13 TP3] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 3096, in run_scheduler_process
    scheduler = Scheduler(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 360, in __init__
    self.init_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 553, in init_model_worker
    self.init_tp_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 511, in init_tp_model_worker
    self.tp_worker = TpModelWorker(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 247, in __init__
    self._init_model_runner()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 330, in _init_model_runner
    self._model_runner = ModelRunner(
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 412, in __init__
    self.initialize(min_per_gpu_memory)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 608, in initialize
    self.init_device_graphs()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 2145, in init_device_graphs
    self.graph_runner = graph_runners[self.device](self)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 370, in __init__
    self.capture()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 526, in capture
    _capture_one_stream()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 513, in _capture_one_stream
    ) = self.capture_one_batch_size(bs, forward, stream_idx)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 732, in capture_one_batch_size
    run_once()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 719, in run_once
    logits_output_or_pp_proxy_tensors = forward(
  File "/opt/venv/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2919, in forward
    hidden_states = self.model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2730, in forward
    hidden_states, residual = layer(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2395, in forward
    hidden_states = self.self_attn(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1366, in forward
    s = self.forward_prepare(
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1424, in forward_prepare
    inner_state = self.forward_absorb_fused_mla_rope_prepare(
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2010, in forward_absorb_fused_mla_rope_prepare
    attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
TypeError: cannot unpack non-iterable ForwardMetadata object

[2026-02-26 05:35:13] Received sigquit from a child process. It usually means the child failed.
[2026-02-26 05:35:13 TP4] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 3096, in run_scheduler_process
    scheduler = Scheduler(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 360, in __init__
    self.init_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 553, in init_model_worker
    self.init_tp_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 511, in init_tp_model_worker
    self.tp_worker = TpModelWorker(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 247, in __init__
    self._init_model_runner()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 330, in _init_model_runner
    self._model_runner = ModelRunner(
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 412, in __init__
    self.initialize(min_per_gpu_memory)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 608, in initialize
    self.init_device_graphs()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 2145, in init_device_graphs
    self.graph_runner = graph_runners[self.device](self)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 370, in __init__
    self.capture()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 526, in capture
    _capture_one_stream()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 513, in _capture_one_stream
    ) = self.capture_one_batch_size(bs, forward, stream_idx)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 732, in capture_one_batch_size
    run_once()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 719, in run_once
    logits_output_or_pp_proxy_tensors = forward(
  File "/opt/venv/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2919, in forward
    hidden_states = self.model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2730, in forward
    hidden_states, residual = layer(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2395, in forward
    hidden_states = self.self_attn(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1366, in forward
    s = self.forward_prepare(
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1424, in forward_prepare
    inner_state = self.forward_absorb_fused_mla_rope_prepare(
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2010, in forward_absorb_fused_mla_rope_prepare
    attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
TypeError: cannot unpack non-iterable ForwardMetadata object

[2026-02-26 05:35:13] Received sigquit from a child process. It usually means the child failed.
[2026-02-26 05:35:13 TP0] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 3096, in run_scheduler_process
    scheduler = Scheduler(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 360, in __init__
    self.init_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 553, in init_model_worker
    self.init_tp_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 511, in init_tp_model_worker
    self.tp_worker = TpModelWorker(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 247, in __init__
    self._init_model_runner()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 330, in _init_model_runner
    self._model_runner = ModelRunner(
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 412, in __init__
    self.initialize(min_per_gpu_memory)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 608, in initialize
    self.init_device_graphs()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 2145, in init_device_graphs
    self.graph_runner = graph_runners[self.device](self)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 370, in __init__
    self.capture()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 526, in capture
    _capture_one_stream()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 513, in _capture_one_stream
    ) = self.capture_one_batch_size(bs, forward, stream_idx)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 732, in capture_one_batch_size
    run_once()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 719, in run_once
    logits_output_or_pp_proxy_tensors = forward(
  File "/opt/venv/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2919, in forward
    hidden_states = self.model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2730, in forward
    hidden_states, residual = layer(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2395, in forward
    hidden_states = self.self_attn(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1366, in forward
    s = self.forward_prepare(
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1424, in forward_prepare
    inner_state = self.forward_absorb_fused_mla_rope_prepare(
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2010, in forward_absorb_fused_mla_rope_prepare
    attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
TypeError: cannot unpack non-iterable ForwardMetadata object

[2026-02-26 05:35:13] Received sigquit from a child process. It usually means the child failed.
[2026-02-26 05:35:13 TP5] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 3096, in run_scheduler_process
    scheduler = Scheduler(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 360, in __init__
    self.init_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 553, in init_model_worker
    self.init_tp_model_worker()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 511, in init_tp_model_worker
    self.tp_worker = TpModelWorker(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 247, in __init__
    self._init_model_runner()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py", line 330, in _init_model_runner
    self._model_runner = ModelRunner(
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 412, in __init__
    self.initialize(min_per_gpu_memory)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 608, in initialize
    self.init_device_graphs()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 2145, in init_device_graphs
    self.graph_runner = graph_runners[self.device](self)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 370, in __init__
    self.capture()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 526, in capture
    _capture_one_stream()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 513, in _capture_one_stream
    ) = self.capture_one_batch_size(bs, forward, stream_idx)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 732, in capture_one_batch_size
    run_once()
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 719, in run_once
    logits_output_or_pp_proxy_tensors = forward(
  File "/opt/venv/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2919, in forward
    hidden_states = self.model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2730, in forward
    hidden_states, residual = layer(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2395, in forward
    hidden_states = self.self_attn(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1366, in forward
    s = self.forward_prepare(
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 1424, in forward_prepare
    inner_state = self.forward_absorb_fused_mla_rope_prepare(
  File "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py", line 2010, in forward_absorb_fused_mla_rope_prepare
    attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
TypeError: cannot unpack non-iterable ForwardMetadata object

[2026-02-26 05:35:13] Received sigquit from a child process. It usually means the child failed.
```
</details>

### Trail 2 : 

#### Docker Image : rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260225
#### Transformers : 4.57.1

#### Command : 
```
SGLANG_USE_AITER=0
python3 -m sglang.launch_server --attention-backend triton --model-path zai-org/GLM-5-FP8 --tp-size 8 --kv-cache-dtype fp8_e4m3
```

#### Error Log : 

<details>
<summary>Click to view Error Log</summary>
  
```bash
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 05:22:34] INFO core.py:501: import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
Traceback (most recent call last):
  File "/opt/venv/lib/python3.10/site-packages/transformers/models/auto/configuration_auto.py", line 1360, in from_pretrained
    config_class = CONFIG_MAPPING[config_dict["model_type"]]
  File "/opt/venv/lib/python3.10/site-packages/transformers/models/auto/configuration_auto.py", line 1048, in __getitem__
    raise KeyError(key)
KeyError: 'glm_moe_dsa'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/sgl-workspace/sglang/python/sglang/launch_server.py", line 32, in <module>
    server_args = prepare_server_args(sys.argv[1:])
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 5679, in prepare_server_args
    return ServerArgs.from_cli_args(raw_args)
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 5165, in from_cli_args
    return cls(**{attr: getattr(args, attr) for attr in attrs})
  File "<string>", line 336, in __init__
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 739, in __post_init__
    self._handle_gpu_memory_settings(gpu_mem)
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 1004, in _handle_gpu_memory_settings
    if not self.use_mla_backend():
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 5198, in use_mla_backend
    model_config = self.get_model_config()
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 5179, in get_model_config
    self.model_config = ModelConfig.from_server_args(self)
  File "/sgl-workspace/sglang/python/sglang/srt/configs/model_config.py", line 250, in from_server_args
    return ModelConfig(
  File "/sgl-workspace/sglang/python/sglang/srt/configs/model_config.py", line 127, in __init__
    self.hf_config = get_config(
  File "/sgl-workspace/sglang/python/sglang/srt/utils/common.py", line 3475, in wrapper
    result = func(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/utils/hf_transformers_utils.py", line 320, in get_config
    raise e
  File "/sgl-workspace/sglang/python/sglang/srt/utils/hf_transformers_utils.py", line 315, in get_config
    config = AutoConfig.from_pretrained(
  File "/opt/venv/lib/python3.10/site-packages/transformers/models/auto/configuration_auto.py", line 1362, in from_pretrained
    raise ValueError(
ValueError: The checkpoint you are trying to load has model type `glm_moe_dsa` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

You can update Transformers with the command `pip install --upgrade transformers`. If this does not work, and the checkpoint is very new, then there may not be a release version that supports this model yet. In this case, you can get the most up-to-date code by installing Transformers from source with the command `pip install git+https://github.com/huggingface/transformers.git`

```
</details>

#### Resolution : 
```
pip install git+https://github.com/huggingface/transformers.git
```
#### Output : 
```
Successfully installed huggingface-hub-1.4.1 markdown-it-py-4.0.0 mdurl-0.1.2 rich-14.3.3 shellingham-1.5.4 transformers-5.3.0.dev0 typer-0.24.1 typer-slim-0.24.0
```
#### Command : 
```
python3 -m sglang.launch_server --attention-backend triton --model-path zai-org/GLM-5-FP8 --tp-size 8 --kv-cache-dtype fp8_e4m3
```

#### Error Log : 

<details>
<summary>Click to view Error Log</summary>
  
```bash
```
</details>
