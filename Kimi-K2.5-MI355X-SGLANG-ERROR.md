# Model : moonshotai/Kimi-K2.5

## Rocm SGLang Docker (20260218)  

##### Docker Image : rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260218
##### Transformers : 4.57.1

##### Command : 
```
sglang serve --model-path moonshotai/Kimi-K2.5 --tp 8 --trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2
```

##### Error Log : 

<details>
<summary>Click to view Error Log</summary>
  
```bash
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
A new version of the following files was downloaded from https://huggingface.co/moonshotai/Kimi-K2.5:
- configuration_deepseek.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
[2026-02-26 16:30:34] server_args=ServerArgs(model_path='moonshotai/Kimi-K2.5', tokenizer_path='moonshotai/Kimi-K2.5', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=True, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.8260405859375, max_running_requests=None, max_queued_requests=None, max_total_tokens=None, chunked_prefill_size=16384, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=8, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, random_seed=603657908, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=None, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='info', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='moonshotai/Kimi-K2.5', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser='kimi_k2', tool_call_parser='kimi_k2', tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='triton', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='pytorch', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method='AMXINT4', kt_cpuinfer=None, kt_threadpool_count=2, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=512, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, enable_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=2048, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=16, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_decode_tp=None, disaggregation_decode_dp=None, disaggregation_prefill_pp=1, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)
/opt/venv/lib/python3.10/site-packages/apex/transformer/functional/fused_rope.py:49: UserWarning: Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0
  warnings.warn("Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0", UserWarning)
[2026-02-26 16:30:34] Ignore import error when loading sglang.srt.multimodal.processors.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
A new version of the following files was downloaded from https://huggingface.co/moonshotai/Kimi-K2.5:
- media_utils.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
A new version of the following files was downloaded from https://huggingface.co/moonshotai/Kimi-K2.5:
- tool_declaration_ts.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
[2026-02-26 16:30:36] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:30:36] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:30:36] Using default HuggingFace chat template with detected content format: openai
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 16:30:40 TP7] Process 249 gpu_id 7 is running on CPUs: [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
[2026-02-26 16:30:40 TP6] Process 248 gpu_id 6 is running on CPUs: [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
[2026-02-26 16:30:40 TP3] Process 245 gpu_id 3 is running on CPUs: [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
[2026-02-26 16:30:40 TP2] Process 244 gpu_id 2 is running on CPUs: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
[2026-02-26 16:30:40 TP4] Process 246 gpu_id 4 is running on CPUs: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
[2026-02-26 16:30:40 TP5] Process 247 gpu_id 5 is running on CPUs: [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
[2026-02-26 16:30:40 TP0] Process 242 gpu_id 0 is running on CPUs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
[2026-02-26 16:30:40 TP1] Process 243 gpu_id 1 is running on CPUs: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
[2026-02-26 16:30:41] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:30:41] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
/sgl-workspace/sglang/python/sglang/srt/utils/hf_transformers_utils.py:558: UserWarning: Using a slow tokenizer. This might cause a significant slowdown. Consider using a fast tokenizer instead.
  warnings.warn(
[2026-02-26 16:30:41 TP7] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:30:41 TP7] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:30:41 TP6] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:30:41 TP6] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:30:41 TP2] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:30:41 TP2] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:30:41 TP4] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:30:41 TP4] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:30:41 TP3] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:30:41 TP3] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:30:41 TP5] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:30:41 TP5] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:30:42 TP0] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:30:42 TP0] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:30:42 TP7] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:30:42 TP7] Init torch distributed begin.
[2026-02-26 16:30:42 TP7] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:30:42 TP1] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:30:42 TP1] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:30:42 TP6] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:30:42 TP6] Init torch distributed begin.
[2026-02-26 16:30:42 TP6] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:30:42 TP4] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:30:42 TP4] Init torch distributed begin.
[2026-02-26 16:30:42 TP4] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:30:42 TP2] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:30:42 TP2] Init torch distributed begin.
[2026-02-26 16:30:42 TP2] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:30:42 TP3] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:30:42 TP3] Init torch distributed begin.
[2026-02-26 16:30:42 TP3] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:30:42 TP5] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:30:42 TP5] Init torch distributed begin.
[2026-02-26 16:30:42 TP5] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:30:42 TP0] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:30:42 TP0] Init torch distributed begin.
[2026-02-26 16:30:42 TP0] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:30:42 TP1] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:30:42 TP1] Init torch distributed begin.
[2026-02-26 16:30:42 TP1] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[Gloo] Rank 0 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 1 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 4 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 5 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 6 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 2 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 3 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 7 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[2026-02-26 16:30:43 TP5] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:30:43 TP1] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:30:43 TP3] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:30:43 TP6] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:30:43 TP4] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:30:43 TP0] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:30:43 TP2] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:30:43 TP7] [AR] All-reduce call path: NCCL (custom AR disabled)
[Gloo] Rank 0 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 1 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 5 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 2 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[2026-02-26 16:30:43 TP0] sglang is using nccl==2.27.7
[Gloo] Rank 4 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 3 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 6 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 7 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:30:59 TP1] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:30:59 TP7] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:30:59 TP0] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:30:59 TP1] [AR] Using AiterCustomAllreduce (AMD default)
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:30:59 TP6] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:30:59 TP7] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:30:59 TP2] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:30:59 TP4] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:30:59 TP3] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:30:59 TP0] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:30:59 TP6] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:30:59 TP4] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:30:59 TP3] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:30:59 TP2] [AR] Using AiterCustomAllreduce (AMD default)
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:30:59 TP5] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:30:59 TP5] [AR] Using AiterCustomAllreduce (AMD default)
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:30:59 TP0] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:30:59 TP1] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:30:59 TP2] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:30:59 TP3] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:30:59 TP7] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:30:59 TP4] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:30:59 TP5] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:30:59 TP6] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:30:59 TP7] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:30:59 TP3] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:30:59 TP0] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:30:59 TP5] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:30:59 TP1] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:30:59 TP4] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:30:59 TP2] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:30:59 TP6] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP7] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP3] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP5] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP0] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP1] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP4] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP2] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP6] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP7] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP5] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP0] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP1] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP3] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP4] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP2] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP6] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:30:59 TP5] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:30:59 TP2] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:30:59 TP3] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:30:59 TP4] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:30:59 TP1] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:30:59 TP0] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:30:59 TP6] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:30:59 TP7] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
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
[2026-02-26 16:30:59 TP0] Init torch distributed ends. elapsed=17.24 s, mem usage=3.81 GB
[2026-02-26 16:30:59 TP7] Init torch distributed ends. elapsed=17.82 s, mem usage=3.69 GB
[2026-02-26 16:30:59 TP6] Init torch distributed ends. elapsed=17.69 s, mem usage=3.70 GB
[2026-02-26 16:30:59 TP5] Init torch distributed ends. elapsed=17.31 s, mem usage=3.76 GB
[2026-02-26 16:30:59 TP4] Init torch distributed ends. elapsed=17.52 s, mem usage=3.69 GB
[2026-02-26 16:30:59 TP3] Init torch distributed ends. elapsed=17.44 s, mem usage=3.83 GB
[2026-02-26 16:30:59 TP2] Init torch distributed ends. elapsed=17.50 s, mem usage=3.83 GB
[2026-02-26 16:30:59 TP1] Init torch distributed ends. elapsed=17.21 s, mem usage=3.41 GB
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
[2026-02-26 16:31:00 TP2] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP2] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP2] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:31:00 TP1] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP1] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP1] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:31:00 TP0] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP0] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP0] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:31:00 TP6] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP6] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP6] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:31:00 TP5] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP5] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP5] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:31:00 TP4] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP4] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP4] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:31:00 TP3] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP3] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP3] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:31:00 TP7] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP7] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:31:00 TP7] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:31:00 TP7] Load weight begin. avail mem=283.68 GB
[2026-02-26 16:31:00 TP1] Load weight begin. avail mem=283.96 GB
[2026-02-26 16:31:00 TP3] Load weight begin. avail mem=283.53 GB
[2026-02-26 16:31:00 TP2] Load weight begin. avail mem=283.54 GB
[2026-02-26 16:31:00 TP5] Load weight begin. avail mem=283.60 GB
[2026-02-26 16:31:00 TP6] Load weight begin. avail mem=283.66 GB
[2026-02-26 16:31:00 TP0] Load weight begin. avail mem=283.56 GB
[2026-02-26 16:31:00 TP4] Load weight begin. avail mem=283.68 GB
[2026-02-26 16:31:00 TP6] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:31:00 TP6] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:31:00 TP1] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:31:00 TP1] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:31:00 TP7] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:31:00 TP7] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:31:00 TP5] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:31:00 TP5] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:31:00 TP0] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:31:00 TP0] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:31:00 TP4] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:31:00 TP4] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:31:00 TP6] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:31:00 TP2] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:31:00 TP2] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:31:00 TP1] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:31:00 TP7] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:31:00 TP3] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:31:00 TP3] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:31:00 TP5] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:31:00 TP0] Config does not support fused shared expert(s). Shared experts fusion optimization is disabled.
[2026-02-26 16:31:00 TP0] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:31:00 TP4] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:31:00 TP2] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:31:00 TP3] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:31:01 TP6] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:31:01 TP1] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:31:01 TP7] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:31:01 TP3] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:31:01 TP5] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:31:01 TP4] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:31:01 TP0] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:31:01 TP2] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:31:02 TP0] Found local HF snapshot for moonshotai/Kimi-K2.5 at /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1; skipping download.

Loading safetensors checkpoint shards:   0% Completed | 0/64 [00:00<?, ?it/s]

Loading safetensors checkpoint shards:   8% Completed | 5/64 [00:00<00:01, 39.83it/s]

Loading safetensors checkpoint shards:  14% Completed | 9/64 [00:00<00:04, 11.39it/s]

Loading safetensors checkpoint shards:  20% Completed | 13/64 [00:00<00:03, 15.80it/s]

Loading safetensors checkpoint shards:  27% Completed | 17/64 [00:00<00:02, 19.51it/s]

Loading safetensors checkpoint shards:  33% Completed | 21/64 [00:01<00:01, 22.38it/s]

Loading safetensors checkpoint shards:  39% Completed | 25/64 [00:01<00:03, 12.59it/s]

Loading safetensors checkpoint shards:  39% Completed | 25/64 [00:20<00:03, 12.59it/s]

Loading safetensors checkpoint shards:  44% Completed | 28/64 [00:31<01:37,  2.72s/it]

Loading safetensors checkpoint shards:  45% Completed | 29/64 [00:47<02:27,  4.21s/it]

Loading safetensors checkpoint shards:  47% Completed | 30/64 [01:03<03:15,  5.75s/it]

Loading safetensors checkpoint shards:  48% Completed | 31/64 [01:18<04:01,  7.32s/it]

Loading safetensors checkpoint shards:  50% Completed | 32/64 [01:34<04:47,  8.97s/it]

Loading safetensors checkpoint shards:  52% Completed | 33/64 [01:50<05:24, 10.47s/it]

Loading safetensors checkpoint shards:  53% Completed | 34/64 [02:07<05:56, 11.87s/it]

Loading safetensors checkpoint shards:  55% Completed | 35/64 [02:23<06:20, 13.11s/it]

Loading safetensors checkpoint shards:  56% Completed | 36/64 [02:40<06:34, 14.09s/it]

Loading safetensors checkpoint shards:  58% Completed | 37/64 [02:57<06:42, 14.92s/it]

Loading safetensors checkpoint shards:  59% Completed | 38/64 [03:14<06:42, 15.48s/it]

Loading safetensors checkpoint shards:  61% Completed | 39/64 [03:32<06:41, 16.07s/it]

Loading safetensors checkpoint shards:  62% Completed | 40/64 [03:49<06:35, 16.50s/it]

Loading safetensors checkpoint shards:  64% Completed | 41/64 [04:07<06:26, 16.82s/it]

Loading safetensors checkpoint shards:  66% Completed | 42/64 [04:23<06:06, 16.64s/it]

Loading safetensors checkpoint shards:  67% Completed | 43/64 [04:40<05:50, 16.71s/it]

Loading safetensors checkpoint shards:  69% Completed | 44/64 [04:56<05:30, 16.55s/it]

Loading safetensors checkpoint shards:  70% Completed | 45/64 [05:13<05:16, 16.66s/it]

Loading safetensors checkpoint shards:  72% Completed | 46/64 [05:30<05:01, 16.72s/it]

Loading safetensors checkpoint shards:  73% Completed | 47/64 [05:46<04:39, 16.44s/it]

Loading safetensors checkpoint shards:  75% Completed | 48/64 [06:01<04:18, 16.13s/it]

Loading safetensors checkpoint shards:  77% Completed | 49/64 [06:17<04:02, 16.15s/it]

Loading safetensors checkpoint shards:  78% Completed | 50/64 [06:32<03:37, 15.56s/it]

Loading safetensors checkpoint shards:  80% Completed | 51/64 [06:46<03:16, 15.12s/it]

Loading safetensors checkpoint shards:  81% Completed | 52/64 [06:59<02:56, 14.73s/it]

Loading safetensors checkpoint shards:  83% Completed | 53/64 [07:14<02:41, 14.71s/it]

Loading safetensors checkpoint shards:  84% Completed | 54/64 [07:29<02:28, 14.82s/it]

Loading safetensors checkpoint shards:  86% Completed | 55/64 [07:45<02:15, 15.05s/it]

Loading safetensors checkpoint shards:  89% Completed | 57/64 [08:00<01:20, 11.56s/it]

Loading safetensors checkpoint shards:  91% Completed | 58/64 [08:01<00:53,  8.99s/it]

Loading safetensors checkpoint shards:  94% Completed | 60/64 [08:15<00:32,  8.23s/it]

Loading safetensors checkpoint shards:  95% Completed | 61/64 [08:29<00:28,  9.54s/it]

Loading safetensors checkpoint shards:  97% Completed | 62/64 [08:44<00:21, 10.78s/it]

Loading safetensors checkpoint shards:  98% Completed | 63/64 [08:58<00:11, 11.76s/it]

Loading safetensors checkpoint shards: 100% Completed | 64/64 [09:14<00:00, 12.77s/it]

Loading safetensors checkpoint shards: 100% Completed | 64/64 [09:14<00:00,  8.66s/it]

[2026-02-26 16:41:05 TP1] Load weight end. elapsed=605.50 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.38 GB, mem usage=73.58 GB.
[2026-02-26 16:41:06 TP3] Load weight end. elapsed=606.34 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=209.96 GB, mem usage=73.58 GB.
[2026-02-26 16:41:06 TP6] Load weight end. elapsed=606.43 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.08 GB, mem usage=73.58 GB.
[2026-02-26 16:41:06 TP5] Load weight end. elapsed=606.46 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.02 GB, mem usage=73.58 GB.
[2026-02-26 16:41:06 TP2] Load weight end. elapsed=606.47 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=209.96 GB, mem usage=73.58 GB.
[2026-02-26 16:41:09 TP7] Load weight end. elapsed=609.07 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.10 GB, mem usage=73.58 GB.
[2026-02-26 16:41:14 TP4] Load weight end. elapsed=614.45 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.10 GB, mem usage=73.58 GB.
[2026-02-26 16:41:18 TP0] Load weight end. elapsed=618.04 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=209.98 GB, mem usage=73.58 GB.
[2026-02-26 16:41:18 TP0] Using KV cache dtype: torch.bfloat16
[2026-02-26 16:41:18 TP6] KV Cache is allocated. #tokens: 2454421, KV size: 160.63 GB
[2026-02-26 16:41:18 TP6] Memory pool end. avail mem=45.36 GB
[2026-02-26 16:41:18 TP2] KV Cache is allocated. #tokens: 2454421, KV size: 160.63 GB
[2026-02-26 16:41:18 TP2] Memory pool end. avail mem=45.24 GB
[2026-02-26 16:41:18 TP5] KV Cache is allocated. #tokens: 2454421, KV size: 160.63 GB
[2026-02-26 16:41:18 TP5] Memory pool end. avail mem=45.30 GB
[2026-02-26 16:41:18 TP0] KV Cache is allocated. #tokens: 2454421, KV size: 160.63 GB
[2026-02-26 16:41:18 TP0] Memory pool end. avail mem=45.25 GB
[2026-02-26 16:41:18 TP4] KV Cache is allocated. #tokens: 2454421, KV size: 160.63 GB
[2026-02-26 16:41:18 TP4] Memory pool end. avail mem=45.38 GB
[2026-02-26 16:41:18 TP3] KV Cache is allocated. #tokens: 2454421, KV size: 160.63 GB
[2026-02-26 16:41:18 TP3] Memory pool end. avail mem=45.23 GB
[2026-02-26 16:41:18 TP7] KV Cache is allocated. #tokens: 2454421, KV size: 160.63 GB
[2026-02-26 16:41:18 TP7] Memory pool end. avail mem=45.37 GB
[2026-02-26 16:41:18 TP1] KV Cache is allocated. #tokens: 2454421, KV size: 160.63 GB
[2026-02-26 16:41:18 TP1] Memory pool end. avail mem=45.65 GB
[2026-02-26 16:41:19 TP2] Capture cuda graph begin. This can take up to several minutes. avail mem=45.01 GB
[2026-02-26 16:41:19 TP4] Capture cuda graph begin. This can take up to several minutes. avail mem=45.15 GB
[2026-02-26 16:41:19 TP1] Capture cuda graph begin. This can take up to several minutes. avail mem=45.43 GB
[2026-02-26 16:41:19 TP0] Capture cuda graph begin. This can take up to several minutes. avail mem=45.03 GB
[2026-02-26 16:41:19 TP0] Capture cuda graph bs [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]
[2026-02-26 16:41:19 TP6] Capture cuda graph begin. This can take up to several minutes. avail mem=45.14 GB
[2026-02-26 16:41:19 TP3] Capture cuda graph begin. This can take up to several minutes. avail mem=45.01 GB
[2026-02-26 16:41:19 TP7] Capture cuda graph begin. This can take up to several minutes. avail mem=45.15 GB
[2026-02-26 16:41:19 TP5] Capture cuda graph begin. This can take up to several minutes. avail mem=45.07 GB

  0%|          | 0/52 [00:00<?, ?it/s]
Capturing batches (bs=512 avail_mem=43.46 GB):   0%|          | 0/52 [00:00<?, ?it/s][aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:41:21 TP1] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:41:21 TP1] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:41:21 TP2] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:41:21 TP2] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:41:21 TP4] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:41:21 TP4] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:41:21 TP7] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:41:21 TP7] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:41:21 TP5] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:41:21 TP5] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:41:21 TP6] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:41:21 TP6] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:41:21 TP3] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:41:21 TP3] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:41:21 TP0] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:41:21 TP0] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:41:23 TP1] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:41:23 TP2] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:41:23 TP4] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:41:24 TP3] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:41:24 TP7] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:41:24 TP5] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:41:24 TP6] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]

Capturing batches (bs=512 avail_mem=43.46 GB):   0%|          | 0/52 [00:04<?, ?it/s]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:41:24 TP0] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:41:24 TP7] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:41:24 TP0] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:41:24 TP5] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:41:24 TP6] Registering 0 cuda graph addresses
[2026-02-26 16:41:24 TP3] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:41:24 TP4] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:41:24 TP7] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:41:24 TP1] Registering 0 cuda graph addresses
[2026-02-26 16:41:24 TP2] Registering 0 cuda graph addresses
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:41:24 TP0] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:41:24 TP5] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:41:24 TP6] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:41:24 TP3] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:41:24 TP4] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:41:24 TP1] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:41:24 TP2] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:41:24 TP6] Scheduler hit an exception: Traceback (most recent call last):
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
  File "/sgl-workspace/sglang/python/sglang/srt/models/kimi_k25.py", line 726, in forward
    hidden_states = general_mm_embed_routine(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py", line 1134, in general_mm_embed_routine
    hidden_states = language_model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
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

[2026-02-26 16:41:24] Received sigquit from a child process. It usually means the child failed.

```
</details>

##### Command ( With Triton Backend ) : 
```
sglang serve --model-path moonshotai/Kimi-K2.5 --attention-backend triton --tp 8 --trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2
```

##### Error Log : 

<details>
<summary>Click to view Error Log</summary>
  
```bash
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 16:49:51] server_args=ServerArgs(model_path='moonshotai/Kimi-K2.5', tokenizer_path='moonshotai/Kimi-K2.5', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=True, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.8260405859375, max_running_requests=None, max_queued_requests=None, max_total_tokens=None, chunked_prefill_size=16384, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=8, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, random_seed=503909685, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=None, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='info', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='moonshotai/Kimi-K2.5', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser='kimi_k2', tool_call_parser='kimi_k2', tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='triton', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='pytorch', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method='AMXINT4', kt_cpuinfer=None, kt_threadpool_count=2, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=512, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, enable_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=2048, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=16, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_decode_tp=None, disaggregation_decode_dp=None, disaggregation_prefill_pp=1, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)
/opt/venv/lib/python3.10/site-packages/apex/transformer/functional/fused_rope.py:49: UserWarning: Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0
  warnings.warn("Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0", UserWarning)
[2026-02-26 16:49:51] Ignore import error when loading sglang.srt.multimodal.processors.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:49:52] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:49:52] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:49:53] Using default HuggingFace chat template with detected content format: openai
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 16:49:57 TP4] Process 5950 gpu_id 4 is running on CPUs: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
[2026-02-26 16:49:57 TP3] Process 5949 gpu_id 3 is running on CPUs: [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
[2026-02-26 16:49:57 TP0] Process 5946 gpu_id 0 is running on CPUs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
[2026-02-26 16:49:57 TP7] Process 5953 gpu_id 7 is running on CPUs: [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
[2026-02-26 16:49:57 TP6] Process 5952 gpu_id 6 is running on CPUs: [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
[2026-02-26 16:49:57 TP5] Process 5951 gpu_id 5 is running on CPUs: [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
[2026-02-26 16:49:57 TP1] Process 5947 gpu_id 1 is running on CPUs: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
[2026-02-26 16:49:57 TP2] Process 5948 gpu_id 2 is running on CPUs: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
[2026-02-26 16:49:58] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:49:58] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
/sgl-workspace/sglang/python/sglang/srt/utils/hf_transformers_utils.py:558: UserWarning: Using a slow tokenizer. This might cause a significant slowdown. Consider using a fast tokenizer instead.
  warnings.warn(
[2026-02-26 16:49:58 TP3] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:49:58 TP3] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:49:58 TP4] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:49:58 TP4] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:49:58 TP0] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:49:58 TP0] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:49:58 TP5] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:49:58 TP5] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:49:59 TP2] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:49:59 TP2] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:49:59 TP6] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:49:59 TP6] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:49:59 TP1] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:49:59 TP1] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:49:59 TP7] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:49:59 TP7] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:49:59 TP3] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:49:59 TP3] Init torch distributed begin.
[2026-02-26 16:49:59 TP3] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:49:59 TP0] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:49:59 TP0] Init torch distributed begin.
[2026-02-26 16:49:59 TP0] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:49:59 TP4] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:49:59 TP4] Init torch distributed begin.
[2026-02-26 16:49:59 TP4] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:49:59 TP5] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:49:59 TP5] Init torch distributed begin.
[2026-02-26 16:49:59 TP5] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:49:59 TP2] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:49:59 TP6] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:49:59 TP2] Init torch distributed begin.
[2026-02-26 16:49:59 TP2] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:49:59 TP6] Init torch distributed begin.
[2026-02-26 16:49:59 TP6] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:49:59 TP1] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:49:59 TP7] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:49:59 TP1] Init torch distributed begin.
[2026-02-26 16:49:59 TP1] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:49:59 TP7] Init torch distributed begin.
[2026-02-26 16:49:59 TP7] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[Gloo] Rank 3 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 0 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 1 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 5 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 4 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 2 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 7 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 6 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[2026-02-26 16:50:00 TP3] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:50:00 TP1] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:50:00 TP0] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:50:00 TP2] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:50:00 TP6] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:50:00 TP5] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:50:00 TP7] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:50:00 TP4] [AR] All-reduce call path: NCCL (custom AR disabled)
[Gloo] Rank 4 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 0 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 1 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 2 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 3 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 5 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 7 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 6 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[2026-02-26 16:50:00 TP0] sglang is using nccl==2.27.7
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:50:12 TP6] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:50:12 TP7] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:50:12 TP1] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:50:12 TP6] [AR] Using AiterCustomAllreduce (AMD default)
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:50:12 TP3] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:50:12 TP2] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:50:12 TP0] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:50:12 TP4] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:50:12 TP7] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:50:12 TP1] [AR] Using AiterCustomAllreduce (AMD default)
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:50:12 TP2] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:50:12 TP3] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:50:12 TP5] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:50:12 TP0] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:50:12 TP4] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:50:12 TP5] [AR] Using AiterCustomAllreduce (AMD default)
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:50:12 TP1] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:50:12 TP7] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:50:12 TP4] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:50:12 TP2] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:50:12 TP5] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:50:12 TP3] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:50:12 TP0] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:50:12 TP6] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:50:12 TP0] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:50:12 TP7] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:50:12 TP1] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:50:12 TP3] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:50:12 TP2] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:50:12 TP4] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:50:12 TP6] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:50:12 TP5] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP1] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP7] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP3] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP5] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP2] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP0] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP6] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP4] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP7] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP1] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP3] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP0] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP2] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP4] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP5] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP6] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:50:12 TP4] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:50:12 TP2] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:50:12 TP1] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:50:12 TP3] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:50:12 TP0] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:50:12 TP6] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:50:12 TP5] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:50:12 TP7] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
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
[2026-02-26 16:50:12 TP0] Init torch distributed ends. elapsed=12.89 s, mem usage=3.81 GB
[2026-02-26 16:50:12 TP7] Init torch distributed ends. elapsed=12.63 s, mem usage=3.69 GB
[2026-02-26 16:50:12 TP6] Init torch distributed ends. elapsed=12.71 s, mem usage=3.70 GB
[2026-02-26 16:50:12 TP5] Init torch distributed ends. elapsed=12.79 s, mem usage=3.76 GB
[2026-02-26 16:50:12 TP4] Init torch distributed ends. elapsed=12.85 s, mem usage=3.69 GB
[2026-02-26 16:50:12 TP3] Init torch distributed ends. elapsed=12.95 s, mem usage=3.83 GB
[2026-02-26 16:50:12 TP2] Init torch distributed ends. elapsed=12.72 s, mem usage=3.83 GB
[2026-02-26 16:50:12 TP1] Init torch distributed ends. elapsed=12.64 s, mem usage=3.41 GB
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
[2026-02-26 16:50:12 TP3] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP3] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP3] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:50:12 TP4] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP4] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP4] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:50:12 TP0] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP0] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP0] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:50:12 TP1] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP7] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP1] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP7] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP1] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:50:12 TP7] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:50:12 TP2] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP2] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP2] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:50:12 TP6] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP6] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP6] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:50:12 TP5] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP5] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:50:12 TP5] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:50:12 TP3] Load weight begin. avail mem=283.53 GB
[2026-02-26 16:50:12 TP4] Load weight begin. avail mem=283.68 GB
[2026-02-26 16:50:12 TP0] Load weight begin. avail mem=283.56 GB
[2026-02-26 16:50:12 TP1] Load weight begin. avail mem=283.96 GB
[2026-02-26 16:50:12 TP7] Load weight begin. avail mem=283.68 GB
[2026-02-26 16:50:12 TP2] Load weight begin. avail mem=283.54 GB
[2026-02-26 16:50:12 TP6] Load weight begin. avail mem=283.66 GB
[2026-02-26 16:50:12 TP5] Load weight begin. avail mem=283.60 GB
[2026-02-26 16:50:12 TP3] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:50:12 TP3] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:50:12 TP3] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:50:12 TP4] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:50:12 TP4] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:50:12 TP1] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:50:12 TP1] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:50:12 TP0] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:50:12 TP0] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:50:12 TP6] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:50:12 TP6] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:50:12 TP2] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:50:12 TP2] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:50:12 TP7] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:50:12 TP7] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:50:12 TP5] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:50:12 TP5] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:50:12 TP4] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:50:12 TP0] Config does not support fused shared expert(s). Shared experts fusion optimization is disabled.
[2026-02-26 16:50:12 TP1] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:50:12 TP0] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:50:12 TP6] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:50:12 TP2] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:50:12 TP7] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:50:12 TP5] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:50:12 TP3] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:50:12 TP4] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:50:12 TP1] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:50:12 TP0] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:50:12 TP6] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:50:12 TP2] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:50:12 TP7] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:50:12 TP5] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:50:13 TP0] Found local HF snapshot for moonshotai/Kimi-K2.5 at /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1; skipping download.

Loading safetensors checkpoint shards:   0% Completed | 0/64 [00:00<?, ?it/s]

Loading safetensors checkpoint shards:   8% Completed | 5/64 [00:00<00:01, 40.00it/s]

Loading safetensors checkpoint shards:  14% Completed | 9/64 [00:00<00:04, 11.35it/s]

Loading safetensors checkpoint shards:  20% Completed | 13/64 [00:00<00:03, 15.70it/s]

Loading safetensors checkpoint shards:  27% Completed | 17/64 [00:00<00:02, 19.43it/s]

Loading safetensors checkpoint shards:  33% Completed | 21/64 [00:01<00:01, 22.33it/s]

Loading safetensors checkpoint shards:  39% Completed | 25/64 [00:01<00:03, 12.19it/s]

Loading safetensors checkpoint shards:  45% Completed | 29/64 [00:01<00:02, 15.13it/s]

Loading safetensors checkpoint shards:  52% Completed | 33/64 [00:01<00:01, 18.03it/s]

Loading safetensors checkpoint shards:  58% Completed | 37/64 [00:02<00:01, 20.74it/s]

Loading safetensors checkpoint shards:  64% Completed | 41/64 [00:02<00:00, 23.12it/s]

Loading safetensors checkpoint shards:  70% Completed | 45/64 [00:02<00:00, 25.07it/s]

Loading safetensors checkpoint shards:  77% Completed | 49/64 [00:02<00:00, 26.64it/s]

Loading safetensors checkpoint shards:  83% Completed | 53/64 [00:03<00:00, 13.03it/s]

Loading safetensors checkpoint shards:  94% Completed | 60/64 [00:03<00:00, 19.31it/s]

Loading safetensors checkpoint shards: 100% Completed | 64/64 [00:03<00:00, 21.37it/s]

Loading safetensors checkpoint shards: 100% Completed | 64/64 [00:03<00:00, 18.76it/s]

[2026-02-26 16:51:05 TP3] Load weight end. elapsed=52.72 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=209.95 GB, mem usage=73.58 GB.
[2026-02-26 16:51:05 TP2] Load weight end. elapsed=52.85 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=209.96 GB, mem usage=73.58 GB.
[2026-02-26 16:51:05 TP1] Load weight end. elapsed=52.96 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.38 GB, mem usage=73.58 GB.
[2026-02-26 16:51:06 TP6] Load weight end. elapsed=53.90 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.08 GB, mem usage=73.58 GB.
[2026-02-26 16:51:07 TP5] Load weight end. elapsed=54.59 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.02 GB, mem usage=73.58 GB.
[2026-02-26 16:51:08 TP7] Load weight end. elapsed=56.18 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.10 GB, mem usage=73.58 GB.
[2026-02-26 16:51:12 TP4] Load weight end. elapsed=60.11 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.10 GB, mem usage=73.58 GB.
[2026-02-26 16:51:17 TP0] Load weight end. elapsed=64.96 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=209.98 GB, mem usage=73.58 GB.
[2026-02-26 16:51:17 TP0] Using KV cache dtype: torch.bfloat16
[2026-02-26 16:51:18 TP4] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:51:18 TP4] Memory pool end. avail mem=45.38 GB
[2026-02-26 16:51:18 TP5] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:51:18 TP5] Memory pool end. avail mem=45.30 GB
[2026-02-26 16:51:18 TP0] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:51:18 TP0] Memory pool end. avail mem=45.25 GB
[2026-02-26 16:51:18 TP1] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:51:18 TP1] Memory pool end. avail mem=45.65 GB
[2026-02-26 16:51:18 TP7] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:51:18 TP7] Memory pool end. avail mem=45.37 GB
[2026-02-26 16:51:18 TP2] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:51:18 TP2] Memory pool end. avail mem=45.24 GB
[2026-02-26 16:51:18 TP3] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:51:18 TP3] Memory pool end. avail mem=45.23 GB
[2026-02-26 16:51:18 TP6] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:51:18 TP6] Memory pool end. avail mem=45.36 GB
[2026-02-26 16:51:18 TP3] Capture cuda graph begin. This can take up to several minutes. avail mem=45.01 GB
[2026-02-26 16:51:18 TP5] Capture cuda graph begin. This can take up to several minutes. avail mem=45.08 GB
[2026-02-26 16:51:18 TP4] Capture cuda graph begin. This can take up to several minutes. avail mem=45.15 GB
[2026-02-26 16:51:18 TP2] Capture cuda graph begin. This can take up to several minutes. avail mem=45.01 GB
[2026-02-26 16:51:18 TP0] Capture cuda graph begin. This can take up to several minutes. avail mem=45.03 GB
[2026-02-26 16:51:18 TP0] Capture cuda graph bs [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]
[2026-02-26 16:51:18 TP1] Capture cuda graph begin. This can take up to several minutes. avail mem=45.43 GB
[2026-02-26 16:51:18 TP7] Capture cuda graph begin. This can take up to several minutes. avail mem=45.15 GB
[2026-02-26 16:51:18 TP6] Capture cuda graph begin. This can take up to several minutes. avail mem=45.13 GB

  0%|          | 0/52 [00:00<?, ?it/s]
Capturing batches (bs=512 avail_mem=43.46 GB):   0%|          | 0/52 [00:00<?, ?it/s][aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:51:20 TP2] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:51:20 TP2] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:51:20 TP4] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:51:20 TP4] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:51:20 TP6] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:51:20 TP6] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:51:20 TP3] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:51:20 TP3] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:51:20 TP1] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:51:20 TP1] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:51:20 TP7] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:51:20 TP7] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:51:20 TP0] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:51:20 TP0] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:51:20 TP5] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:51:20 TP5] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:51:21 TP3] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:51:22 TP2] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:51:22 TP6] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]

Capturing batches (bs=512 avail_mem=43.46 GB):   0%|          | 0/52 [00:02<?, ?it/s]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:51:22 TP0] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:51:22 TP1] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:51:22 TP5] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:51:22 TP4] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:51:22 TP7] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:51:22 TP7] Registering 0 cuda graph addresses
[2026-02-26 16:51:22 TP4] Registering 0 cuda graph addresses
[2026-02-26 16:51:22 TP0] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:51:22 TP5] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:51:22 TP1] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:51:22 TP2] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:51:22 TP6] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:51:22 TP3] Registering 0 cuda graph addresses
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:51:22 TP4] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:51:22 TP7] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:51:22 TP5] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:51:22 TP0] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:51:22 TP1] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:51:22 TP2] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:51:22 TP6] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:51:22 TP3] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:51:22 TP4] Scheduler hit an exception: Traceback (most recent call last):
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
  File "/sgl-workspace/sglang/python/sglang/srt/models/kimi_k25.py", line 726, in forward
    hidden_states = general_mm_embed_routine(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py", line 1134, in general_mm_embed_routine
    hidden_states = language_model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
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

[2026-02-26 16:51:22] Received sigquit from a child process. It usually means the child failed.
```
</details>

##### Command ( With NSA Backend ) : 
```
sglang serve --model-path moonshotai/Kimi-K2.5 --tp 8 --trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --nsa-prefill-backend tilelang --nsa-decode-backend tilelang```

##### Error Log : 

<details>
<summary>Click to view Error Log</summary>
  
```bash
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 16:44:47] server_args=ServerArgs(model_path='moonshotai/Kimi-K2.5', tokenizer_path='moonshotai/Kimi-K2.5', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=True, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.8260405859375, max_running_requests=None, max_queued_requests=None, max_total_tokens=None, chunked_prefill_size=16384, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=8, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, random_seed=735892918, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=None, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, custom_sigquit_handler=None, log_level='info', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='moonshotai/Kimi-K2.5', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser='kimi_k2', tool_call_parser='kimi_k2', tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='triton', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='pytorch', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend='tilelang', nsa_decode_backend='tilelang', disable_flashinfer_autotune=False, speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method='AMXINT4', kt_cpuinfer=None, kt_threadpool_count=2, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=512, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, enable_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=2048, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=16, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_decode_tp=None, disaggregation_decode_dp=None, disaggregation_prefill_pp=1, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)
/opt/venv/lib/python3.10/site-packages/apex/transformer/functional/fused_rope.py:49: UserWarning: Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0
  warnings.warn("Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0", UserWarning)
[2026-02-26 16:44:48] Ignore import error when loading sglang.srt.multimodal.processors.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:44:49] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:44:49] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:44:49] Using default HuggingFace chat template with detected content format: openai
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
INFO:aiter:import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so
[2026-02-26 16:44:53 TP1] Process 3088 gpu_id 1 is running on CPUs: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
[2026-02-26 16:44:53 TP5] Process 3092 gpu_id 5 is running on CPUs: [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
[2026-02-26 16:44:53 TP0] Process 3087 gpu_id 0 is running on CPUs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
[2026-02-26 16:44:53 TP7] Process 3094 gpu_id 7 is running on CPUs: [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
[2026-02-26 16:44:53 TP4] Process 3091 gpu_id 4 is running on CPUs: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
[2026-02-26 16:44:53 TP6] Process 3093 gpu_id 6 is running on CPUs: [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
[2026-02-26 16:44:53 TP3] Process 3090 gpu_id 3 is running on CPUs: [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
[2026-02-26 16:44:53 TP2] Process 3089 gpu_id 2 is running on CPUs: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
[2026-02-26 16:44:54] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:44:54] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
/sgl-workspace/sglang/python/sglang/srt/utils/hf_transformers_utils.py:558: UserWarning: Using a slow tokenizer. This might cause a significant slowdown. Consider using a fast tokenizer instead.
  warnings.warn(
[2026-02-26 16:44:54 TP1] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:44:54 TP1] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:44:54 TP5] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:44:54 TP5] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:44:54 TP7] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:44:54 TP7] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:44:54 TP0] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:44:54 TP0] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:44:54 TP4] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:44:54 TP4] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:44:54 TP6] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:44:54 TP6] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:44:54 TP2] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:44:54 TP2] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:44:54 TP3] Reloaded tiktoken model from /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1/tiktoken.model
[2026-02-26 16:44:54 TP3] #words: 163840 - BOS ID: 163584 - EOS ID: 163585
[2026-02-26 16:44:55 TP1] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:44:55 TP1] Init torch distributed begin.
[2026-02-26 16:44:55 TP1] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:44:55 TP5] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:44:55 TP5] Init torch distributed begin.
[2026-02-26 16:44:55 TP5] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:44:55 TP7] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:44:55 TP7] Init torch distributed begin.
[2026-02-26 16:44:55 TP7] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:44:55 TP0] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:44:55 TP0] Init torch distributed begin.
[2026-02-26 16:44:55 TP0] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:44:55 TP4] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:44:55 TP6] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:44:55 TP4] Init torch distributed begin.
[2026-02-26 16:44:55 TP4] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:44:55 TP6] Init torch distributed begin.
[2026-02-26 16:44:55 TP6] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:44:55 TP3] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:44:55 TP3] Init torch distributed begin.
[2026-02-26 16:44:55 TP3] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[2026-02-26 16:44:55 TP2] Calling super().encode with {'add_special_tokens': False}
[2026-02-26 16:44:55 TP2] Init torch distributed begin.
[2026-02-26 16:44:55 TP2] Failed to import amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
[Gloo] Rank 0 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 1 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 2 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 3 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 4 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 6 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 5 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 7 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[2026-02-26 16:44:56 TP0] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:44:56 TP7] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:44:56 TP3] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:44:56 TP6] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:44:56 TP2] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:44:56 TP4] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:44:56 TP1] [AR] All-reduce call path: NCCL (custom AR disabled)
[2026-02-26 16:44:56 TP5] [AR] All-reduce call path: NCCL (custom AR disabled)
[Gloo] Rank 0 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 1 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 4 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 2 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 3 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 5 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 6 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 7 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[2026-02-26 16:44:56 TP0] sglang is using nccl==2.27.7
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:45:08 TP5] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:45:08 TP7] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:45:08 TP4] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:45:08 TP1] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:45:08 TP3] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:45:08 TP2] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:45:08 TP6] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:45:08 TP5] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:45:08 TP4] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:45:08 TP1] [AR] Using AiterCustomAllreduce (AMD default)
[aiter] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:45:08 TP0] import [module_custom_all_reduce] under /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
[2026-02-26 16:45:08 TP3] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:45:08 TP6] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:45:08 TP2] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:45:08 TP0] [AR] Using AiterCustomAllreduce (AMD default)
[2026-02-26 16:45:08 TP7] [AR] Using AiterCustomAllreduce (AMD default)
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:45:08 TP0] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:45:08 TP7] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:45:08 TP6] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:45:08 TP1] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:45:08 TP3] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:45:08 TP5] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:45:08 TP2] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[2026-02-26 16:45:08 TP4] type hints mismatch, override to --> allocate_meta_buffer(size: int | typing.SupportsIndex) -> torch.Tensor
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:45:08 TP0] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:45:08 TP7] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:45:08 TP3] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:45:08 TP1] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:45:08 TP5] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:45:08 TP4] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:45:08 TP2] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[2026-02-26 16:45:08 TP6] type hints mismatch, override to --> init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex], rank: int | typing.SupportsIndex, fully_connected: bool) -> int
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP7] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP0] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP3] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP1] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP5] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP4] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP2] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP6] type hints mismatch, override to --> register_input_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP7] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP0] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP3] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP1] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP4] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP2] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[aiter] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP6] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP5] type hints mismatch, override to --> register_output_buffer(_fa: int | typing.SupportsIndex, t: torch.Tensor, handles: List[torch.Tensor], offsets: List[int | typing.SupportsIndex]) -> None
[2026-02-26 16:45:08 TP6] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:45:08 TP0] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:45:08 TP1] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:45:08 TP4] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:45:08 TP7] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:45:08 TP5] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:45:08 TP2] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
[2026-02-26 16:45:08 TP3] Failed to initialize QuickAllReduce: name 'amdsmi_shut_down' is not defined
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
[2026-02-26 16:45:08 TP0] Init torch distributed ends. elapsed=13.05 s, mem usage=3.81 GB
[2026-02-26 16:45:08 TP7] Init torch distributed ends. elapsed=13.10 s, mem usage=3.69 GB
[2026-02-26 16:45:08 TP6] Init torch distributed ends. elapsed=13.04 s, mem usage=3.70 GB
[2026-02-26 16:45:08 TP5] Init torch distributed ends. elapsed=13.29 s, mem usage=3.76 GB
[2026-02-26 16:45:08 TP4] Init torch distributed ends. elapsed=13.04 s, mem usage=3.69 GB
[2026-02-26 16:45:08 TP3] Init torch distributed ends. elapsed=12.99 s, mem usage=3.83 GB
[2026-02-26 16:45:08 TP2] Init torch distributed ends. elapsed=12.97 s, mem usage=3.83 GB
[2026-02-26 16:45:08 TP1] Init torch distributed ends. elapsed=13.50 s, mem usage=3.41 GB
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
[2026-02-26 16:45:08 TP2] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP2] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP2] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:45:08 TP3] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP3] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP3] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:45:08 TP0] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP0] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP0] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:45:08 TP7] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP7] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP7] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:45:08 TP6] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP6] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP6] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:45:08 TP1] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP1] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP1] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:45:08 TP5] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP4] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP5] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP4] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-26 16:45:08 TP5] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:45:08 TP4] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/opt/venv/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-26 16:45:08 TP2] Load weight begin. avail mem=283.54 GB
[2026-02-26 16:45:08 TP3] Load weight begin. avail mem=283.53 GB
[2026-02-26 16:45:08 TP0] Load weight begin. avail mem=283.56 GB
[2026-02-26 16:45:08 TP7] Load weight begin. avail mem=283.68 GB
[2026-02-26 16:45:08 TP6] Load weight begin. avail mem=283.66 GB
[2026-02-26 16:45:08 TP4] Load weight begin. avail mem=283.68 GB
[2026-02-26 16:45:08 TP5] Load weight begin. avail mem=283.60 GB
[2026-02-26 16:45:08 TP1] Load weight begin. avail mem=283.96 GB
[2026-02-26 16:45:09 TP2] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:45:09 TP2] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:45:09 TP2] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:45:09 TP3] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:45:09 TP3] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:45:09 TP0] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:45:09 TP0] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:45:09 TP7] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:45:09 TP7] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:45:09 TP6] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:45:09 TP6] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:45:09 TP1] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:45:09 TP1] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:45:09 TP4] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:45:09 TP4] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:45:09 TP5] Multimodal attention backend not set. Use aiter_attn.
[2026-02-26 16:45:09 TP5] Using aiter_attn as multimodal attention backend.
[2026-02-26 16:45:09 TP3] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:45:09 TP0] Config does not support fused shared expert(s). Shared experts fusion optimization is disabled.
[2026-02-26 16:45:09 TP0] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:45:09 TP7] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:45:09 TP6] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:45:09 TP1] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:45:09 TP4] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:45:09 TP5] Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod
[2026-02-26 16:45:09 TP2] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:45:09 TP3] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:45:09 TP0] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:45:09 TP1] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:45:09 TP6] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:45:09 TP4] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:45:09 TP7] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:45:09 TP5] Using CompressedTensorsWNA16TritonMoE (ROCm)
[2026-02-26 16:45:09 TP0] Found local HF snapshot for moonshotai/Kimi-K2.5 at /home/models/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1; skipping download.

Loading safetensors checkpoint shards:   0% Completed | 0/64 [00:00<?, ?it/s]

Loading safetensors checkpoint shards:   8% Completed | 5/64 [00:00<00:01, 39.55it/s]

Loading safetensors checkpoint shards:  14% Completed | 9/64 [00:00<00:04, 11.04it/s]

Loading safetensors checkpoint shards:  20% Completed | 13/64 [00:00<00:03, 15.41it/s]

Loading safetensors checkpoint shards:  27% Completed | 17/64 [00:00<00:02, 19.16it/s]

Loading safetensors checkpoint shards:  33% Completed | 21/64 [00:01<00:01, 22.06it/s]

Loading safetensors checkpoint shards:  39% Completed | 25/64 [00:01<00:03, 12.06it/s]

Loading safetensors checkpoint shards:  45% Completed | 29/64 [00:01<00:02, 14.99it/s]

Loading safetensors checkpoint shards:  52% Completed | 33/64 [00:01<00:01, 17.89it/s]

Loading safetensors checkpoint shards:  58% Completed | 37/64 [00:02<00:01, 20.60it/s]

Loading safetensors checkpoint shards:  64% Completed | 41/64 [00:02<00:01, 22.98it/s]

Loading safetensors checkpoint shards:  70% Completed | 45/64 [00:02<00:00, 24.95it/s]

Loading safetensors checkpoint shards:  77% Completed | 49/64 [00:02<00:00, 26.52it/s]

Loading safetensors checkpoint shards:  81% Completed | 52/64 [00:03<00:00, 12.28it/s]

Loading safetensors checkpoint shards:  89% Completed | 57/64 [00:03<00:00, 16.39it/s]

Loading safetensors checkpoint shards:  97% Completed | 62/64 [00:03<00:00, 21.29it/s]

Loading safetensors checkpoint shards: 100% Completed | 64/64 [00:03<00:00, 18.54it/s]

[2026-02-26 16:46:00 TP2] Load weight end. elapsed=51.25 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=209.96 GB, mem usage=73.58 GB.
[2026-02-26 16:46:00 TP3] Load weight end. elapsed=51.94 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=209.95 GB, mem usage=73.58 GB.
[2026-02-26 16:46:01 TP1] Load weight end. elapsed=52.77 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.38 GB, mem usage=73.58 GB.
[2026-02-26 16:46:02 TP6] Load weight end. elapsed=53.40 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.08 GB, mem usage=73.58 GB.
[2026-02-26 16:46:02 TP5] Load weight end. elapsed=53.70 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.02 GB, mem usage=73.58 GB.
[2026-02-26 16:46:05 TP7] Load weight end. elapsed=56.51 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.10 GB, mem usage=73.58 GB.
[2026-02-26 16:46:11 TP4] Load weight end. elapsed=62.08 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=210.10 GB, mem usage=73.58 GB.
[2026-02-26 16:46:13 TP0] Load weight end. elapsed=64.78 s, type=KimiK25ForConditionalGeneration, dtype=torch.bfloat16, avail mem=209.98 GB, mem usage=73.58 GB.
[2026-02-26 16:46:13 TP0] Using KV cache dtype: torch.bfloat16
[2026-02-26 16:46:14 TP6] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:46:14 TP4] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:46:14 TP6] Memory pool end. avail mem=45.36 GB
[2026-02-26 16:46:14 TP4] Memory pool end. avail mem=45.38 GB
[2026-02-26 16:46:14 TP2] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:46:14 TP2] Memory pool end. avail mem=45.24 GB
[2026-02-26 16:46:14 TP5] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:46:14 TP5] Memory pool end. avail mem=45.30 GB
[2026-02-26 16:46:14 TP7] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:46:14 TP7] Memory pool end. avail mem=45.37 GB
[2026-02-26 16:46:14 TP3] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:46:14 TP0] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:46:14 TP3] Memory pool end. avail mem=45.23 GB
[2026-02-26 16:46:14 TP0] Memory pool end. avail mem=45.25 GB
[2026-02-26 16:46:14 TP1] KV Cache is allocated. #tokens: 2454391, KV size: 160.63 GB
[2026-02-26 16:46:14 TP1] Memory pool end. avail mem=45.65 GB
[2026-02-26 16:46:14 TP4] Capture cuda graph begin. This can take up to several minutes. avail mem=45.15 GB
[2026-02-26 16:46:14 TP7] Capture cuda graph begin. This can take up to several minutes. avail mem=45.15 GB
[2026-02-26 16:46:14 TP3] Capture cuda graph begin. This can take up to several minutes. avail mem=45.01 GB
[2026-02-26 16:46:14 TP6] Capture cuda graph begin. This can take up to several minutes. avail mem=45.13 GB
[2026-02-26 16:46:14 TP5] Capture cuda graph begin. This can take up to several minutes. avail mem=45.07 GB
[2026-02-26 16:46:14 TP0] Capture cuda graph begin. This can take up to several minutes. avail mem=45.03 GB
[2026-02-26 16:46:14 TP0] Capture cuda graph bs [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]
[2026-02-26 16:46:14 TP1] Capture cuda graph begin. This can take up to several minutes. avail mem=45.43 GB
[2026-02-26 16:46:14 TP2] Capture cuda graph begin. This can take up to several minutes. avail mem=45.01 GB

  0%|          | 0/52 [00:00<?, ?it/s]
Capturing batches (bs=512 avail_mem=43.46 GB):   0%|          | 0/52 [00:00<?, ?it/s][aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:46:16 TP2] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:46:16 TP2] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:46:16 TP0] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:46:16 TP0] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:46:16 TP7] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:46:16 TP7] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:46:16 TP3] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:46:16 TP3] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:46:16 TP6] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:46:16 TP6] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:46:16 TP1] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:46:16 TP1] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:46:16 TP4] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:46:16 TP4] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[2026-02-26 16:46:16 TP5] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[2026-02-26 16:46:16 TP5] type hints mismatch, override to --> rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:46:17 TP7] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:46:18 TP4] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:46:18 TP6] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:46:18 TP5] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:46:18 TP3] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:46:18 TP2] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:46:18 TP1] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]

Capturing batches (bs=512 avail_mem=43.46 GB):   0%|          | 0/52 [00:03<?, ?it/s]
[aiter] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[2026-02-26 16:46:18 TP0] type hints mismatch, override to --> get_graph_buffer_ipc_meta(_fa: int | typing.SupportsIndex) -> Tuple[torch.Tensor, torch.Tensor]
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:46:18 TP0] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:46:18 TP3] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:46:18 TP7] Registering 0 cuda graph addresses
[2026-02-26 16:46:18 TP4] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:46:18 TP6] Registering 0 cuda graph addresses
[2026-02-26 16:46:18 TP2] Registering 0 cuda graph addresses
[2026-02-26 16:46:18 TP5] Registering 0 cuda graph addresses
[aiter] Registering 0 cuda graph addresses
[2026-02-26 16:46:18 TP1] Registering 0 cuda graph addresses
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:46:18 TP0] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:46:18 TP3] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:46:18 TP6] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:46:18 TP5] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:46:18 TP4] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:46:18 TP2] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:46:18 TP7] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[aiter] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:46:18 TP1] type hints mismatch, override to --> register_graph_buffers(_fa: int | typing.SupportsIndex, handles: List[torch.Tensor], offsets: List[torch.Tensor]) -> None
[2026-02-26 16:46:18 TP3] Scheduler hit an exception: Traceback (most recent call last):
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
  File "/sgl-workspace/sglang/python/sglang/srt/models/kimi_k25.py", line 726, in forward
    hidden_states = general_mm_embed_routine(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py", line 1134, in general_mm_embed_routine
    hidden_states = language_model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
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

[2026-02-26 16:46:18] Received sigquit from a child process. It usually means the child failed.
[2026-02-26 16:46:18 TP5] Scheduler hit an exception: Traceback (most recent call last):
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
  File "/sgl-workspace/sglang/python/sglang/srt/models/kimi_k25.py", line 726, in forward
    hidden_states = general_mm_embed_routine(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py", line 1134, in general_mm_embed_routine
    hidden_states = language_model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
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

[2026-02-26 16:46:18] Received sigquit from a child process. It usually means the child failed.
[2026-02-26 16:46:18 TP7] Scheduler hit an exception: Traceback (most recent call last):
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
  File "/sgl-workspace/sglang/python/sglang/srt/models/kimi_k25.py", line 726, in forward
    hidden_states = general_mm_embed_routine(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py", line 1134, in general_mm_embed_routine
    hidden_states = language_model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
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

[2026-02-26 16:46:18] Received sigquit from a child process. It usually means the child failed.
[2026-02-26 16:46:18 TP2] Scheduler hit an exception: Traceback (most recent call last):
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
  File "/sgl-workspace/sglang/python/sglang/srt/models/kimi_k25.py", line 726, in forward
    hidden_states = general_mm_embed_routine(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py", line 1134, in general_mm_embed_routine
    hidden_states = language_model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
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

[2026-02-26 16:46:18] Received sigquit from a child process. It usually means the child failed.
[2026-02-26 16:46:18 TP6] Scheduler hit an exception: Traceback (most recent call last):
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
  File "/sgl-workspace/sglang/python/sglang/srt/models/kimi_k25.py", line 726, in forward
    hidden_states = general_mm_embed_routine(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py", line 1134, in general_mm_embed_routine
    hidden_states = language_model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
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

[2026-02-26 16:46:18] Received sigquit from a child process. It usually means the child failed.
[2026-02-26 16:46:18 TP0] Scheduler hit an exception: Traceback (most recent call last):
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
  File "/sgl-workspace/sglang/python/sglang/srt/models/kimi_k25.py", line 726, in forward
    hidden_states = general_mm_embed_routine(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py", line 1134, in general_mm_embed_routine
    hidden_states = language_model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
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

[2026-02-26 16:46:18] Received sigquit from a child process. It usually means the child failed.
[2026-02-26 16:46:18 TP4] Scheduler hit an exception: Traceback (most recent call last):
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
  File "/sgl-workspace/sglang/python/sglang/srt/models/kimi_k25.py", line 726, in forward
    hidden_states = general_mm_embed_routine(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py", line 1134, in general_mm_embed_routine
    hidden_states = language_model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
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

[2026-02-26 16:46:18] Received sigquit from a child process. It usually means the child failed.
[2026-02-26 16:46:18 TP1] Scheduler hit an exception: Traceback (most recent call last):
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
  File "/sgl-workspace/sglang/python/sglang/srt/models/kimi_k25.py", line 726, in forward
    hidden_states = general_mm_embed_routine(
  File "/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py", line 1134, in general_mm_embed_routine
    hidden_states = language_model(
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
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

[2026-02-26 16:46:18] Received sigquit from a child process. It usually means the child failed.
```
</details>
