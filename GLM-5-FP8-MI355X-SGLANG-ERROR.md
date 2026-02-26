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

#### Output : 
```

```


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

#### Output : 
```
```

echo "Hello, World!"
for i in {1..5}; do
    echo "Iteration $i"
done
