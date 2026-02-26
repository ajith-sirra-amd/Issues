# Model : moonshotai/Kimi-K2.5

## Here is what I tried : 

- Rocm SGLang Docker (20260218) with Default Transformer [See Here](#error-log-)
- Rocm SGLang Docker (20260218) with Updated Transformer [See Here](#error-log--1)
- Rocm SGLang Docker (20260225) with Default Transformer [See Here](#error-log--2)
- Rocm SGLang Docker (20260225) with Updated Transformer [See Here](#error-log--3)
- Rocm LMSYS Docker with Updated Transformer without any code changes [See Here](#error-log--4)
- Rocm LMSYS Docker with Updated Transformer with code changes [See Here](#error-log--5)

  
## Rocm SGLang Docker (20260218)  

##### Docker Image : rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260218
##### Transformers : 4.57.1

##### Command : 
```
python3 -m sglang.launch_server --attention-backend triton --model-path zai-org/GLM-5-FP8 --tp-size 8
```

##### Error Log : 

<details>
<summary>Click to view Error Log</summary>
  
```bash

```
</details>

##### Resolution : 
```
pip install git+https://github.com/huggingface/transformers.git
```
##### Output : 
```
Successfully installed huggingface-hub-1.4.1 markdown-it-py-4.0.0 mdurl-0.1.2 rich-14.3.3 shellingham-1.5.4 transformers-5.3.0.dev0 typer-0.24.1 typer-slim-0.24.0
```
##### Transformers : 5.3.0.dev0
##### Command : 
```
python3 -m sglang.launch_server --attention-backend triton --model-path zai-org/GLM-5-FP8 --tp-size 8 --kv-cache-dtype fp8_e4m3
```
