# Awesome-LLM

Large Language Models (LLMs) are foundational machine learning models to process and understand natural language. These models are trained on massive amounts of text data to learn patterns in the language. LLMs can perform many types of language tasks, such as translating languages, analyzing sentiments, chatbot, conversational agents, and much more.



## Frameworks

* [transformers](https://github.com/huggingface/text-generation-inference): 🤗 Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models.
* [datasets](https://github.com/huggingface/datasets): 🤗 The largest hub of ready-to-use datasets for ML models with fast, easy-to-use and efficient data manipulation tools.
* [FastChat](https://github.com/lm-sys/FastChat): FastChat is an open platform for training, serving, and evaluating large language model based chatbots.
* [OpenChatKit](https://github.com/togethercomputer/OpenChatKit): OpenChatKit provides a powerful, open-source base to create both specialized and general purpose chatbots for various applications.
* [llm](https://github.com/rustformers/llm): llm is an ecosystem of Rust libraries for working with large language models - it's built on top of the fast, efficient GGML library for machine learning.
* [llm-rs-python](https://github.com/LLukas22/llm-rs-python): 🐍❤️🦀 Unofficial python bindings for the rust llm library.
* [llama.cpp](https://github.com/ggerganov/llama.cpp): Port of Facebook's LLaMA model in C/C++
* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python): Python bindings for llama.cpp
* [ctransformers](https://github.com/marella/ctransformers): Python bindings for the Transformer models implemented in C/C++ using GGML library.
* [langchain](https://github.com/hwchase17/langchain): ⚡ Building applications with LLMs through composability ⚡
* [langchainjs](https://github.com/hwchase17/langchainjs): langchain but in browser
* 


### Training

* [accelerate](https://github.com/huggingface/accelerate): 🚀 A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision.
* [optimum](https://github.com/huggingface/optimum): 🚀 Accelerate training and inference of 🤗 Transformers and 🤗 Diffusers with easy to use hardware optimization tools.
* [FairScale](https://github.com/facebookresearch/fairscale): FairScale is a PyTorch extension library for high performance and large scale training.
* [DeepSpeed](https://github.com/microsoft/DeepSpeed): DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective.
* [FlexFlow](https://github.com/flexflow/FlexFlow): FlexFlow is a deep learning framework that accelerates distributed DNN training by automatically searching for efficient parallelization strategies.
* [Mistral](https://github.com/stanford-crfm/mistral): A framework for transparent and accessible large-scale language model training, built with Hugging Face 🤗.
* [MMEngine](https://github.com/open-mmlab/mmengine): MMEngine is a foundational library for training deep learning models based on PyTorch.

### Serving

* [text-generation-inference](https://github.com/huggingface/text-generation-inference): A Rust, Python and gRPC server for text generation inference. Used in production at [HuggingFace](https://huggingface.co/) to power LLMs api-inference widgets.
* [vLLM](https://github.com/vllm-project/vllm) :A high-throughput and memory-efficient inference and serving engine for LLMs.
* [Basaran](https://github.com/hyperonym/basaran): Basaran is an open-source alternative to the OpenAI text completion API. It provides a compatible streaming API for your Hugging Face Transformers-based text generation models.
* [node-llmatic](https://github.com/fardjad/node-llmatic): LLMatic can be used as a drop-in replacement for OpenAI's API (see the supported endpoints). It uses llama-node with llama.cpp backend to run the models locally.

### Quantization

* [PEFT](https://github.com/huggingface/peft): 🤗Parameter-Efficient-Fine-Tuning methods enables efficient adaption of pre-trained language models (PLMs) to various downstream applications without fine-tuning all the model's parameters. 
* [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ): An easy-to-use LLMs quantization package with user-friendly apis, based on GPTQ algorithm.
* [ExLLama](https://github.com/turboderp/exllama): A standalone Python/C++/CUDA implementation of Llama for use with 4-bit GPTQ weights, designed to be fast and memory-efficient on modern GPUs.
* [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa): 4 bits quantization of LLaMA using GPTQ.
* [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq): Efficient and accurate low-bit weight quantization (INT3/4) for LLMs, supporting instruction-tuned models and multi-modal LMs.
* [CTranslate2](https://github.com/OpenNMT/CTranslate2): CTranslate2 is a C++ and Python library for efficient inference with Transformer models.
* 

### Prompting

* [Prompt Engineering Guide](https://www.promptingguide.ai/)
* [Prompt Engineering Blog by lilianweng](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
* [Prompt Reflexion](https://colab.research.google.com/drive/13FqOO9DoFZa6B0JnJhrpmkxGePBupCyE#scrollTo=s-Quxe2MQsHc)
* 

### Leaderboard

* [LLM Explorer](https://llm.extractum.io/)
* [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
* [Can AI Code - Results?](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results)


## Tutorials

* [HuggingFace - Training a causal language model from scratch](https://huggingface.co/learn/nlp-course/chapter7/6)
* [Training Conversational AI](https://erichartford.com/meet-samantha)
* [Training Your Own LLM using privateGPT](https://levelup.gitconnected.com/training-your-own-llm-using-privategpt-f36f0c4f01ec)
* [Fine-tuning OpenLLaMA-7B with QLoRA for instruction following](https://georgesung.github.io/ai/qlora-ift/)
* [Building with Instruction-Tuned LLMs: A Step-by-Step Guide](https://github.com/FourthBrain/Building-with-Instruction-Tuned-LLMs-A-Step-by-Step-Guide)
* [Finetune Falcon-7b on a Google colab](https://colab.research.google.com/drive/1BiQiw31DT7-cDp1-0ySXvvhzqomTdI-o?usp=sharing)
* [NTK Aware Scaled RotaryEmbedding](https://colab.research.google.com/drive/1VI2nhlyKvd5cw4-zHvAIk00cAVj2lCCC#scrollTo=d2ceb547)
* 
    

## Blogs/Articles

* [Analytics Vidhya - What are LLMs?](https://www.analyticsvidhya.com/blog/2023/03/an-introduction-to-large-language-models-llms/)
* [Replit - How to train your own Large Language Models](https://blog.replit.com/llm-training)
* [HuggingFace - Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
* [Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU](https://huggingface.co/blog/trl-peft)
* [The Falcon has landed in the Hugging Face ecosystem](https://huggingface.co/blog/falcon)
* [LLM Parameters Demystified](https://txt.cohere.com/llm-parameters-best-outputs-language-ai/)
* [Building LLM applications for production](https://huyenchip.com/2023/04/11/llm-engineering.html)
* [Reddit: NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation.](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
* [LocalLLaMa Wiki Models](https://www.reddit.com/r/LocalLLaMA/wiki/index/#wiki_models)
* [Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
  



