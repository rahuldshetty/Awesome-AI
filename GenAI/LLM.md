# Awesome-LLM

Large Language Models (LLMs) are foundational machine learning models to process and understand natural language. These models are trained on massive amounts of text data to learn patterns in the language. LLMs can perform many types of language tasks, such as translating languages, analyzing sentiments, chatbot, conversational agents, and much more.



## Frameworks

* [transformers](https://github.com/huggingface/text-generation-inference): 🤗 Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models.
* [datasets](https://github.com/huggingface/datasets): 🤗 The largest hub of ready-to-use datasets for ML models with fast, easy-to-use and efficient data manipulation tools.
* [FastChat](https://github.com/lm-sys/FastChat): FastChat is an open platform for training, serving, and evaluating large language model based chatbots.
* [OpenChatKit](https://github.com/togethercomputer/OpenChatKit): OpenChatKit provides a powerful, open-source base to create both specialized and general purpose chatbots for various applications.

### Training

* [accelerate](https://github.com/huggingface/accelerate): 🚀 A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision.
* [optimum](https://github.com/huggingface/optimum): 🚀 Accelerate training and inference of 🤗 Transformers and 🤗 Diffusers with easy to use hardware optimization tools.

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

### Leaderboard

* [LLM Explorer](https://llm.extractum.io/)
* [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)


## Tutorials

* [HuggingFace - Training a causal language model from scratch](https://huggingface.co/learn/nlp-course/chapter7/6)
* [Training Conversational AI](https://erichartford.com/meet-samantha)
* [Training Your Own LLM using privateGPT](https://levelup.gitconnected.com/training-your-own-llm-using-privategpt-f36f0c4f01ec)
* 
    

## Blogs/Articles

* [Analytics Vidhya - What are LLMs?](https://www.analyticsvidhya.com/blog/2023/03/an-introduction-to-large-language-models-llms/)
* [Replit - How to train your own Large Language Models](https://blog.replit.com/llm-training)
* [HuggingFace - Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
* [Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU](https://huggingface.co/blog/trl-peft)
* 
  



