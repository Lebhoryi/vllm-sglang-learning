import os

from vllm import LLM, SamplingParams


# 将模型加载和推理代码放到 main 函数中
def main():
    # 1. 加载模型（模型目录通过环境变量 MODEL_PATH 指定）
    model_path = os.environ["MODEL_PATH"]
    llm = LLM(model=model_path, trust_remote_code=True)

    # 2. 设置采样参数
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64)

    # 3. 输入提示词
    prompts = ["Hello, my name is", "What is the capital of France?"]

    # 4. 运行推理
    outputs = llm.generate(prompts, sampling_params)

    # 5. 输出结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text!r}")
        print("-" * 50)

if __name__ == "__main__":
    main()
