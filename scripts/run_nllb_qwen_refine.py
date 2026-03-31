from sacrebleu import BLEU
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from vllm import LLM, SamplingParams


def main() -> None:
    # 将 GPU/多进程相关初始化放到 main 内，避免 vLLM spawn 重新导入时重复执行。
    nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M"
    ).cuda()
    llm = LLM(
        model="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8",
        gpu_memory_utilization=0.9,
    )

    with open("data/enzh/wmt19_source.txt", "r") as f:
        sources = f.readlines()
    with open("data/enzh/wmt19_target.txt", "r") as f:
        targets = f.readlines()

    outputs = []
    for src in sources:
        inputs = nllb_tokenizer(src.strip(), return_tensors="pt").to("cuda")
        draft = nllb_tokenizer.decode(
            nllb_model.generate(**inputs, max_length=200)[0],
            skip_special_tokens=True,
        )

        prompt = (
            "Translate the following English text to Chinese, improving the draft "
            f"if needed:\nEnglish: {src.strip()}\nDraft: {draft}\nChinese:"
        )
        sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
        response = llm.generate(prompt, sampling_params)[0].outputs[0].text.strip()
        outputs.append(response)

    with open("outputs/nllb_qwen_refine_wmt19.txt", "w") as f:
        f.writelines(o + "\n" for o in outputs)

    bleu = BLEU()
    score = bleu.corpus_score(outputs, [targets])
    print(f"BLEU: {score.score}")


if __name__ == "__main__":
    main()
