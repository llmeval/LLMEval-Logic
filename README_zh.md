<p align="center">
  <img src="llmeval-logo.png" width="200">
</p>

<h2 align="center">LLMEval-Logic：经求解器验证、含对抗强化的中文逻辑推理评测基准</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2605.19597"><img src="https://img.shields.io/badge/论文-Arxiv-blue.svg?style=for-the-badge" alt="论文"></a>
  <a href="https://huggingface.co/datasets/llmeval-fdu/LLMEval-Logic"><img src="https://img.shields.io/badge/数据集-HuggingFace-yellow.svg?style=for-the-badge" alt="数据集"></a>
  <a href="https://llmeval.com/"><img src="https://img.shields.io/badge/官网-llmeval.com-2ea44f.svg?style=for-the-badge" alt="官网"></a>
  <a href="https://github.com/llmeval"><img src="https://img.shields.io/badge/组织-LLMEval-green.svg?style=for-the-badge" alt="LLMEval"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0%20%2B%20Eval--Only-blue.svg?style=for-the-badge" alt="License"></a>
</p>

> **说明：** 英文版 README 请参见 [README.md](README.md)。

## 🔔 动态

- 🎉 **[2026-05]** 论文已上 arXiv：[arXiv:2605.19597](https://arxiv.org/abs/2605.19597)。
- 📂 **[2026-05]** **80% 公开版** 同步发布。剩余 20%（50 道 Base / 36 道 Hard / 50 个 rubric）由复旦 NLP 实验室作为**私有抗污染测试集**自留。

## 📚 数据集介绍

LLMEval-Logic 是一个中文逻辑推理评测基准，构造采用**三阶段审计流水线**：(a) 由具备逻辑基础的标注员从真实情境**正向撰写**题目，而非从公式反向模板化；(b) 每条题目经过四层规范化处理，并由人工编写的 rubric 检查表与 **Z3 SMT 求解器**双重审计自然语言到形式语言（NL→FL）的翻译；(c) 通过闭环**对抗强化 Agent 工作流**淘汰过于简单的样本。数据集由两个配对子集组成：

- **LLMEval-Logic-Base** —— 命题逻辑与一阶逻辑的单问题题目，每题附 Z3 验证答案、金标 FL，以及 atom 级 NL→FL rubric（全集 246 道 / 1,400 atoms；公开版 196 道）。
- **LLMEval-Logic-Hard** —— 在 Base 同一模型空间上经六种对抗策略（分支化、有效干扰项、显式不确定、集值输出、反事实变体、别名/共指变化）生成的多问题题目。论文中最强前沿模型 Hard 子集 Item Accuracy 仅 **37.5%**。

## 🗂️ 项目结构

```
.
├── bench/                                  80% 公开发布
│   ├── base/                               Base 题目 + 配套 rubric
│   │   ├── llmeval_logic_base.json         196 道 + 金标 FL + 答案
│   │   └── rubrics/                        196 个 per-problem rubric（NL + Z3 atoms）
│   └── hard/
│       └── llmeval_logic_hard.json         154 道 / 766 子问题
│
├── code/
│   ├── client.py                           OpenAI 兼容的可插拔模型分发
│   ├── lib/                                共享：HTTP / FL schema / Z3 engine / IO
│   │
│   ├── nl_eval/                            直接评测轴（Base + Hard 都用）
│   │   ├── eval.py
│   │   └── llm_judge/
│   │
│   └── fl_eval/                            形式化评测轴（只用 Base）
│       ├── formalize/                      NL → 候选 FL JSON
│       ├── z3_judge/                       Z3 求解 → "Z3" 列
│       └── rubric_judge/                rubric 原子打分 → "Rubric" 列
│
├── scripts/split.py                        确定性的分层切分脚本（seed=2026）
├── evaluate.py                             ★ 一键端到端入口
├── requirements.txt
└── .env.example                            复制为 .env，填 OPENAI_BASE_URL + OPENAI_API_KEY
```

仓库结构刻意对应论文中**两条正交的评测轴**：`bench/base/` 装的是形式化评测需要的全部材料（题目 + rubric）；`bench/hard/` 是只参与直接答案评测的多问题子集。

## 💾 数据结构

### Base 题目（`bench/base/llmeval_logic_base.json`）

每条 Base 题目至少包含：

- **`id`** —— 在全 436 道题目上分配的全局整数 id（Base 占 `0..245`，Hard 占 `246..435`）。同一道题在公开 / 私有 / 全集中 `id` 完全一致，也是 rubric 文件名（如 `id=10` ↔ `bench/base/rubrics/010.json`）。
- **`title`** —— 简短中文标签。
- **`logictype`** —— `pl`（命题逻辑）或 `fol`（一阶逻辑）。
- **`original.background` / `original.question` / `original.answer`** —— 中文自然语言背景、问题与参考答案（自由文本）。
- **`formalization.parameters` / `.translation` / `.premise` / `.question` / `.answer`** —— 人工审定的金标 FL（参数、NL→符号映射、形式前提、形式查询，以及 Z3 求解的金标答案）。
- **`label_type`** —— 答案类型标签列表（`possible` / `necessary` / `enumerate_models` / `count_models` 等）。

公开发布的 Base 是这套 id 的一个子集（`0..245` 中的 196 个），私有保留集是其补集；二者相对 `0..245` 都是稀疏的。

### Hard 题目（`bench/hard/llmeval_logic_hard.json`）

Hard 题目占同一套全局 id 空间的 `246..435` 段，**故意不带形式化**：

- **`id`** —— 全局整数 id（`246..435`）。公开 Hard 取其中 154 个，私有 Hard 是补集。
- **`title`** —— ≤ 10 字的中文短标签。
- **`background`** —— 中文自然语言情境描述。
- **`question`** —— 子问题字符串列表。
- **`answer`** —— 与 `question` 等长的金标答案列表（Z3 + 人工双重核对）。

### Rubric 文件（`bench/base/rubrics/<id>.json`）

每题一份的 NL→FL 忠实度原子级检查表，分三组：

- **`logical_relation`** —— 候选 FL 是否保留了原题的逻辑关系？
- **`stated_constraint`** —— 题目陈述的约束是否被保留？
- **`query_alignment`** —— 候选的 query 在语义上是否与 NL 提问一致？

每个 atom 同时带"自然语言判断标准"和"Z3 可执行公式"——生产评判器据此混合 solver 自动判定的 atom 与 LLM 判定的 atom。

## 🛠️ 使用指南

```bash
git clone https://github.com/llmeval/LLMEval-Logic.git
cd LLMEval-Logic

pip install -r requirements.txt
cp .env.example .env
# 编辑 .env：把 OPENAI_BASE_URL 和 OPENAI_API_KEY 填成任意 OpenAI 兼容端点
# （OpenAI / OpenRouter / vLLM / SGLang / Ollama …均可）

python evaluate.py --model openai/gpt-4o
```

`evaluate.py` 是**唯一一键入口**：传入任意 OpenAI 兼容模型 id，即可端到端跑完四个阶段（`nl-base` / `nl-hard` / `fl-free` / `fl-fixed`），最后打印一份汇总记分板。所有阶段**支持断点续跑**：重跑同样的命令会从上次中断处继续。

常用 flag：

```bash
# 自定义评判模型（默认 openai/gpt-4o）：
python evaluate.py --model my-model --judge-model anthropic/claude-3.5-sonnet

# 只跑前 3 题做烟雾测试：
python evaluate.py --model openai/gpt-4o --limit 3

# 跳过 / 只跑某一阶段：
python evaluate.py --model openai/gpt-4o --skip nl-hard
python evaluate.py --model openai/gpt-4o --only fl-fixed
```

如果你的服务商兼容 OpenAI Chat Completions、模型 id 可原样转发，无需改任何代码。若模型需要额外请求参数（如某种 reasoning 开关），在 `code/client.py:MODEL_CONFIGS` 注册一个 key 后传给 `--model` 即可。

## 📊 评测指标

| 阶段       | 子集                                       | 指标               | 衡量什么                                                                  |
|------------|--------------------------------------------|--------------------|---------------------------------------------------------------------------|
| `nl-base`  | `bench/base/`                              | Item / Sub-Q Acc   | 模型自由回答与 Z3 验证过的参考答案是否一致（LLM 评判）                       |
| `nl-hard`  | `bench/hard/`                              | Item / Sub-Q Acc   | 同上，但在多问题对抗强化子集上评测                                          |
| `fl-free`  | `bench/base/` + `bench/base/rubrics/`      | Z3 / Rubric / Both | 模型自主选择符号系统 → Z3 求解 + atom 级 rubric 打分                        |
| `fl-fixed` | `bench/base/` + `bench/base/rubrics/`      | Z3 / Rubric / Both | 注入金标 parameters/translation，模型只写 premise/question；rubric 含 Z3 预筛 |

`Z3` 列表示"模型形式化经 Z3 求解导出的自然语言答案是否与参考答案一致"；`Rubric` 列表示"`bench/base/rubrics/<id>.json` 中所有人工 atom 是否都通过"（Z3 + LLM 混合判断）；`Both` 列为两者交集 —— 最严格的一列。

论文所有数字均使用 `gpt-5.1-chat` 作 LLM-as-Judge、三次独立采样平均得到。与另外两个前沿评判模型（Claude Opus 4.6 / Gemini 3.1 Pro）的成对 Cohen's κ ∈ [0.873, 0.922]，按 Landis & Koch (1977) 标准属于 "almost perfect"。

## 🔐 私有 20% 留存

参照 [LLMEval-Fair](https://github.com/llmeval/LLMEval-Fair) 的抗污染评测做法，本次公开发布仅包含 LLMEval-Logic 的 **80%**，剩余 **20%**（50 道 Base / 36 道 Hard / 50 个 rubric）**不对外开源**，由复旦 NLP 实验室内部维护。

公开切分由确定性的、固定种子（`seed=2026`）**分层随机抽样**得到 —— Base 按答案类型派生类（`enum / nec / pos / pos+nec / count / other`）分层，Hard 按子问题数桶分层。`scripts/split.py` 在全集上重跑可逐字节复现当前公开切分。

如需在私有留存集上提交模型评测（用于官方排行榜），请联系 <mingzhang23@m.fudan.edu.cn>。

## 👥 贡献

欢迎贡献！请直接提 issue 或 PR。

## 📮 联系我们

如有问题或建议，请：

- 在 GitHub 上提 issue

- 或直接联系作者：

  Ming Zhang: mingzhang23@m.fudan.edu.cn

## 📝 引用

如果本工作对你的研究有帮助，请引用：

```bibtex
@misc{zhang2026llmevallogic,
  title         = {{LLMEval-Logic}: A Solver-Verified Chinese Benchmark for Logical Reasoning of LLMs with Adversarial Hardening},
  author        = {Ming Zhang and Qiyuan Peng and Yinxi Wei and Yujiong Shen and Kexin Tan and Yuhui Wang and Zhenghao Xiang and Junjie Ye and Zhangyue Yin and Zhiheng Xi and Shihan Dou and Tao Gui and Maxm Pan and Ruizhi Yang and Qi Zhang and Xuanjing Huang},
  year          = {2026},
  eprint        = {2605.19597},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2605.19597}
}
```

## 🔗 相关项目

| 项目 | 简介 | 论文 | 代码 |
|------|------|------|------|
| **LLMEval-Fair**（ACL 2026 主会） | 鲁棒公平评测，覆盖 13 个学科、20 万+ 题目 | [arXiv](https://arxiv.org/abs/2508.05452) | [GitHub](https://github.com/llmeval/LLMEval-Fair) |
| **LLMEval-Med**（EMNLP 2025 Findings） | 经医生验证的临床大模型基准 | [arXiv](https://arxiv.org/abs/2506.04078) | [GitHub](https://github.com/llmeval/LLMEval-Med) |
| **LLMEval-2**（AAAI 2024） | 第二期：专业领域评测 | [arXiv](https://arxiv.org/abs/2312.07398) | [GitHub](https://github.com/llmeval/LLMEval-2) |
| **LLMEval-1**（AAAI 2024） | 第一期：通用能力评测 | [arXiv](https://arxiv.org/abs/2312.07398) | [GitHub](https://github.com/llmeval/LLMEval-1) |

完整项目列表与排行榜：[llmeval.com](https://llmeval.com/) · 所有数据集托管于 [🤗 llmeval-fdu](https://huggingface.co/llmeval-fdu)

---

<p align="center">
  <b>LLMEval</b> | 复旦大学自然语言处理实验室
</p>
