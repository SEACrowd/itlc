# Inference-Time Language Control (ITLC)

Inference-Time Language Control (ITLC) is a novel method designed to enhance cross-lingual language control and mitigate language confusion in Large Language Models (LLMs) that is introduced in ["Language Surgery in Multilingual Large Language Models"](). ITLC leverages latent injection to enable precise manipulation of language-specific information during inference, while preserving semantic integrity.

ITLC addresses the challenge of language confusion in LLMs, which can lead to inconsistent language generation. By exploiting naturally emerging representation alignment in the middle layers of LLMs, ITLC disentangles language-specific and language-agnostic information. This allows for effective control over the generated language without compromising semantic meaning.

The key contributions of ITLC and the corresponding ["Language Surgery in Multilingual Large Language Models"]() paper include:
- Confirming the presence of representation alignment in LLMs and analyzing its behavior.
- Demonstrating a method to extract and manipulate language-specific information.
- Introducing a practical solution for cross-lingual language control and language confusion mitigation.

The method has been experimentally validated, showing strong cross-lingual control capabilities and effectiveness in reducing language confusion. ITLC is open-source, providing a valuable tool for improving the multilingual performance of LLMs.

## Getting Started

To use ITLC, follow these steps:

1. **Installation**: Clone the repository and install dependencies.
2. **Data Preparation**: Prepare your dataset in the required format.
3. **Model Setup**: Configure the model and ITLC parameters.
4. **Inference**: Run the ITLC method on your data.

### Installation

To install ITLC, follow these steps:

1. Clone the repository: `git clone https://github.com/SEACrowd/itlc.git`
2. Install the required dependencies: `pip install -r requirements.txt`

### Usage

ITLC can be used to control the language of generated text in LLMs. Here's an example of how to use ITLC to generate text in a specific language:

```python
from itlc import ITLC

# Initialize the ITLC model
itlc = ITLC(model_name='qwen2.5-0.5b-instruct')

# Set the target language
target_language = 'Indonesian'

# Generate text in the target language
generated_text = itlc.generate(prompt='Translate the following English text to Indonesian: Hello, how are you?', target_language=target_language)

print(generated_text)
```

You can check moew examples of ITLC on the example script: [`itlc_example.py`](./itlc_example.py)

## Methodology & Implication

**Inference-Time Language Control (ITLC)** is a novel method proposed in the paper "Language Surgery in Multilingual Large Language Models" that leverages latent injection to enable precise cross-lingual language control and mitigate language confusion in large language models (LLMs). ITLC builds on the naturally emerging representation alignment in LLMs, particularly in the middle layers, to disentangle language-specific and language-agnostic information.

### Core Components

ITLC consists of several core components that work together to achieve language control:

- **Latent Extraction**: Isolates language-specific information from the model's representations by extracting hidden states at the middle layer.
- **Linear Discriminant Analysis (LDA)**: Disentangles language-specific information by maximizing class separability and reducing dimensionality.
- **Language Vector Construction**: Constructs language vectors by leveraging the neural network's weights to identify active dimensions for each language.
- **Vector Injection**: Projects the language vector back to the original embedding space and injects it into the hidden states during inference.
- **Language Shift Strategy**: Divides the language vector injection into three strategies based on the temporal scope of application: prompt only, generated tokens only, and both phases.

### Application

ITLC has several applications in natural language processing:

- **Cross-Lingual Language Control**: Enables zero-shot cross-lingual language generation by controlling the language of generated text.
- **Mitigating Language Confusion**: Alleviates the cross-lingual language confusion problem in LLMs, leading to more consistent language generation.
- **Language-Specific Manipulation**: Allows for language-specific manipulation while preserving the semantic integrity of the generation.

### Results

#### Cross-Lingual Language Control

ITLC demonstrates strong cross-lingual control capabilities, as shown in the following table. Compared to baseline monolingual prompting, ITLC achieves similar to slightly higher generation performance across various target languages, as measured by BLEU and BERTScore metrics. This indicate that latent intervention in ITLC maintain semantic language-agnostic information.

| Target Language | Baseline BLEU | ITLC BLEU | Baseline BERTScore | ITLC BERTScore |
| --- | --- | --- | --- | --- |
| ID | 19.29 | 14.3 | 62.9 | 63.6 |
| TH | 0.0 | 15.97 | 62.8 | 64.1 |
| TR | 6.05 | 15.97 | 60.7 | 60.2 |
| JA | 0.0 | 15.97 | 62.0 | 60.2 |
| FR | 7.78 | 10.97 | 63.3 | 63.2 |
| ES | 10.88 | 7.17 | 64.0 | 64.4 |
| AR | 7.13 | 11.88 | 63.8 | 65.5 |
| ZH | 0.0 | 0.0 | 63.6 | 62.0 |
| KO | 7.54 | 4.11 | 63.1 | 63.8 |
| AVG | 6.52 | 10.70 | 62.91 | 63.00 |

#### Mitigating Language Confusion

ITLC effectively mitigates language confusion in LLMs, as demonstrated in Table 2. The method significantly improves Language Confusion Precision Rate (LCPR), Language Precision Rate (LPR), and Word Precision Rate (WPR) compared to baseline zeros-shot and 5-shot ICL.

| Method | LCPR | LPR | WPR |
| --- | --- | --- | --- |
| Baseline | 29.41 | 19.75 | 73.45 |
| \+ Q/A template (0-shot) | 44.68 | 35.36 | 75.94 |
| \+ 5-shot | 56.78 | 50.63 | 76.16 |
| \+ ITLC (prompt-only, α=0.8) | 65.71 | 66.41 | 74.24 |
| \+ ITLC (gen-only, α=0.6) | 71.35 | 80.46 | 67.67 |
| \+ ITLC (prompt-and-gen, α=0.5) | 78.93 | 85.08 | 77.15 |  

## Citation
If you are using or develop method inspired by ITLC, please cite the following publication:

```
@article{lopo2025itlc,
      title={Language Surgery in Multilingual Large Language Models}, 
      author={Joanito Agili Lopo and Muhammad Ravi Shulthan Habibi and Tack Hwa Wong and Ghozali and Fajri Koto and Genta Indra Winata and Peerat Limkonchotiwat and Alham Fikri Aji and Samuel Cahyawijaya},
      year={2025},
      eprint={TODO},
      journal={TODO}
}
```
