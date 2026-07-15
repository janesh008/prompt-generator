# LittleNest — OpenRouter / Qwen Prompt Generator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/ETSY-pipeline/blob/main/prompt-generator/LittleNest_Prompt_Generator.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Generates AI image-generation prompt batches for **LittleNest Etsy digital PNG clipart bundles** using [OpenRouter](https://openrouter.ai)'s OpenAI-compatible API with the `qwen/qwen3-5-flash` model.

---

## 📁 Files

| File | Purpose |
|---|---|
| `LittleNest_Prompt_Generator.ipynb` | **Start here** — step-by-step Colab notebook |
| `qwen_client.py` | OpenRouter HTTP client with retry logic |
| `littlenest_prompt_generator.py` | Locked system prompt + `PromptRequest` builder |
| `prompt_validator.py` | Structural validator and section parser |
| `.env.example` | Environment variable template (safe to commit) |
| `README.md` | This file |

---

## 🚀 Quickstart — Google Colab

1. Click the **Open in Colab** badge above
2. Run **Step 1** to mount your Google Drive
3. Run **Step 2** to install `requests`
4. Run **Step 3 (Option A or B)** to load the module
5. Run **Step 4** and paste your [OpenRouter API key](https://openrouter.ai/keys) in the hidden prompt
6. Edit the variables in **Step 6** (cartoon, theme, count)
7. Run **Steps 7 → 9** to generate, validate, and save

> The notebook is self-contained — each step explains what it does and why.

---

## 🔑 Environment Setup

Copy `.env.example` to your own shell session (never commit a filled-in `.env`):

```bash
cp prompt-generator/.env.example .env
# then edit .env with your real key
```

**Required:**

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

**Optional overrides:**

```bash
OPENROUTER_MODEL=qwen/qwen3-5-flash
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_TEMPERATURE=0.7
OPENROUTER_MAX_TOKENS=12000
OPENROUTER_HTTP_REFERER=https://colab.research.google.com
OPENROUTER_APP_TITLE=LittleNest Colab Prompt Generator
```

**In Colab (hidden prompt — recommended):**

```python
from qwen_client import set_openrouter_api_key
set_openrouter_api_key()   # password prompt; key goes into os.environ only
```

---

## 💻 CLI Usage (Node.js wrapper)

```powershell
node prompt-generator/generate-prompts.js `
  --cartoon "Lilo & Stitch" `
  --theme birthday `
  --count 130 `
  --out "D:\Janesh\ETSY\CrispPNGCo\lilo_stitch_prompts.txt"
```

With all options:

```powershell
node prompt-generator/generate-prompts.js `
  --cartoon "Minnie Mouse" `
  --theme "baby shower" `
  --style "soft watercolor" `
  --sections "full bundle plus alphabet and banner" `
  --count 180 `
  --reference "pastel pink palette, compact chibi proportions, gentle painterly edges" `
  --out "D:\Janesh\ETSY\CrispPNGCo\minnie_baby_shower_prompts.txt"
```

---

## 🐍 Python API

### Generating a batch

```python
import sys
sys.path.append("/content/drive/MyDrive/ETSY/ETSY-pipeline/prompt-generator")

from qwen_client import set_openrouter_api_key
from littlenest_prompt_generator import PromptRequest, generate_littlenest_prompts, save_prompt_batch

set_openrouter_api_key()   # hidden prompt

request = PromptRequest(
    cartoon="Lilo & Stitch",
    theme="birthday",
    prompt_count=130,
)

prompt_text = generate_littlenest_prompts(request)
```

### Validating the output

```python
from prompt_validator import validate_prompt_batch

validation = validate_prompt_batch(prompt_text)
print(validation.ok)           # True / False
print(validation.issues)       # list of structural problems
print(validation.section_counts)  # {section_name: int}

validation.raise_if_invalid()  # raises ValueError if not ok
```

### Saving to Drive

```python
save_prompt_batch(
    prompt_text,
    "/content/drive/MyDrive/ETSY/output/lilo_stitch_prompts.txt",
)
```

### Parsing prompts section-by-section

```python
from prompt_validator import parse_prompt_batch

prompts_by_section = parse_prompt_batch(prompt_text)
print(prompts_by_section["MAIN_CHARACTER"][0])
```

### Overriding model / temperature at runtime

```python
import os
os.environ["OPENROUTER_MODEL"]       = "qwen/qwen3-5-flash"
os.environ["OPENROUTER_TEMPERATURE"] = "0.75"
os.environ["OPENROUTER_MAX_TOKENS"]  = "14000"
```

---

## ✅ Validation Rules

The validator checks:

- All **19 locked section headings** are present in the correct order
- Active sections have **≥ 10 prompts** each
- Inactive sections contain exactly `(not applicable for this roster)`
- No **code fences** (` ``` `) or code-style wrapper variables appear
- Always-active sections (`MAIN_CHARACTER`, `PATTERN`, `PROP`, `SCENE`) are not marked inactive

> Validation is intentionally conservative — it catches structure problems but cannot verify character roster accuracy. Always review the first and last prompt in each `SUB_CHARACTER_N` section before feeding a batch into your image pipeline.

---

## 🔒 Security

- **Never** hardcode your API key in a cell or script
- **Never** commit a filled-in `.env` — add `.env` to your `.gitignore`
- In shared Colabs, use [Colab Secrets](https://colab.research.google.com/drive/1UOVq1VcP12Y1kDXVwcuJJJRtmDAJ_0aB):

  ```python
  from google.colab import userdata
  import os
  os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")
  ```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `requests` | ≥ 2.28 | HTTP client for OpenRouter API |

Install: `pip install requests`

Everything else is Python standard library (3.8+).
