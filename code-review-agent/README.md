# AI Code Review & Debugging Agent (Local LLM)

A command-line tool that reviews Python code for bugs, logic errors, and edge cases — using a **local, offline LLM** (no cloud API, no API keys, no internet required after setup).

## Use Case

Get instant, private code reviews without sending code to any cloud service. Useful for:
- Quick sanity checks before committing code
- Learning to spot common bug patterns
- Reviewing code offline / on restricted networks

## How It Works

1. You point the tool at a Python file or a folder of files.
2. The tool reads the code and sends it, along with a detailed review prompt, to a local LLM running via **Ollama**.
3. The LLM (running entirely on your own machine) analyzes the code and returns:
   - Bugs found
   - A suggested fix
   - A plain-English explanation
4. Results are printed to the terminal and saved to `review_report.txt`.

No dataset or training is involved — this uses prompt engineering with a pre-trained open-source model, not a custom-trained ML model.

## Technologies Used

- **Python 3** — core script
- **Ollama** — runs the local LLM
- **Qwen2.5-Coder (3B)** — small, open-source, code-focused LLM
- **Requests** library — talks to Ollama's local REST API

## Setup

1. Install [Ollama](https://ollama.com/download)
2. Pull the model:
ollama pull qwen2.5-coder:3b

3. Install Python dependency:

pip install requests


## Usage

Review a single file:

python reviewer.py sample_code/buggy_example.py


Review an entire folder:

python reviewer.py sample_code

Output is printed to the terminal and saved to `review_report.txt`.

## Known Limitations

- Small local LLMs (1.5B-3B parameters) are **not always consistent** — the same code reviewed twice can give different quality results.
- Best suited for files under ~200 lines; very large files are skipped to avoid unreliable reviews.
- Not a replacement for a human reviewer or larger cloud-based models — this is optimized for speed, privacy, and offline use, not maximum accuracy.

## Possible Future Improvements

- Support for more languages beyond Python
- A simple web interface (e.g. Streamlit) instead of command-line only
- Option to run multiple review passes and combine results for higher reliability