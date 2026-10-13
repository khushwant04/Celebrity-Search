# Celebrity Search (LangChain + Ollama)

A small Streamlit app that chains three prompts to a local Llama 2 model (via Ollama) to describe a
person, find their date of birth, and list major world events from around that date.

## How it works

`main.py` builds a LangChain `SequentialChain` of three `LLMChain`s, each with its own
`ConversationBufferMemory`:

```
name   -> "Tell me about {name}"                          -> person
person -> "When was {person} born"                        -> dob
dob    -> "Mention 5 major events happened around {dob}"  -> description
```

The Streamlit page shows the chain output plus two expanders with the memory buffers.

`example.py` is a minimal single-prompt Streamlit demo against the same model, and
`langchain.ipynb` contains a few one-off `llm.predict(...)` calls.

## Project structure

```
main.py            # Celebrity search app (sequential chain)
example.py         # Minimal Ollama + Streamlit demo
langchain.ipynb    # Scratch notebook
requirements.txt
```

## Getting started

Prerequisites: Python 3.9+ and [Ollama](https://ollama.com) running locally on the default
port (`http://localhost:11434`) with the `llama2` model pulled.

```bash
ollama pull llama2

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

streamlit run main.py      # celebrity search
streamlit run example.py   # simple prompt demo
```

Type a name into the text box and the app runs the three chained prompts.

## Notes

- Uses the legacy `LLMChain` / `SequentialChain` / `ConversationBufferMemory` APIs, which were
  removed from the `langchain` package in 1.0, so `requirements.txt` pins `langchain<1.0`.
- No API keys are needed; everything runs against the local Ollama server. The model name and base
  URL are hardcoded in `main.py` and `example.py`.

## License

Apache License 2.0 - see [LICENSE](LICENSE).
