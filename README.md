# Sovereign AI Workbench

It lets organizations automate document-based workflows using local AI, so they can benefit from AI without sending sensitive business data to cloud services.

The Sovereign AI Workbench is a local-first application for document analysis and finance-compliance workflows. It can accept natural-language requests, create and validate a workflow plan, and execute approved steps. In document-analysis mode, it extracts requirements from a company SOP, checks a quarterly financial report against those requirements, and generates a gap register. It also includes a general-chat mode for local questions.

## How it works

- **Local model inference:** The application uses Ollama and the `gemma4:e4b` model for planning and language tasks.
- **Document workflow:** Provide an SOP PDF and a quarterly-report PDF for compliance analysis.
- **Plan, validate, execute:** Proposed workflow steps are validated before execution; interactive mode asks for confirmation.
- **Audit trail:** Workflow activity is recorded in a local `audit.jsonl` file.
- **Web interface:** Start the Streamlit interface with `streamlit run app.py`. The terminal interface is available with `python main.py`.

Install the application's Python dependencies, install and start [Ollama](https://ollama.com/), and make the model available before running the app:

```bash
ollama pull gemma4:e4b
streamlit run app.py
```

The application is designed to run inference locally. Keep Ollama pointed at your local server if you need to ensure prompts and document text are not sent to a hosted inference service. Local inference does not by itself secure the host, uploaded files, or generated outputs.

## Gemma 4 model license

This project uses Google's Gemma 4 model through Ollama (`gemma4:e4b`). Google publishes Gemma 4 under the **Apache License, Version 2.0**. The license applies to the Gemma model; it does not establish the license for this application's source code or other dependencies.

Review the [official Gemma 4 license](https://ai.google.dev/gemma/apache_2) and the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) for the applicable terms, including requirements that may apply when redistributing the model or derivatives. See the [Ollama Gemma 4 library page](https://ollama.com/library/gemma4) for the Ollama model distribution.
