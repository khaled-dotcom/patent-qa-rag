# Starts the Q&A API + website on http://localhost:8010  (API docs: http://localhost:8010/docs)
# LLM: set $env:GROQ_API_KEY = "gsk_..." first to use Groq; otherwise start_llm.ps1 (local llama.cpp) is used.
# Optional: require an API key for /api/* calls ->   $env:RAG_API_KEY = "choose-a-long-secret"
# Python: $env:RAG_PYTHON if set, else .venv\Scripts\python.exe, else D:\TDAC_local\rag_venv, else python on PATH.
Set-Location $PSScriptRoot
$py = "python"
foreach ($c in @($env:RAG_PYTHON, ".venv\Scripts\python.exe", "D:\TDAC_local\rag_venv\Scripts\python.exe")) {
    if ($c -and (Test-Path $c)) { $py = $c; break }
}
& $py -m uvicorn app:app --host 0.0.0.0 --port 8010
