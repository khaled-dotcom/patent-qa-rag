# Optional, only when NOT using Groq: local LLM (Llama 3.2 3B, 4-bit GGUF) on http://127.0.0.1:8081 via llama.cpp.
# Set $env:LLAMA_SERVER and $env:LLAMA_MODEL to your llama-server.exe and .gguf file.
$server = if ($env:LLAMA_SERVER) { $env:LLAMA_SERVER } else { "D:\TDAC_local\llama\llama-server.exe" }
$model  = if ($env:LLAMA_MODEL)  { $env:LLAMA_MODEL }  else { "D:\TDAC_Models\base_models_4bit_gguf\Llama-3.2-3B-Instruct-Q4_K_M.gguf" }
& $server -m $model -c 8192 -ngl 99 -np 1 -fa on --host 127.0.0.1 --port 8081
