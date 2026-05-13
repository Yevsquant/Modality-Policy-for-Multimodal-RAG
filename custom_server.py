# custom_server.py
import vllm_ext  # This triggers _register_pruned_model() and _register_pruned_processor()
import sys
import runpy

if __name__ == "__main__":
    # Start the standard vLLM OpenAI API server
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')