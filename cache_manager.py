def setup_lmcache_environment(num_prompts, num_tokens):
    """
    Configure LMCache environment variables.
    Args:
        num_prompts: Number of prompts to process
        num_tokens: Number of tokens per prompt
    """
    cpu_size = num_prompts * num_tokens * 1.5 / 10000  # 1.5GB per 10000 tokens

    env_vars = {
        "LMCACHE_CHUNK_SIZE": "256",         # Set tokens per chunk
        "LMCACHE_LOCAL_CPU": "True",         # Enable local CPU backend
        #"LMCACHE_MAX_LOCAL_CPU_SIZE": str(cpu_size),  # Dynamic CPU memory limit (GB)
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "5.0",
        "LMCACHE_LOCAL_DISK": "file://$HOME/lmcache_storage",
        "LMCACHE_MAX_LOCAL_DISK_SIZE": "5.0",
        # Disable page cache
        # This should be turned on for better performance if most local CPU memory is used
        "LMCACHE_EXTRA_CONFIG": f'{'use_odirect': True}'
    }
    for key, value in env_vars.items():
        os.environ[key] = value