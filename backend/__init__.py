# Configure SSL verification and Hugging Face Hub before any HTTP clients load.
# Must run before importing target_schema or vector_store (which use SentenceTransformer).
from backend.config import HF_HUB_DISABLE_SSL_VERIFY, DISABLE_SSL_VERIFY

# Suppress InsecureRequestWarning when SSL verification is disabled
if DISABLE_SSL_VERIFY or HF_HUB_DISABLE_SSL_VERIFY:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if HF_HUB_DISABLE_SSL_VERIFY:
    import requests
    from huggingface_hub import configure_http_backend

    def _hf_backend_factory():
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend_factory=_hf_backend_factory)
