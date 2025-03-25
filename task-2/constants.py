import os

AZURE_REPO_URL = "Azure/AzurePublicDataset"

AZURE_DATASET_PATHS = {
    "code": "data/AzureLLMInferenceTrace_code.csv",
    "conv": "data/AzureLLMInferenceTrace_conv.csv",
}

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "none"

DEFAULT_SUBPROCESSES = 1

DEFAULT_MESSAGES = "Tell me a story"

DEFAULT_DATASET_FOLDER = os.path.expanduser("~/.cache/tracestorm")