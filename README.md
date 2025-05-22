# TopicTreeSeg: Hierarchical Topic Segmentation of Large Transcripts

- Package Implementation of "TreeSeg: Hierarchical Topic Segmentation of Large Transcripts"
- Paper: https://arxiv.org/abs/2407.12028
- Original Fork: https://github.com/AugmendTech/treeseg.git (This package aims to be a distributable version)

TopicTreeSeg is an algorithm for segmenting large meeting transcripts or other long texts in a hierarchical manner using embeddings and divisive clustering. It produces a binary tree of segments by recursively splitting segments into sub-segments. The approach is completely unsupervised and can work with various embedding models.

## Installation

```bash
pip install TopicTreeSeg
```

## How to use

First, ensure you have your transcript in the required format, which is a list of dictionaries. Each dictionary should contain at least a key with the text of an utterance. By default, this key is `"composite"`, but you can configure it.

```python
import os
from TopicTreeSeg import TreeSeg, ollama_embeddings, openai_embeddings # Make sure these are importable

# Example transcript
transcript = [
    {'speaker': 'Alice', 'composite': "Okay team, let's kick off the weekly sync. First agenda item is the Q3 roadmap planning."},
    {'speaker': 'Bob', 'composite': "Right. I've drafted the initial proposal based on the feedback from the product team."},
    {'speaker': 'Alice', 'composite': "Great. Can you share the highlights? We need to finalize the key initiatives this week."},
    {'speaker': 'Bob', 'composite': "Sure. The main pillars are customer acquisition, platform stability, and launching the new mobile feature."},
    {'speaker': 'Charlie', 'composite': "On platform stability, I wanted to raise an issue regarding the recent deployment."},
    {'speaker': 'Charlie', 'composite': "We saw a spike in error rates after the update went live Tuesday."},
    {'speaker': 'Alice', 'composite': "Okay, thanks Charlie. Let's make that the next discussion point after Bob finishes the roadmap overview."},
    {'speaker': 'Bob', 'composite': "Okay, back to the roadmap. For customer acquisition, we're planning two major campaigns..."}
    # ... more utterances
]
```

You can configure TreeSeg to use different embedding providers. Below are examples for Ollama and OpenAI.

### Using Ollama Embeddings

Ensure the `OLLAMA_HOST` environment variable is set to your Ollama API endpoint (e.g., `http://localhost:11434` if running locally).

```python
# Configuration for Ollama
ollama_config = {
    "MIN_SEGMENT_SIZE": 2,
    "LAMBDA_BALANCE": 0, # Adjust for segment balance if needed
    "UTTERANCE_EXPANSION_WIDTH": 2, # How many previous utterances to include for context
    "EMBEDDINGS": {
        "embeddings_func": ollama_embeddings,
        "headers": {}, # Ollama client typically doesn't need special headers
        "model": "nomic-embed-text",  # Or any other model supported by your Ollama instance
        "endpoint": os.getenv("OLLAMA_HOST") # e.g., "http://localhost:11434"
    },
    "TEXT_KEY": "composite" # Key in your transcript entries that contains the text
}

# Initialize TreeSeg with Ollama config
segmenter_ollama = TreeSeg(configs=ollama_config, entries=transcript)

# Perform segmentation (e.g., to get 3 segments)
# Note: The number of segments is a target, actual segments might vary based on data.
segments_ollama = segmenter_ollama.segment_meeting(K=3) 

print("Segments (Ollama):", segments_ollama)
# segments_ollama will be a list of 0s and 1s indicating segment boundaries.
# A '1' indicates the start of a new segment.
```

### Using OpenAI Embeddings

Ensure the `OPENAI_API_KEY` environment variable is set with your OpenAI API key.

```python
# Configuration for OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

EMBEDDINGS_HEADERS_OPENAI = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}",
}

openai_config = {
    "MIN_SEGMENT_SIZE": 2,
    "LAMBDA_BALANCE": 0,
    "UTTERANCE_EXPANSION_WIDTH": 2,
    "EMBEDDINGS": {
        "embeddings_func": openai_embeddings,
        "headers": EMBEDDINGS_HEADERS_OPENAI,
        "model": "text-embedding-ada-002", # Or other OpenAI embedding models
        "endpoint": "https://api.openai.com/v1/embeddings"
    },
    "TEXT_KEY": "composite"
}

# Initialize TreeSeg with OpenAI config
segmenter_openai = TreeSeg(configs=openai_config, entries=transcript)

# Perform segmentation
segments_openai = segmenter_openai.segment_meeting(K=3)

print("Segments (OpenAI):", segments_openai)
```
