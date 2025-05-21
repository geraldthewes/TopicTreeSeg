# TreeSeg: Hierarchical Topic Segmentation of Large Transcripts

- Package Implementation of "TreeSeg: Hierarchical Topic Segmentation of Large Transcripts"
- Paper: https://arxiv.org/abs/2407.12028

TreeSeg is an algorithm for segmenting large meetings transcripts in a hierarchical manner using embeddings and divisive clustering. It produces a binary tree of segments by recursively splitting segments into sub-segments. The approach is completely unsupervised. While this implementation is using OpenAI's embeddings model, any embeddings model can be used as an alternative.

Forked from:  https://github.com/AugmendTech/treeseg.git

## How to use

```
   transcript = [
    {'speaker': 'Alice', 'composite': 'Okay team, let\'s kick off the weekly sync. First agenda item is the Q3 roadmap planning.'},
    {'speaker': 'Bob', 'composite': 'Right. I\'ve drafted the initial proposal based on the feedback from the product team.'},
    {'speaker': 'Alice', 'composite': 'Great. Can you share the highlights? We need to finalize the key initiatives this week.'},
    {'speaker': 'Bob', 'composite': 'Sure. The main pillars are customer acquisition, platform stability, and launching the new mobile feature.'},
    {'speaker': 'Charlie', 'composite': 'On platform stability, I wanted to raise an issue regarding the recent deployment.'},
    {'speaker': 'Charlie', 'composite': 'We saw a spike in error rates after the update went live Tuesday.'},
    {'speaker': 'Alice', 'composite': 'Okay, thanks Charlie. Let\'s make that the next discussion point after Bob finishes the roadmap overview.'},
    {'speaker': 'Bob', 'composite': 'Okay, back to the roadmap. For customer acquisition, we\'re planning two major campaigns...'}
    # ... more utterances
]
```

```
    EMBEDDINGS_HEADERS = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY"),
    }
```
```
    config =   {
        "MIN_SEGMENT_SIZE": 2,
        "LAMBDA_BALANCE": 0,
        "UTTERANCE_EXPANSION_WIDTH": 2,
        "EMBEDDINGS": {
            "embeddings_func": ollama_embeddings, # openai_embeddings
            "headers": "", # For OpenAI EMBEDDINGS_HEADERS
            "model": "nomic-embed-text",  # or "text-embedding-ada-002" for openai
            "endpoint": os.getenv("OLLAMA_HOST","")   # "https://api.openai.com/v1/embeddings"
        },
		"TEXT_KEY": "composite"
    }

    segmenter = TreeSeg(configs=config, entries=transcript)

    segments = segmenter.segment_meeting(3)
```
    
