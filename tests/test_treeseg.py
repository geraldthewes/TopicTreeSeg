import unittest
import numpy as np
from TopicTreeSeg.treeseg import SegNode # Assuming TopicTreeSeg is installed or in PYTHONPATH
from TopicTreeSeg import TreeSeg # Added for TestTreeSeg

# Mock configurations for testing
MOCK_CONFIGS = {
    "MIN_SEGMENT_SIZE": 2,
    "LAMBDA_BALANCE": 0.0, # Using 0.0 for simpler likelihood calculation in tests
    "UTTERANCE_EXPANSION_WIDTH": 0, # Not directly used by SegNode, but part of typical configs
    "EMBEDDINGS": { # Not directly used by SegNode if embeddings are pre-supplied
        "embeddings_func": None,
        "model": "mock-model",
        "endpoint": "mock-endpoint"
    },
    "TEXT_KEY": "text"
}

# Mock entries - list of dictionaries, each with an 'index' and 'embedding'
MOCK_ENTRIES_SIMPLE = [
    {"index": 0, "embedding": np.array([0.1, 0.2]), "text": "utt1"},
    {"index": 1, "embedding": np.array([0.3, 0.4]), "text": "utt2"},
    {"index": 2, "embedding": np.array([0.7, 0.8]), "text": "utt3"},
    {"index": 3, "embedding": np.array([0.9, 1.0]), "text": "utt4"},
]

# More complex entries for testing split logic
MOCK_ENTRIES_SPLITTABLE = [
    {"index": i, "embedding": np.array([i*0.1, i*0.1 + 0.05]), "text": f"utt{i}"} for i in range(10)
    # Creates 10 utterances, e.g. [0, 0.05], [0.1, 0.15], ..., [0.9, 0.95]
]


class TestSegNode(unittest.TestCase):

    def test_segnode_initialization_basic(self):
        node = SegNode(identifier="*", entries=MOCK_ENTRIES_SIMPLE, configs=MOCK_CONFIGS)
        self.assertEqual(node.identifier, "*")
        self.assertEqual(len(node.entries), len(MOCK_ENTRIES_SIMPLE))
        self.assertTrue(np.array_equal(node.segment, [0, 1, 2, 3]))
        self.assertIsNotNone(node.mu)
        # For MOCK_ENTRIES_SIMPLE (4 entries) < 2 * MIN_SEGMENT_SIZE (2*2=4), it should be a leaf.
        # Let's adjust MIN_SEGMENT_SIZE or entries for a non-leaf test.
        
        configs_leaf = MOCK_CONFIGS.copy()
        configs_leaf["MIN_SEGMENT_SIZE"] = 3 # 4 entries < 2 * 3 = 6, so it's a leaf
        node_leaf = SegNode(identifier="L", entries=MOCK_ENTRIES_SIMPLE, configs=configs_leaf)
        self.assertTrue(node_leaf.is_leaf)

        configs_non_leaf = MOCK_CONFIGS.copy()
        configs_non_leaf["MIN_SEGMENT_SIZE"] = 2 # 4 entries >= 2 * 2 = 4, so it's not a leaf
        node_non_leaf = SegNode(identifier="NL", entries=MOCK_ENTRIES_SIMPLE, configs=configs_non_leaf)
        self.assertFalse(node_non_leaf.is_leaf)


    def test_segnode_compute_likelihood(self):
        node = SegNode(identifier="*", entries=MOCK_ENTRIES_SIMPLE, configs=MOCK_CONFIGS)
        # Expected mu: mean of [[0.1,0.2], [0.3,0.4], [0.7,0.8], [0.9,1.0]]
        # mu = [ (0.1+0.3+0.7+0.9)/4, (0.2+0.4+0.8+1.0)/4 ] = [2.0/4, 2.4/4] = [0.5, 0.6]
        expected_mu = np.array([[0.5, 0.6]])
        self.assertTrue(np.allclose(node.mu, expected_mu))

        # squared_error = sum of squared distances from mu
        # e0 = [0.1,0.2]-[0.5,0.6] = [-0.4, -0.4], sq_err0 = (-0.4)^2 + (-0.4)^2 = 0.16 + 0.16 = 0.32
        # e1 = [0.3,0.4]-[0.5,0.6] = [-0.2, -0.2], sq_err1 = (-0.2)^2 + (-0.2)^2 = 0.04 + 0.04 = 0.08
        # e2 = [0.7,0.8]-[0.5,0.6] = [ 0.2,  0.2], sq_err2 = ( 0.2)^2 + ( 0.2)^2 = 0.04 + 0.04 = 0.08
        # e3 = [0.9,1.0]-[0.5,0.6] = [ 0.4,  0.4], sq_err3 = ( 0.4)^2 + ( 0.4)^2 = 0.16 + 0.16 = 0.32
        # total_sq_err = 0.32 + 0.08 + 0.08 + 0.32 = 0.80
        # NLL = total_sq_err + LAMBDA_BALANCE * N^2 = 0.80 + 0.0 * 4^2 = 0.80
        self.assertAlmostEqual(node.negative_log_likelihood, 0.80, places=5)

    def test_segnode_optimize_split_simple(self):
        # Need enough entries to not be a leaf and allow splitting
        # MOCK_ENTRIES_SIMPLE has 4 entries. MIN_SEGMENT_SIZE = 2. Not a leaf.
        node = SegNode(identifier="*", entries=MOCK_ENTRIES_SIMPLE, configs=MOCK_CONFIGS)
        self.assertFalse(node.is_leaf) # Should not be a leaf
        
        # With MIN_SEGMENT_SIZE = 2, N=4.
        # Possible splits (n indicates end of first segment):
        # n=0: X0=[e0], X1=[e1,e2,e3] -> too small (M0=1 < MIN_SEGMENT_SIZE=2) - loop starts at MIN_SEGMENT_SIZE-1 = 1
        # n=1: X0=[e0,e1], X1=[e2,e3] -> M0=2, M1=2. This is the only valid split point.
        # n=2: X0=[e0,e1,e2], X1=[e3] -> too small (M1=1 < MIN_SEGMENT_SIZE=2) - loop ends at N-MIN_SEGMENT_SIZE-1 = 4-2-1 = 1
        
        # So, the only split point considered should be after index 1.
        # X0_embs = [[0.1,0.2], [0.3,0.4]] -> mu0 = [0.2, 0.3]
        #   err0_0 = [-0.1,-0.1], sq_err0_0 = 0.01+0.01=0.02
        #   err0_1 = [0.1,0.1], sq_err0_1 = 0.01+0.01=0.02. Total = 0.04
        # X1_embs = [[0.7,0.8], [0.9,1.0]] -> mu1 = [0.8, 0.9]
        #   err1_0 = [-0.1,-0.1], sq_err1_0 = 0.01+0.01=0.02
        #   err1_1 = [0.1,0.1], sq_err1_1 = 0.01+0.01=0.02. Total = 0.04
        # Total loss = 0.04 + 0.04 = 0.08
        # (LAMBDA_BALANCE is 0, so no balance penalty)
        
        self.assertAlmostEqual(node.split_negative_log_likelihood, 0.08, places=5)
        self.assertEqual(len(node.split_entries[0]), 2) # First part of split
        self.assertEqual(len(node.split_entries[1]), 2) # Second part of split
        self.assertEqual(node.split_entries[0][0]["index"], 0)
        self.assertEqual(node.split_entries[0][1]["index"], 1)
        self.assertEqual(node.split_entries[1][0]["index"], 2)
        self.assertEqual(node.split_entries[1][1]["index"], 3)

    def test_segnode_split_loss_delta(self):
        node = SegNode(identifier="*", entries=MOCK_ENTRIES_SIMPLE, configs=MOCK_CONFIGS)
        # NLL = 0.80 (from previous test)
        # split_NLL = 0.08 (from previous test)
        # delta = 0.08 - 0.80 = -0.72
        expected_delta = node.split_negative_log_likelihood - node.negative_log_likelihood
        self.assertAlmostEqual(node.split_loss_delta(), expected_delta, places=5)
        self.assertAlmostEqual(node.split_loss_delta(), -0.72, places=5)

    def test_segnode_split(self):
        node = SegNode(identifier="*", entries=MOCK_ENTRIES_SIMPLE, configs=MOCK_CONFIGS)
        self.assertIsNone(node.left)
        self.assertIsNone(node.right)

        left_child, right_child = node.split()

        self.assertIsNotNone(node.left)
        self.assertIsNotNone(node.right)
        self.assertEqual(left_child.identifier, "*L")
        self.assertEqual(right_child.identifier, "*R")
        self.assertEqual(len(left_child.entries), 2)
        self.assertEqual(len(right_child.entries), 2)
        self.assertEqual(left_child.entries[0]["index"], 0)
        self.assertEqual(right_child.entries[0]["index"], 2)
        
        # Children of this split should be leaves as len(entries)=2 which is not < 2*MIN_SEGMENT_SIZE (2*2=4) is false.
        # len(entries) = 2.  2 * MIN_SEGMENT_SIZE = 2 * 2 = 4.
        # is_leaf condition: len(entries) < 2 * self.MIN_SEGMENT_SIZE
        # For children: 2 < 4 is True. So they should be leaves.
        self.assertTrue(left_child.is_leaf)
        self.assertTrue(right_child.is_leaf)

# Mock embedding function for TreeSeg tests
async def mock_embeddings_func(config, chunks):
    # Simulate embedding generation
    # For simplicity, let's say each embedding is the length of the chunk repeated, as a list/np.array
    # And each embedding vector has a fixed dimension, e.g., 2
    embeddings = []
    for i, chunk_text in enumerate(chunks):
        # Create a pseudo-random but deterministic embedding based on chunk index or content
        # For these tests, let's use a simple scheme: [index*0.1, index*0.1 + 0.01]
        # This matches the structure of MOCK_ENTRIES_SPLITTABLE for consistency if needed,
        # but here it's based on the order of chunks passed.
        embeddings.append(np.array([float(i) * 0.01, float(i) * 0.01 + 0.005]))
    return embeddings

class TestTreeSeg(unittest.TestCase):
    def setUp(self):
        self.transcript_simple = [
            {"composite": "Hello world", "speaker": "A"},
            {"composite": "This is a test", "speaker": "B"},
            {"composite": "Another sentence here", "speaker": "A"},
            {"composite": "Final utterance", "speaker": "C"},
        ]
        self.transcript_empty = []
        self.transcript_too_small = [
             {"composite": "Tiny transcript", "speaker": "A"}
        ]

        self.base_configs = {
            "MIN_SEGMENT_SIZE": 1, # Small for testing basic splits
            "LAMBDA_BALANCE": 0.0,
            "UTTERANCE_EXPANSION_WIDTH": 0, # No expansion for simpler block checking
            "EMBEDDINGS": {
                "embeddings_func": mock_embeddings_func, # Use our mock
                "headers": {},
                "model": "mock-model",
                "endpoint": "mock-endpoint"
            },
            "TEXT_KEY": "composite"
        }

    def test_treeseg_initialization_and_extract_blocks_simple(self):
        configs = self.base_configs.copy()
        segmenter = TreeSeg(configs=configs, entries=self.transcript_simple)
        
        self.assertEqual(len(segmenter.blocks), len(self.transcript_simple))
        for i, entry in enumerate(self.transcript_simple):
            self.assertEqual(segmenter.blocks[i]["convo"], entry["composite"])
            self.assertEqual(segmenter.blocks[i]["index"], i)
            self.assertEqual(segmenter.blocks[i]["speaker"], entry["speaker"])
            self.assertTrue("embedding" in segmenter.blocks[i])
            # Check if embedding looks like what mock_embeddings_func would produce
            # Given UTTERANCE_EXPANSION_WIDTH = 0
            expected_embedding = np.array([float(i) * 0.01, float(i) * 0.01 + 0.005])
            self.assertTrue(np.allclose(segmenter.blocks[i]["embedding"], expected_embedding))


    def test_treeseg_extract_blocks_with_expansion(self):
        configs = self.base_configs.copy()
        configs["UTTERANCE_EXPANSION_WIDTH"] = 1
        
        segmenter = TreeSeg(configs=configs, entries=self.transcript_simple)
        self.assertEqual(len(segmenter.blocks), len(self.transcript_simple))
        
        # Expected convo with expansion width 1:
        # Block 0: utt0
        # Block 1: utt0
#utt1
        # Block 2: utt1
#utt2
        # Block 3: utt2
#utt3
        
        expected_convos = [
            self.transcript_simple[0]["composite"],
            self.transcript_simple[0]["composite"] + "\n" + self.transcript_simple[1]["composite"],
            self.transcript_simple[1]["composite"] + "\n" + self.transcript_simple[2]["composite"],
            self.transcript_simple[2]["composite"] + "\n" + self.transcript_simple[3]["composite"],
        ]
        
        for i, block in enumerate(segmenter.blocks):
            self.assertEqual(block["convo"], expected_convos[i])
            self.assertTrue("embedding" in block) # Embeddings should still be generated

    def test_treeseg_empty_transcript(self):
        configs = self.base_configs.copy()
        segmenter = TreeSeg(configs=configs, entries=self.transcript_empty)
        self.assertEqual(len(segmenter.blocks), 0)
        
        # segment_meeting should handle empty blocks gracefully
        segments = segmenter.segment_meeting(K=3)
        self.assertEqual(segments, []) # Or appropriate response for empty

    def test_treeseg_transcript_too_small_for_segmentation(self):
        # MIN_SEGMENT_SIZE = 1. A single utterance is >= 2 * MIN_SEGMENT_SIZE (1*1=1) is false.
        # So it should be a leaf.
        # Let's make MIN_SEGMENT_SIZE = 1, so 1 entry is NOT < 2*1. Root node is not leaf.
        # But if K=1, it should just return one segment.
        configs = self.base_configs.copy() # MIN_SEGMENT_SIZE is 1
        
        segmenter = TreeSeg(configs=configs, entries=self.transcript_too_small)
        self.assertEqual(len(segmenter.blocks), 1)
        
        # If K=1, should return a single segment.
        segments_k1 = segmenter.segment_meeting(K=1)
        self.assertEqual(len(segments_k1), 1) # One entry means one segment marker
        self.assertEqual(segments_k1, [0]) # Typically first segment starts with 0 if not empty

        # If K > 1, but cannot split further (root is leaf or becomes leaf quickly)
        # Root node: entries=1. MIN_SEGMENT_SIZE=1. is_leaf: 1 < 2*1 is True. So root is leaf.
        segments_k3 = segmenter.segment_meeting(K=3)
        # If root is leaf, it means no splits possible. It returns one segment.
        self.assertEqual(len(segments_k3), 1) 
        self.assertEqual(segments_k3, [0])


    def test_treeseg_segment_meeting_simple_split(self):
        # Use MOCK_ENTRIES_SIMPLE which has 4 utterances.
        # Configs: MIN_SEGMENT_SIZE=1. UTTERANCE_EXPANSION_WIDTH=0.
        # This means each utterance is its own block for embedding.
        # Embeddings from mock_embeddings_func:
        # b0: [0.00, 0.005]
        # b1: [0.01, 0.015]
        # b2: [0.02, 0.025]
        # b3: [0.03, 0.035]
        
        configs = self.base_configs.copy()
        configs["MIN_SEGMENT_SIZE"] = 1 # Allows splitting down to single utterances
        
        # Need to provide entries that mock_embeddings_func will process
        # The mock_embeddings_func assigns embeddings based on index in the list of chunks
        # TreeSeg.blocks will be the entries for SegNode
        
        # Let's use transcript_simple (4 entries)
        segmenter = TreeSeg(configs=configs, entries=self.transcript_simple)
        
        # With K=2, we expect one split.
        # Root node NLL (calculated similarly to TestSegNode, but with new embeddings)
        # Split NLL
        # If a split occurs, it will be at some point.
        # The mock embeddings are linearly increasing, so a split in the middle is likely.
        
        segments_k2 = segmenter.segment_meeting(K=2)
        self.assertIsNotNone(segments_k2)
        self.assertEqual(len(segments_k2), len(self.transcript_simple)) # Transitions for each utterance
        self.assertEqual(segments_k2[0], 0) # First segment always starts at 0 (unless empty)
        # Check that there is exactly one '1' which indicates K=2 segments (one split)
        num_splits = sum(s for s in segments_k2 if s == 1)
        self.assertEqual(num_splits, 1) 
        
        # With K=4, and MIN_SEGMENT_SIZE=1, it should be possible to split into 4 segments
        # Each utterance becomes a segment
        segments_k4 = segmenter.segment_meeting(K=4)
        self.assertIsNotNone(segments_k4)
        self.assertEqual(len(segments_k4), len(self.transcript_simple))
        # Expected: [0, 1, 1, 1] (segment 1 starts at utt0, seg2 at utt1, seg3 at utt2, seg4 at utt3)
        # The first '0' is implicit start of first segment. Subsequent '1's are new segments.
        # The output format is a list of transitions.
        # The number of '1's should be K-1 if fully segmented.
        # However, the problem asks for K segments, and the output is transitions.
        # If K leaves, there are K-1 splits.
        # The transitions_hat logic in TreeSeg:
        # transitions_hat = [0] * len(self.leaves[0].segment)
        # for leaf in self.leaves[1:]: segment_transition[0]=1; transitions_hat.extend(segment_transition)
        # So, for K segments, there are K leaves. The first leaf contributes its length of 0s.
        # The K-1 other leaves contribute a 1 at their start, then 0s.
        
        # If K=4, leaves = [L0, L1, L2, L3] where each leaf is one utterance.
        # L0.segment = [idx0] -> transitions_hat = [0] (length 1)
        # L1.segment = [idx1] -> transitions_hat extended with [1] -> [0,1]
        # L2.segment = [idx2] -> transitions_hat extended with [1] -> [0,1,1]
        # L3.segment = [idx3] -> transitions_hat extended with [1] -> [0,1,1,1]
        self.assertEqual(segments_k4, [0, 1, 1, 1])


    def test_treeseg_segment_meeting_no_split_if_k1(self):
        configs = self.base_configs.copy()
        segmenter = TreeSeg(configs=configs, entries=self.transcript_simple)
        segments_k1 = segmenter.segment_meeting(K=1)
        # Expect one segment, so all zeros
        self.assertEqual(segments_k1, [0] * len(self.transcript_simple))

if __name__ == '__main__':
    unittest.main()
