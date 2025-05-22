import unittest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

# Assuming TopicTreeSeg is installed or in PYTHONPATH for imports
from topic_treeseg.embeddings import Embeddings 
from topic_treeseg.ollama import ollama_embeddings, get_ollama_async_client_instance, _global_ollama_client


class TestOllamaEmbeddings(unittest.TestCase):

    def setUp(self):
        self.ollama_config = Embeddings(
            embeddings_func=None, # Not used by the function directly
            model="nomic-embed-text",
            endpoint="http://localhost:11434" # Mocked endpoint
        )
        self.chunks = ["test chunk 1", "test chunk 2"]
        
        # Reset global client for isolation between tests
        global _global_ollama_client
        _global_ollama_client = None


    @patch('TopicTreeSeg.ollama.AsyncClient') # Patching where AsyncClient is looked up
    def test_ollama_embeddings_success(self, MockAsyncClient):
        mock_client_instance = AsyncMock()
        # The actual response from ollama client is a dictionary, and 'embeddings' is a key in it.
        # The ollama_embeddings function extracts this.
        mock_client_instance.embed = AsyncMock(return_value={'embeddings': [[0.1, 0.2], [0.3, 0.4]]})
        
        # Configure the class mock to return our instance mock
        MockAsyncClient.return_value = mock_client_instance
        
        loop = asyncio.get_event_loop()
        result_embeddings = loop.run_until_complete(
            ollama_embeddings(self.ollama_config, self.chunks)
        )

        self.assertEqual(result_embeddings, [[0.1, 0.2], [0.3, 0.4]])
        mock_client_instance.embed.assert_called_once_with(
            model=self.ollama_config.model,
            input=self.chunks # ollama client uses 'input' not 'prompts' for embed
        )
        MockAsyncClient.assert_called_with(host=self.ollama_config.endpoint)


    @patch('TopicTreeSeg.ollama.AsyncClient')
    def test_ollama_embeddings_api_error(self, MockAsyncClient):
        mock_client_instance = AsyncMock()
        mock_client_instance.embed = AsyncMock(side_effect=Exception("Ollama API Error"))
        MockAsyncClient.return_value = mock_client_instance

        loop = asyncio.get_event_loop()
        with self.assertRaisesRegex(Exception, "Ollama API Error"):
            loop.run_until_complete(
                ollama_embeddings(self.ollama_config, self.chunks)
            )

    def test_get_ollama_async_client_instance_singleton(self):
        # This test checks if the global client is reused if already set.
        global _global_ollama_client
        
        # Case 1: Client is already set
        _global_ollama_client = "mocked_client_first_call"
        client1 = get_ollama_async_client_instance("http://some-endpoint-that-wont-be-used")
        self.assertEqual(client1, "mocked_client_first_call")
        
        _global_ollama_client = None # Reset

        # Case 2: Client is None, should try to create one (requires patching AsyncClient)
        # This part is more complex if we want to avoid actual client creation
        # and is covered by the success/error tests that patch AsyncClient.
        # Here, we focus on the singleton aspect when _global_ollama_client is manipulated.



if __name__ == '__main__':
    unittest.main()
