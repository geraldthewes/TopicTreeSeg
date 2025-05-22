import unittest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

# Assuming TopicTreeSeg is installed or in PYTHONPATH for imports
from TopicTreeSeg.embeddings import Embeddings 
from TopicTreeSeg.ollama import ollama_embeddings, get_ollama_async_client_instance, _global_ollama_client
from TopicTreeSeg.openai import openai_embeddings

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


class TestOpenAIEmbeddings(unittest.TestCase):

    def setUp(self):
        self.openai_config = Embeddings(
            embeddings_func=None, # Not used
            headers={"Authorization": "Bearer testkey"},
            model="text-embedding-ada-002",
            endpoint="https://api.openai.com/v1/embeddings" # Mocked endpoint
        )
        self.chunks = ["openai test chunk 1", "openai test chunk 2"]

    @patch('TopicTreeSeg.openai.aiohttp.ClientSession') # Patching where ClientSession is looked up
    def test_openai_embeddings_success(self, MockClientSession):
        mock_session_instance = AsyncMock() # This is the mock for ClientSession()
        mock_response_obj = MagicMock() # This is the mock for the response object
        mock_response_obj.status = 200
        # The response.json() should be an async method
        mock_response_obj.json = AsyncMock(return_value={
            "data": [
                {"embedding": [0.5, 0.6]},
                {"embedding": [0.7, 0.8]}
            ]
        })
        
        # session.post() returns an async context manager
        mock_post_cm = AsyncMock() 
        mock_post_cm.__aenter__.return_value = mock_response_obj # What 'async with ... as response:' yields
        
        mock_session_instance.post = MagicMock(return_value=mock_post_cm)
        
        MockClientSession.return_value = mock_session_instance # When ClientSession() is called

        loop = asyncio.get_event_loop()
        result_embeddings = loop.run_until_complete(
            openai_embeddings(self.openai_config, self.chunks)
        )

        self.assertEqual(result_embeddings, [[0.5, 0.6], [0.7, 0.8]])
        mock_session_instance.post.assert_called_once_with(
            self.openai_config.endpoint,
            json={"model": self.openai_config.model, "input": self.chunks},
            timeout=unittest.mock.ANY 
        )
        MockClientSession.assert_called_once_with(headers=self.openai_config.headers)

    @patch('TopicTreeSeg.openai.aiohttp.ClientSession')
    def test_openai_embeddings_api_error(self, MockClientSession):
        mock_session_instance = AsyncMock()
        mock_response_obj = MagicMock()
        mock_response_obj.status = 400 # Simulate an error status
        mock_response_obj.text = AsyncMock(return_value="Bad Request Error") # Simulate error text
        
        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__.return_value = mock_response_obj
        mock_session_instance.post = MagicMock(return_value=mock_post_cm)
        MockClientSession.return_value = mock_session_instance

        loop = asyncio.get_event_loop()
        with self.assertRaisesRegex(Exception, "EmbeddingRequestFailed: status=400, details=Bad Request Error"):
            loop.run_until_complete(
                openai_embeddings(self.openai_config, self.chunks)
            )

if __name__ == '__main__':
    unittest.main()
