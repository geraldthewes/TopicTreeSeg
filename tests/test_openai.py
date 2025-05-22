import unittest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

# Assuming TopicTreeSeg is installed or in PYTHONPATH for imports
from TopicTreeSeg.embeddings import Embeddings 
from TopicTreeSeg.ollama import ollama_embeddings, get_ollama_async_client_instance, _global_ollama_client
from TopicTreeSeg.openai import openai_embeddings


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
