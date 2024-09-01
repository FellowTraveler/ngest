# Copyright 2024 Chris Odom
# MIT License

import aiohttp
import os
import datetime
import logging
import asyncio
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable
import PIL.Image
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import pipeline, AutoTokenizer, AutoModel, BertModel, BertTokenizer
import ast
import syn
from PyPDF2 import PdfReader
from typing import List, Dict, Any, Union, Optional, Generator
import esprima
from esprima import nodes
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import aiofiles
from collections import defaultdict
import copy
import pprint
import configparser
import dotenv
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import torch
from aiolimiter import AsyncLimiter

import ngest
import clang.cindex
from ngest.base_importer import NBaseImporter
from ngest.project import Project
from ngest.file import File
from ngest.document import Document
from ngest.pdf import PDF
from ngest.cpp_processor import CppProcessor
from ngest.custom_errors import FileProcessingError

# Load environment variables
# Load environment variables
dotenv.load_dotenv()

# Constants (now loaded from environment variables with fallbacks to config file)
config = configparser.ConfigParser()
config_dir = os.path.expanduser("~/.ngest")
config_file = os.path.join(config_dir, 'config.ini')

# Default hardcoded values
DEFAULT_MAX_FILE_SIZE = 31457280
DEFAULT_MEDIUM_CHUNK_SIZE = 10000
DEFAULT_SMALL_CHUNK_SIZE = 1000
DEFAULT_NEO4J_URL = "bolt://localhost:7689"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "mynewpassword"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_LOCAL_MODEL = "nomic-embed-text"
DEFAULT_EMBEDDING_API_MODEL = "text-embedding-ada-002"

DEFAULT_SUMMARY_API_MODEL = "gpt-4o-mini"
DEFAULT_SUMMARY_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_SUMMARY_API_KEY = "your-summary-api-key"  # Replace with your default key if needed

DEFAULT_EMBEDDING_API_URL = "https://api.openai.com/v1/embeddings"

if not os.path.exists(config_dir):
    os.makedirs(config_dir)

if not os.path.isfile(config_file):
    with open(config_file, 'w') as f:
        f.write(f"[Limits]\nMAX_FILE_SIZE = {DEFAULT_MAX_FILE_SIZE}\n\n"
                f"[Chunks]\nMEDIUM_CHUNK_SIZE = {DEFAULT_MEDIUM_CHUNK_SIZE}\nSMALL_CHUNK_SIZE = {DEFAULT_SMALL_CHUNK_SIZE}\n\n"
                f"[Database]\nNEO4J_URL = {DEFAULT_NEO4J_URL}\nNEO4J_USER = {DEFAULT_NEO4J_USER}\nNEO4J_PASSWORD = {DEFAULT_NEO4J_PASSWORD}\n\n"
                f"[Ollama]\nOLLAMA_URL = {DEFAULT_OLLAMA_URL}\n\n"
                f"[Models]\nEMBEDDING_LOCAL_MODEL = {DEFAULT_EMBEDDING_LOCAL_MODEL}\nEMBEDDING_API_MODEL = {DEFAULT_EMBEDDING_API_MODEL}\n\n"
                f"[SummaryAPI]\nSUMMARY_API_URL = {DEFAULT_SUMMARY_API_URL}\nSUMMARY_API_MODEL = {DEFAULT_SUMMARY_API_MODEL}\nSUMMARY_API_KEY = {DEFAULT_SUMMARY_API_KEY}")
else:
    config.read(config_file)

MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', config.get('Limits', 'MAX_FILE_SIZE', fallback=DEFAULT_MAX_FILE_SIZE)))
MEDIUM_CHUNK_SIZE = int(os.getenv('MEDIUM_CHUNK_SIZE', config.get('Chunks', 'MEDIUM_CHUNK_SIZE', fallback=DEFAULT_MEDIUM_CHUNK_SIZE)))
SMALL_CHUNK_SIZE = int(os.getenv('SMALL_CHUNK_SIZE', config.get('Chunks', 'SMALL_CHUNK_SIZE', fallback=DEFAULT_SMALL_CHUNK_SIZE)))
NEO4J_URL = os.getenv('NEO4J_URL', config.get('Database', 'NEO4J_URL', fallback=DEFAULT_NEO4J_URL))
NEO4J_USER = os.getenv('NEO4J_USER', config.get('Database', 'NEO4J_USER', fallback=DEFAULT_NEO4J_USER))
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', config.get('Database', 'NEO4J_PASSWORD', fallback=DEFAULT_NEO4J_PASSWORD))
OLLAMA_URL = os.getenv('OLLAMA_URL', config.get('Ollama', 'OLLAMA_URL', fallback=DEFAULT_OLLAMA_URL))

EMBEDDING_LOCAL_MODEL = os.getenv('EMBEDDING_LOCAL_MODEL', config.get('Models', 'EMBEDDING_LOCAL_MODEL', fallback=DEFAULT_EMBEDDING_LOCAL_MODEL))
EMBEDDING_API_MODEL = os.getenv('EMBEDDING_API_MODEL', config.get('Models', 'EMBEDDING_API_MODEL', fallback=DEFAULT_EMBEDDING_API_MODEL))

SUMMARY_API_URL = os.getenv('SUMMARY_API_URL', config.get('SummaryAPI', 'SUMMARY_API_URL', fallback=DEFAULT_SUMMARY_API_URL))
SUMMARY_API_MODEL = os.getenv('SUMMARY_API_MODEL', config.get('SummaryAPI', 'SUMMARY_API_MODEL', fallback=DEFAULT_SUMMARY_API_MODEL))
SUMMARY_API_KEY = os.getenv('SUMMARY_API_KEY', config.get('SummaryAPI', 'SUMMARY_API_KEY', fallback=DEFAULT_SUMMARY_API_KEY))

# Configure logging with different levels
logging.basicConfig(level=logging.info, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the logging level based on a condition (e.g., a command-line argument or an environment variable)
hide_info_logs = False  # Set this to True to hide info logs

if hide_info_logs:
    logging.getLogger().setLevel(logging.WARNING)
else:
    logging.getLogger().setLevel(logging.INFO)


class NNeo4JImporter(NBaseImporter):
    """
    A file importer that ingests various file types into a Neo4j database.
    """
    def __init__(self, neo4j_url: str = NEO4J_URL, neo4j_user: str = NEO4J_USER, neo4j_password: str = NEO4J_PASSWORD, use_api=True, api_url=SUMMARY_API_URL, api_key=SUMMARY_API_KEY):
        self.use_api = use_api
        self.summary_api_url = api_url
        self.summary_api_key = api_key
        self.summary_api_model = SUMMARY_API_MODEL
        self.cpp_processor = CppProcessor(do_summarize_text=self.do_summarize_text)
        self.neo4j_url = neo4j_url
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = AsyncGraphDatabase.driver(self.neo4j_url, auth=(self.neo4j_user, self.neo4j_password))
        logger.info(f"Neo4J initialized with URL {neo4j_url}")

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        logging.getLogger("transformers").setLevel(logging.ERROR)

        self.semaphore_summarizer = asyncio.Semaphore(1)  # Limit to 1 concurrent tasks
        self.semaphore_embedding = asyncio.Semaphore(1)  # Limit to 1 concurrent tasks

        self.lock_summarizer = asyncio.Lock()
        self.summarizer = pipeline("summarization", model="google/pegasus-xsum", device=self.device)
#        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=self.device)

#        self.image_model = models.densenet121(pretrained=True).to(self.device)
#        self.image_model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.code_model = AutoModel.from_pretrained("microsoft/codebert-base").to(self.device)
        self.code_model.eval()

        self.rate_limiter_db = AsyncLimiter(50, 1)  # 3 operations per second
        
        if self.use_api:
            self.rate_limiter_summary = AsyncLimiter(10, 1)  # 10 operations per second for API
            self.rate_limiter_embedding = AsyncLimiter(10, 1)  # 10 operations per second for API
        else:
            self.rate_limiter_summary = AsyncLimiter(1, 1)  # 2 operations per second for local
            self.rate_limiter_embedding = AsyncLimiter(1, 1)  # 2 operations per second for local

        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_model.eval()
        
        self.header_files = {}

        self.file_paths = []
        self.file_paths_lock = asyncio.Lock()

        self.progress_callback_scan = None
        self.progress_callback_summarize = None
        self.progress_callback_store = None
        
        self.progress_callback_summarize_start = None
        self.progress_callback_store_start = None

    def set_progress_callback_scan(self, callback):
        self.progress_callback_scan = callback
    def set_progress_callback_summarize(self, callback):
        self.progress_callback_summarize = callback
    def set_progress_callback_store(self, callback):
        self.progress_callback_store = callback
        
    def set_progress_callback_summarize_start(self, callback):
        self.progress_callback_summarize_start = callback
    def set_progress_callback_store_start(self, callback):
        self.progress_callback_store_start = callback

    async def add_file_path(self, path: str):
        async with self.file_paths_lock:
            self.file_paths.append(path)

    async def get_file_paths(self) -> List[str]:
        async with self.file_paths_lock:
            return self.file_paths.copy()

    async def get_file_paths_length(self) -> int:
        async with self.file_paths_lock:
            return len(self.file_paths)

    async def closeDB(self):
        await self.driver.close()

    @asynccontextmanager
    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=4, max=10), retry=retry_if_exception_type(ServiceUnavailable))
    async def get_session(self):
        """
        Get a session for Neo4j database operations.

        Yields:
            The Neo4j session.
        """
        async with self.driver.session() as session:
            try:
                yield session
            except Exception as e:
                logger.error(f"Error in get_session: {e}")
            finally:
                await session.close()

    async def make_embedding(self, text: str) -> List[float]:
        try:
            # this only limits the creation of new threads.
            async with self.rate_limiter_embedding:
                return await self.generate_embedding(text)
        except Exception as e:
            logger.error(f"Error making embedding: {e}")
            return []
            
    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=15, max=30))
    async def generate_embedding(self, text: str) -> List[float]:
        async with self.semaphore_embedding:  # Ensure only 5 concurrent tasks
            try:
                loop = asyncio.get_running_loop()
#                logger.info(f"Started generating an embedding.")
                embedding = await loop.run_in_executor(None, self._generate_embedding, text)
#                logger.info(f"Finished generating an embedding: {embedding}")
                return embedding
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                raise

    def _generate_embedding(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()

    # Function to use Ollama for generating embeddings
#    async def generate_embedding(self, text: str) -> List[float]:
#        try:
#            response = await ollama_generate_embedding(prompt=text)
#            if response and 'embedding' in response:
#                return response['embedding']
#            else:
#                raise ValueError("Failed to generate embedding")
#        except Exception as e:
#            logger.error(f"Error generating embedding: {e}")
#            raise

    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, topLevelInputPath: str, topLevelOutputPath: str, currentOutputPath: str, project_id: str) -> None:
        """
        Ingest a file by determining its type and using the appropriate ingestion method.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            topLevelInputPath (str): The top-level input path.
            topLevelOutputPath (str): The top-level output path.
            currentOutputPath (str): The current output path.
            project_id (str): The project ID.
            
        Example...
        Completed scanning for:
        inputPath:          /Users/au/src/blindsecp/sec2-v2.pdf
        inputLocation:      /Users/au/src/blindsecp
        inputName:          sec2-v2.pdf
        topLevelOutputPath: /Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3
        currentOutputPath:  /Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3/blindsecp
        project_id:          02688845-0e74-4c5f-9b50-af4772dca5e3

        """
        def strip_top_level(inputPath, topLevelInputPath):
            # Ensure both paths are absolute and normalized
            inputPath = os.path.abspath(inputPath)
            topLevelInputPath = os.path.abspath(topLevelInputPath)
            
            # Remove the top level path
            if inputPath.startswith(topLevelInputPath):
                return inputPath[len(topLevelInputPath):].lstrip(os.sep)
            else:
                return inputPath

#       logger.info(f"Starting scanning for file: {inputPath}")
        try:
            localPath = strip_top_level(inputPath, topLevelInputPath)
            file_type = self.ascertain_file_type(inputPath)

            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    file = await File.create_in_database(
                        session,
                        filename=inputName,
                        full_path=localPath,
                        size_in_bytes=file_type['size_in_bytes'],
                        project_id=project_id,
                        created_date=file_type['created_date'],
                        modified_date=file_type['modified_date'],
                        extension=file_type['extension']
                    )
                    if not file:
                        logger.error("Failed to create file in database")
                        return -1
                    file_id = file.id
        
            ingest_method = getattr(self, f"Ingest{file_type['type'].capitalize()}", None)
            
            if ingest_method:
                await self.add_file_path(inputPath)

                # localPath                 == "blindsecp/sec2-v2.pdf"
                ## RESUME
                
#            APPLICATION: "global"
#             1. projectsFolder:          "/Users/au/.ngest/projects"
#                                                                  "/"
#            PROJECT:  "blindsecp"
#                project_id:                                         "02688845-0e74-4c5f-9b50-af4772dca5e3"
#                topLevelOutputPath:      "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3"
#                topLevelOutputPath:           $projectsFolder     "/"           $project_id
#                projectsFolder:          "/Users/au/.ngest/projects"
#                topLevelOutputPath:      "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3"
#             1. projectName: "blindsecp"                                                                "blindsecp"
#                project_root: "blindsecp"                                                               "blindsecp"
#             2. inputLocation: "/Users/au/src/blindsecp"                                  "/Users/au/src/blindsecp"
#                project_input_location: "/Users/au/src"                                   "/Users/au/src"
#                topLevelOutputPath:      "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3"
#                currentOutputPath:       "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3/blindsecp"
#                topLevelOutputPath:      "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3"
#                topLevelOutputPath:           $projectsFolder     "/"           $project_id
#                projectName:                                                                            "blindsecp"
#                currentOutputPath:            $projectsFolder     "/"           $project_id             "/" $project_root
#             3. project_id:                                         "02688845-0e74-4c5f-9b50-af4772dca5e3"
#                topLevelOutputPath:           $projectsFolder     "/"           $project_id
#                projectName:                                                                            "blindsecp"
#             4. currentOutputPath:       "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3/blindsecp"
#                currentOutputPath:            $projectsFolder     "/"           $project_id             "/" $project_root
#             5. topLevelOutputPath:      "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3"
#                topLevelOutputPath:           $projectsFolder     "/"           $project_id
#                currentOutputPath:            $projectsFolder     "/"           $project_id             "/blindsecp"
#                currentOutputPath:            $projectsFolder     "/"           $project_id             "/" $project_root
#                currentOutputPath:       "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3/blindsecp"
#             6. projectSummary:     "..."
#             7. project_summary_embedding: [...]
#                topLevelOutputPath:      "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3"
#                currentOutputPath:       "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3/blindsecp"
#                                                                                                       "/"
#            FILE:  "blindsecp/sec2-v2.pdf"                                                             "/"
#                project_input_location: "/Users/au/src"
#                inputLocation:          "/Users/au/src/blindsecp"
#                inputPath:              "/Users/au/src/blindsecp/sec2-v2.pdf"
#                inputName:                                      "sec2-v2.pdf"
#                Local path:                           "blindsecp/sec2-v2.pdf"
#                File extension:  "pdf"                                  "pdf"
#                FileID:         <md5 hash of file contents>
#                Project ID:     "02688845-0e74-4c5f-9b50-af4772dca5e3"
#                Creation date:  ...
#                Modified date:  ...
#                ingestion date: 5/12/23
#                inputLocation: "/Users/au/src/blindsecp"                                  "/Users/au/src/blindsecp"
#                inputPath: "/Users/au/src/blindsecp/sec2-v2.pdf"                          "/Users/au/src/blindsecp/sec2-v2.pdf"
#                Local path: "blindsecp/sec2-v2.pdf"                                                     "blindsecp/sec2-v2.pdf"
#                Filename:   "sec2-v2.pdf"                                                                         "sec2-v2.pdf"
#                File extension:  "pdf"                                                                                    "pdf"
#                topLevelOutputPath:     "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3"
#                currentOutputPath:      "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3/blindsecp"
#                project_id:                                        "02688845-0e74-4c5f-9b50-af4772dca5e3"
#
#                DOCUMENT
#                Name: "blindsecp/sec2-v2.pdf"
#                Document ID: "..."
#                Document Type: PDF
#                Project ID: ...
#                ingestion date: 5/12/23
#                Author: ...
#                Title:
#                contents_markdown: "..."
#                summary: "..."
#                summary_embedding: [...]
#                

                async with self.rate_limiter_db:
                    async with self.get_session() as session:
                        document = await Document.create_in_database(
                            session,
                            project_id=project_id,
                            full_path=localPath,
                            content_type=file_type['type']
                        )
                        if not document:
                            logger.error("Failed to create document in database")
                            return -1
                        document_id = document.id

                await ingest_method(inputPath, inputLocation, inputName, localPath, currentOutputPath, project_id)
            else:
                logger.warning(f"No ingest method for file type: {file_type['type']}, inputPath: {inputPath}")
        except Exception as e:
            logger.error(f"Error ingesting {file_type} file {inputPath}: {e}")
            raise FileProcessingError(f"Error ingesting {file_type} file {inputPath}: {e}")
        finally:
            # Update progress for scanning phase
            await self.update_progress_scan(1)
            await self.update_progress_summarize(1)
            await self.update_progress_store(1)

        logger.info(f"Completed scanning for inputPath: {inputPath}, inputLocation: {inputLocation}, inputName: {inputName}, currentOutputPath: {currentOutputPath}")

    async def do_summarize_text(self, text, max_length=50, min_length=25) -> str:
        async with self.rate_limiter_summary:
            if self.use_api:
                return await self.api_summarize_text(text)
            else:
                return await self.local_summarize_text(text, max_length, min_length)

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=15, max=30),
           retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)))
    async def api_summarize_text(self, text):
        payload = {
            "model": self.summary_api_model,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Summarize the content you are provided."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ],
            "temperature": 1,
            "max_tokens": 256,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "response_format": {
                "type": "text"
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.summary_api_key}"
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.summary_api_url,  # Ensure this is set to "https://api.openai.com/v1/chat/completions"
                    json=payload,
                    headers=headers,
                    timeout=60  # Add a timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data["choices"][0]["message"]["content"], list):
                            summary = data["choices"][0]["message"]["content"][0]["text"]
                        else:
                            summary = data["choices"][0]["message"]["content"]
                        return summary
#                    if response.status == 200:
#                        data = await response.json()
#                        summary = data["choices"][0]["message"]["content"][0]["text"]
#                        return summary
                    else:
                        raise Exception(f"API request failed with status {response.status}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"API summarization error: {e}")
                raise  # Re-raise the error to trigger the retry
               
#    async def api_summarize_text(self, text, max_length, min_length):
#        async with aiohttp.ClientSession() as session:
#            try:
#                async with session.post(
#                    self.summary_api_url,
#                    json={"text": text, "max_length": max_length, "min_length": min_length},
#                    headers={"Authorization": f"Bearer {self.summary_api_key}"},
#                    timeout=60  # Add a timeout
#                ) as response:
#                    if response.status == 200:
#                        data = await response.json()
#                        return data["summary"]
#                    else:
#                        raise Exception(f"API request failed with status {response.status}")
#            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
#                logger.error(f"API summarization error: {e}")
#                raise  # Re-raise the error to trigger the retry

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=15, max=30))
    async def local_summarize_text(self, text, max_length=50, min_length=25) -> str:
        """
        Summarize the given text using a pre-trained model.

        Args:
            text (str): The text to be summarized.

        Returns:
            str: The summary of the text.
        """
        async with self.semaphore_summarizer:  # Ensure only 5 concurrent tasks
            try:
                loop = asyncio.get_running_loop()
#                logger.info(f"Summarizing text...")
                async with self.lock_summarizer:
                    retval = await loop.run_in_executor(None, lambda: self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text'])
#                logger.info(f"Finished summarizing text.")
                return retval
            except Exception as e:
                logger.error(f"Error summarizing text: {e}")
                return ""
        
    async def process_chunks(self, text, parent_node_id, parent_node_type, chunk_size, chunk_type, project_id):
        """
        Process text chunks and create corresponding nodes in the Neo4j database.

        Args:
            text: The text to be chunked.
            parent_node_id: The ID of the parent node.
            parent_node_type: The type of the parent node.
            chunk_size: The size of each chunk.
            chunk_type: The type of the chunk.
            project_id: The project ID.
        """
        chunks = self.chunk_text(text, chunk_size)
        for i, chunk in enumerate(chunks):
            await self.create_chunk_with_embedding(chunk, parent_node_id, parent_node_type, chunk_type, project_id)

    async def create_chunk_with_embedding(self, chunk, parent_node_id, parent_node_type, chunk_type, project_id):
        """
        Create a chunk node with embedding in the Neo4j database.

        Args:
            chunk: The text chunk.
            parent_node_id: The ID of the parent node.
            parent_node_type: The type of the parent node.
            chunk_type: The type of the chunk.
            project_id: The project ID.
        """
        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    chunk_id = await self.run_query_and_get_element_id(session,
                        f"CREATE (n:{chunk_type} {{content: $content, type: $chunk_type, project_id: $project_id}}) RETURN elementId(n)",
                        content=chunk, chunk_type=chunk_type, project_id=project_id
                    )
                    await (await session.run(
                        f"MATCH (p:{parent_node_type} {{elementId: $parent_node_id, project_id: $project_id}}) "
                        f"MATCH (c:{chunk_type} {{elementId: $chunk_id, project_id: $project_id}}) "
                        f"CREATE (p)-[:HAS_CHUNK]->(c), (c)-[:PART_OF]->(p)",
                        parent_node_id=parent_node_id, chunk_id=chunk_id, project_id=project_id
                    )).consume()
        except Exception as e:
            logger.error(f"Error creating chunk during db query.")
            raise DatabaseError(f"Error creating chunk during db query.")

        try:
#            logger.info(f"Making embedding for chunk_id: {chunk_id}")
            embedding = await self.make_embedding(chunk)
#            logger.info(f"Finished making embedding for chunk_id: {chunk_id}")
        except Exception as e:
            logger.error(f"Error creating chunk embedding: {e}")
            raise DatabaseError(f"Error creating chunk embedding: {e}")
        
        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    embedding_id = await self.run_query_and_get_element_id(session,
                        "CREATE (n:Embedding {embedding: $embedding, type: 'embedding', project_id: $project_id}) RETURN elementId(n)",
                        embedding=embedding, project_id=project_id
                    )
                    await (await session.run(
                        f"MATCH (c:{chunk_type} {{elementId: $chunk_id, project_id: $project_id}}) "
                        f"MATCH (e:Embedding {{elementId: $embedding_id, project_id: $project_id}}) "
                        f"CREATE (c)-[:HAS_EMBEDDING]->(e)",
                        chunk_id=chunk_id, embedding_id=embedding_id, project_id=project_id
                    )).consume()
        except Exception as e:
            logger.error(f"Error storing chunk with embedding: {e}")
            raise DatabaseError(f"Error storing chunk with embedding: {e}")

    async def IngestText(self, input_path: str, input_location: str, input_name: str, localPath: str, current_output_path: str, project_id: str) -> None:
            try:
                # File reading operation - doesn't need rate limiting
                async with aiofiles.open(input_path, 'r') as file:
                    file_content = await file.read()
            except Exception as e:
                logger.error(f"IngestText: Error reading file {input_path}: {e}")
                raise FileProcessingError(f"IngestText: Error reading file {input_path}: {e}")

            try:
                # Database operation - needs rate limiting
                async with self.rate_limiter_db:
                    async with self.get_session() as session:
                        parent_doc_id = await self.run_query_and_get_element_id(session,
                            "CREATE (n:Document {name: $name, type: 'text', project_id: $project_id}) RETURN elementId(n)",
                            name=input_name, project_id=project_id
                        )

                # This method likely contains multiple database operations,
                # so we don't wrap it in rate_limiter here. It should handle its own rate limiting internally.
                await self.process_chunks(file_content, parent_doc_id, "Document", MEDIUM_CHUNK_SIZE, "MediumChunk", project_id)
        
            except Exception as e:
                logger.error(f"Error ingesting text file {input_path}: {e}")
                raise DatabaseError(f"Error ingesting text file {input_path}: {e}")

    async def extract_image_features(self, image: PIL.Image.Image) -> List[float]:
        """
        Extract features from an image using a pre-trained model.

        Args:
            image (PIL.Image.Image): The image to be processed.

        Returns:
            List[float]: The extracted features.
        """
        try:
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor = preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.image_model(tensor)
            return features.squeeze().cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            raise FileProcessingError(f"Error extracting image features: {e}")

    async def IngestImg(self, input_path: str, input_location: str, input_name: str, localPath: str, current_output_path: str, project_id: str) -> None:
        """
        Ingest an image file by extracting features and creating nodes in the database.

        Args:
            input_path (str): The path to the input file.
            input_location (str): The location of the input file.
            input_name (str): The name of the input file.
            current_output_path (str): The current output path.
            project_id (str): The project ID.
        """
        try:
            image = PIL.Image.open(input_path)
            await self.process_image_file(image, input_name, project_id)
            
        except Exception as e:
            logger.error(f"Error ingesting image: {e}")
            raise FileProcessingError(f"Error ingesting image: {e}")

    async def process_image_file(self, image: PIL.Image.Image, input_name: str, project_id: str):
        """
        Process an image file by extracting features and storing them in the database.

        Args:
            image (PIL.Image.Image): The image to be processed.
            input_name (str): The name of the input file.
            project_id (str): The project ID.
        """
        features = await self.extract_image_features(image)
        await self.store_image_features(features, input_name, project_id)

    async def store_image_features(self, features: List[float], input_name: str, project_id: str):
        """
        Store the extracted image features in the database.

        Args:
            features (List[float]): The extracted features.
            input_name (str): The name of the input file.
            project_id (str): The project ID.
        """
        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    image_id = await self.run_query_and_get_element_id(session,
                        "CREATE (n:Image {name: $name, type: 'image', project_id: $project_id}) RETURN elementId(n)",
                        name=input_name, project_id=project_id
                    )
                    features_id = await self.run_query_and_get_element_id(session,
                        "CREATE (n:ImageFeatures {features: $features, type: 'image_features', project_id: $project_id}) RETURN elementId(n)",
                        features=features, project_id=project_id
                    )
                    await (await session.run(
                        "MATCH (i:Image), (f:ImageFeatures) WHERE elementId(i) = $image_id AND elementId(f) = $features_id AND i.project_id = $project_id AND f.project_id = $project_id "
                        "CREATE (i)-[:HAS_FEATURES]->(f)",
                        image_id=image_id, features_id=features_id, project_id=project_id).consume()
                    )
        except Exception as e:
            logger.error(f"Error storing image features: {e}")
            raise DatabaseError(f"Error storing image features: {e}")

    def get_code_features(self, code: str) -> List[float]:
        """
        Extract features from code using a pre-trained model.

        Args:
            code (str): The code to be processed.

        Returns:
            List[float]: The extracted features.
        """
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.code_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()


    async def IngestCpp(self, inputPath: str, inputLocation: str, inputName: str, localPath: str, currentOutputPath: str, project_id: str):
        try:
            await self.cpp_processor.process_cpp_file(inputPath, inputLocation, project_id)
        except FileProcessingError as e:
            logger.error(f"Error ingesting C++ file {inputPath}: {e}")
            raise  # Re-raise the FileProcessingError
        except Exception as e:
            logger.error(f"Unexpected error ingesting C++ file {inputPath}: {e}")
            raise FileProcessingError(f"Unexpected error ingesting C++ file {inputPath}: {e}")

    def get_raw_code(self, node):
        # Extract raw code from the source node
        if node.extent.start.file and node.extent.end.file:
            with open(node.extent.start.file.name, 'r') as file:
                file.seek(node.extent.start.offset)
                raw_code = file.read(node.extent.end.offset - node.extent.start.offset)
            return raw_code
        return ''

    async def update_progress_scan(self, increment=1):
        if self.progress_callback_scan:
            await self.progress_callback_scan(increment)
            
    async def update_progress_summarize(self, increment=1):
        if self.progress_callback_summarize:
            await self.progress_callback_summarize(increment)
            
    async def update_progress_store(self, increment=1):
        if self.progress_callback_store:
            await self.progress_callback_store(increment)
        
#    async def summarize_all_cpp(self, project_id):
#        try:
#            tasks = await self.cpp_processor.prepare_summarization_tasks()
#            logger.info(f"Prepared {len(tasks)} summarization tasks")
#
#            for task_type, name, info in tasks:
#                logger.info(f"Starting summarization for {task_type} {name}")
#                try:
#                    if task_type == 'Class':
#                        await self.summarize_cpp_class_prep(name, info)
#                    elif task_type == 'Method':
#                        await self.summarize_cpp_method_prep(name, info)
#                    elif task_type == 'Function':
#                        await self.summarize_cpp_function_prep(name, info)
#                    logger.info(f"Completed summarization for {task_type} {name}")
#                except Exception as e:
#                    logger.error(f"Error in summarization task for {task_type} {name}: {e}")
#
#            logger.info("Completed all summarization tasks")
#
#        except Exception as e:
#            logger.error(f"Error while gathering summarization tasks: {e}")
#            raise FileProcessingError(f"Error while gathering summarization tasks: {e}")

    async def summarize_all_cpp(self, project_id):
        try:
            tasks = await self.cpp_processor.prepare_summarization_tasks()
            semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks
            total_tasks = len(tasks)
            completed_tasks = 0
            logger.info(f"Prepared {total_tasks} summarization tasks")

            async def process_task(task):
                nonlocal completed_tasks
                async with semaphore:
                    task_type, name, details = task
                    result = False
                    try:
                        logger.info(f"Starting summarization for {task_type} {name}")
                        if task_type == 'Class':
                            result = await self.summarize_cpp_class_prep(name, details)
                        elif task_type == 'Method':
                            result = await self.summarize_cpp_method_prep(name, details)
                        elif task_type == 'Function':
                            result = await self.summarize_cpp_function_prep(name, details)
                        else:
                            logger.warning(f"Unknown task type {task_type} for {name}")

                        if result:
                            logger.info(f"Completed summarization for {task_type} {name}")
                            completed_tasks += 1
                            logger.info(f"Completed {completed_tasks}/{total_tasks} summarization tasks")
                        else:
                            logger.warning(f"Summarization for {task_type} {name} did not produce a result")
                    except Exception as e:
                        logger.error(f"Error in summarization task for {task_type} {name}: {e}")
                        logger.error(f"Error details: {traceback.format_exc()}")

            await asyncio.gather(*[process_task(task) for task in tasks])

            if completed_tasks != total_tasks:
                logger.warning(f"Only {completed_tasks} out of {total_tasks} summarization tasks completed successfully")
            else:
                logger.info("Completed all summarization tasks successfully")

        except Exception as e:
            logger.error(f"Error while gathering summarization tasks: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            raise FileProcessingError(f"Error while gathering summarization tasks: {e}")

#    async def summarize_all_cpp(self, project_id):
#        try:
#            tasks = await self.cpp_processor.prepare_summarization_tasks()
#
#            summarization_tasks = []
#            for task_type, name, info in tasks:
#                if task_type == 'Class':
#                    summarization_tasks.append(self.summarize_cpp_class_prep(name, info))
#                elif task_type == 'Method':
#                    summarization_tasks.append(self.summarize_cpp_method_prep(name, info))
#                elif task_type == 'Function':
#                    summarization_tasks.append(self.summarize_cpp_function_prep(name, info))
#
#            total_tasks = len(summarization_tasks)
#            completed_tasks = 0
#
#            # Use asyncio.as_completed to process tasks as they finish
#            for future in asyncio.as_completed(summarization_tasks):
#                try:
#                    await future
#                    completed_tasks += 1
#                    logger.info(f"Completed {completed_tasks}/{total_tasks} summarization tasks")
#                except Exception as e:
#                    logger.error(f"Error in summarization task: {e}")
#
#            if completed_tasks != total_tasks:
#                logger.warning(f"Only {completed_tasks} out of {total_tasks} summarization tasks completed")
#
#            # Ensure all tasks are completed
##            await asyncio.gather(*summarization_tasks)
#
#        except Exception as e:
#            logger.error(f"Error while gathering summarization and embedding tasks: {e}")
#            raise FileProcessingError(f"Error while gathering summarization and embedding tasks: {e}")


    async def summarize_cpp_class_prep(self, name, class_info):
        try:
            logger.info(f"Starting summarization for class: {name}, namespace: {class_info.get('namespace', 'N/A')}")
            
            class_name, class_scope, interface_summary, implementation_summary = await self.cpp_processor.summarize_cpp_class(class_info)
            logger.info(f"Summaries created for class: {name}")
            
            interface_embedding = await self.make_embedding(interface_summary)
            implementation_embedding = await self.make_embedding(implementation_summary)
            logger.info(f"Embeddings created for class: {name}")
            
            details = {
                'interface_summary': interface_summary,
                'implementation_summary': implementation_summary,
                'interface_embedding': interface_embedding,
                'implementation_embedding': implementation_embedding
            }
            await self.cpp_processor.update_class(name, details)
            await self.update_progress_summarize(1)
            logger.info(f"Finished summarizing/embedding class: {name}")
            return True

        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_prep for class {name}: {e}")
            # Try to salvage partial information
#            partial_details = {k: v for k, v in class_info.items() if k in ['interface_description', 'implementation_description']}
#            await self.cpp_processor.update_class(name, partial_details)
            return False
            

    async def summarize_cpp_method_prep(self, name, method_info):
        function_name = name
        try:
            if name is None:
                logger.info(f"Skipping anonymous method")
                return True
        except Exception as e:
            logger.error(f"Error in summarize_cpp_method_prep grabbing dict values: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_method_prep grabbing dict values: {e}")

        try:
            function_name, function_scope, function_summary = await self.cpp_processor.summarize_cpp_function(method_info)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_method_prep: {e}\nmethod_info contents:\n{pprint.pformat(method_info, indent=4)}")
#            await self.cpp_processor.update_method(name, {})
            raise FileProcessingError(f"Error in summarize_cpp_method_prep calling summarize_cpp_function for method {name}: {e}")
        
        try:
            embedding = await self.make_embedding(function_summary)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_method_prep calling make_embedding: {e}")
#            details = {
#                'summary': function_summary,
#            }
#            await self.cpp_processor.update_method(name, details)
#            await self.cpp_processor.update_method(name, {})
            raise FileProcessingError(f"Error in summarize_cpp_method_prep calling make_embedding: {e}")
            
        try:
            details = {
                'summary': function_summary,
                'embedding': embedding
            }
            await self.cpp_processor.update_method(name, details)
            await self.update_progress_summarize(1)
            logger.info(f"Finished summarizing/embedding cpp method: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error in summarize_cpp_method_prep for method {name}: {e}")
            # Ensure we still update the method with empty details
#            await self.cpp_processor.update_method(name, {})
            return False


    async def summarize_cpp_function_prep(self, name, function_info):
        function_name = name
        try:
            if name is None:
                logger.info(f"Skipping anonymous function")
                return function_info
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function_prep grabbing dict values: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_function_prep grabbing dict values: {e}")
        
        try:
            function_name, function_scope, function_summary = await self.cpp_processor.summarize_cpp_function(function_info)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function_prep calling summarize_cpp_function: {e}")
#            await self.cpp_processor.update_function(name, {})
            raise FileProcessingError(f"Error in summarize_cpp_function_prep calling summarize_cpp_function: {e}")

        try:
            embedding = await self.make_embedding(function_summary)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function_prep calling make_embedding: {e}")
#            details = {
#                'summary': function_summary,
#            }
#            await self.cpp_processor.update_function(name, details)
#            await self.cpp_processor.update_function(name, {})
            raise FileProcessingError(f"Error in summarize_cpp_function_prep calling make_embedding: {e}")
        
        try:
            details = {
                'summary': function_summary,
                'embedding': embedding
            }
            await self.cpp_processor.update_function(name, details)
            await self.update_progress_summarize(1)
            logger.info(f"Finished summarizing/embedding cpp function: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function_prep calling update_functions: {e}")
            await self.cpp_processor.update_function(name, {})
#            raise FileProcessingError(f"Error in summarize_cpp_function_prep calling update_functions: {e}")
            return False


    async def store_all_cpp(self, project_id):
        try:
            tasks = await self.cpp_processor.prepare_summarization_tasks()

            for task_type, name, info in tasks:
                if task_type == 'Class':
                    if 'interface_summary' not in info or 'implementation_summary' not in info:
                        logger.warning(f"Class {name} missing summaries, skipping storage")
                        continue
                    await self.store_summary_cpp_class(name, info, project_id)
                elif task_type == 'Method':
                    if 'summary' not in info:
                        logger.warning(f"Method {name} missing summary, skipping storage")
                        continue
                    await self.store_summary_cpp_method(name, info, project_id)
                elif task_type == 'Function':
                    if 'summary' not in info:
                        logger.warning(f"Function {name} missing summary, skipping storage")
                        continue
                    await self.store_summary_cpp_function(name, info, project_id)
                
                await self.update_progress_store(1)

        except Exception as e:
            logger.error(f"Error in store_all_cpp: {e}")
            raise FileProcessingError(f"Error in store_all_cpp: {e}")


    async def store_summary_cpp_class(self, full_name, info, project_id):
        logger.info(f"Storing summary for class: {full_name}")
        try:
            type_name = info.get('type', 'Class')
            
            query = """
                MERGE (n:{type} {{name: $full_name, project_id: $project_id}})
                ON CREATE SET n.namespace = $namespace,
                              n.short_name = $short_name,
                              n.scope = $scope,
                              n.interface_summary = $interface_summary,
                              n.implementation_summary = $implementation_summary,
                              n.interface_embedding = $interface_embedding,
                              n.implementation_embedding = $implementation_embedding,
                              n.file_path = $file_path,
                              n.raw_code = $raw_code
                ON MATCH SET
                    n.interface_summary = $interface_summary,
                    n.implementation_summary = $implementation_summary,
                    n.interface_embedding = $interface_embedding,
                    n.implementation_embedding = $implementation_embedding
                RETURN elementId(n)
            """.format(type=type_name)
            
            params = {
                'full_name': full_name,
                'namespace': info.get('namespace', ''),
                'short_name': info.get('short_name', ''),
                'project_id': project_id if project_id else 'Unknown',
                'scope': info.get('scope', ''),
                'interface_summary': info.get('interface_summary', ''),
                'implementation_summary': info.get('implementation_summary', ''),
                'interface_embedding': info.get('interface_embedding', []),
                'implementation_embedding': info.get('implementation_embedding', []),
                'file_path': info.get('file_path', ''),
                'raw_code': info.get('raw_code', '')
            }
            
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    await self.run_query_and_get_element_id(session, query, **params)
                    logger.info(f"Finished storing CPP class: {full_name}")
        except Exception as e:
            logger.error(f"Error storing summary for CPP class {full_name}: {e}")
            raise FileProcessingError(f"Error storing summary for CPP class {full_name}: {e}")
            
            
        
    async def store_summary_cpp_method(self, full_name, info, project_id):
        logger.info(f"Storing summary for method: {full_name}")
        try:
            type_name = info.get('type', 'Method')
            
            query = """
                MERGE (n:{type} {{name: $full_name, project_id: $project_id}})
                ON CREATE SET n.namespace = $namespace,
                              n.short_name = $short_name,
                              n.scope = $scope,
                              n.summary = $summary,
                              n.embedding = $embedding,
                              n.file_path = $file_path,
                              n.raw_code = $raw_code
                ON MATCH SET
                    n.summary = $summary,
                    n.embedding = $embedding
                RETURN elementId(n)
            """.format(type=type_name)
            
            params = {
                'full_name': full_name,
                'namespace': info.get('namespace', ''),
                'short_name': info.get('short_name', ''),
                'project_id': project_id if project_id else 'Unknown',
                'scope': info.get('scope', ''),
                'summary': info.get('summary', ''),
                'embedding': info.get('embedding', []),
                'file_path': info.get('file_path', ''),
                'raw_code': info.get('raw_code', '')
            }
            
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    element_id = await self.run_query_and_get_element_id(session, query, **params)
                    
                    # Create relationship between class and method
                    class_scope = info.get('scope', '')
                    if class_scope:
                        await session.run(
                            "MATCH (c) "
                            "WHERE (c.name = $class_full_name AND c:Class AND c.project_id = $project_id) "
                            "OR (c.name = $class_full_name AND c:Struct AND c.project_id = $project_id) "
                            "MATCH (m:{type}) WHERE elementId(m) = $element_id AND m.project_id = $project_id "
                            "MERGE (c)-[:HAS_METHOD]->(m) "
                            "MERGE (m)-[:BELONGS_TO]->(c)".format(type=type_name),
                            {"class_full_name": class_scope, "element_id": element_id, "project_id": project_id}
                        )
                    
                    logger.info(f"Finished storing CPP method: {full_name}")
        except Exception as e:
            logger.error(f"Error storing summary for CPP method {full_name}: {e}")
            raise FileProcessingError(f"Error storing summary for CPP method {full_name}: {e}")
    
    
        
    async def store_summary_cpp_function(self, full_name, info, project_id):
        logger.info(f"Storing summary for function: {full_name}")
        try:
            type_name = info.get('type', 'Function')
            
            query = """
                MERGE (n:{type} {{name: $full_name, project_id: $project_id}})
                ON CREATE SET n.namespace = $namespace,
                              n.short_name = $short_name,
                              n.scope = $scope,
                              n.summary = $summary,
                              n.embedding = $embedding,
                              n.file_path = $file_path,
                              n.raw_code = $raw_code
                ON MATCH SET
                    n.summary = $summary,
                    n.embedding = $embedding
                RETURN elementId(n)
            """.format(type=type_name)
            
            params = {
                'full_name': full_name,
                'namespace': info.get('namespace', ''),
                'short_name': info.get('short_name', ''),
                'project_id': project_id if project_id else 'Unknown',
                'scope': info.get('scope', ''),
                'summary': info.get('summary', ''),
                'embedding': info.get('embedding', []),
                'file_path': info.get('file_path', ''),
                'raw_code': info.get('raw_code', '')
            }
            
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    await self.run_query_and_get_element_id(session, query, **params)
                    logger.info(f"Finished storing CPP function: {full_name}")
        except Exception as e:
            logger.error(f"Error storing summary for CPP function {full_name}: {e}")
            raise FileProcessingError(f"Error storing summary for CPP function {full_name}: {e}")
        
        
        
    def get_full_scope(self, node):
        scopes = []
        current = node.semantic_parent
        while current and current.kind != clang.cindex.CursorKind.TRANSLATION_UNIT:
            scopes.append(current.spelling)
            current = current.semantic_parent
        return "::".join(reversed(scopes))

    async def IngestPython(self, inputPath: str, inputLocation: str, inputName: str, localPath: str, currentOutputPath: str, project_id: str) -> None:
        """
        Ingest a Python file by parsing it, summarizing classes and functions, and storing them in the database.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            project_id (str): The project ID.
        """
        try:
            async with aiofiles.open(inputPath, 'r') as file:
                content = await file.read()
                tree = ast.parse(content, filename=inputPath)

            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

            for cls in classes:
                class_summary = await self.summarize_python_class(cls)
                embedding = await self.make_embedding(class_summary)
                        
                async with self.rate_limiter_db:
                    async with self.get_session() as session:
                        class_id = await self.run_query_and_get_element_id(session,
                            "CREATE (n:Class {name: $name, summary: $summary, embedding: $embedding, project_id: $project_id}) RETURN elementId(n)",
                            name=cls.name, summary=class_summary, embedding=embedding, project_id=project_id
                        )

                for func in cls.body:
                    if isinstance(func, ast.FunctionDef):
                        function_summary = await self.summarize_python_function(func)
                        embedding = await self.make_embedding(function_summary)
                        
                        async with self.rate_limiter_db:
                            async with self.get_session() as session:
                                function_id = await self.run_query_and_get_element_id(session,
                                    "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, project_id: $project_id}) RETURN elementId(n)",
                                    name=func.name, summary=function_summary, embedding=embedding, project_id=project_id
                                )
                                await (await session.run(
                                    "MATCH (c:Class {elementId: $class_id, project_id: $project_id}) "
                                    "MATCH (f:Function {elementId: $function_id, project_id: $project_id}) "
                                    "CREATE (c)-[:HAS_METHOD]->(f)",
                                    class_id=class_id, function_id=function_id, project_id=project_id
                                )).consume()

            for func in functions:
                function_summary = await self.summarize_python_function(func)
                embedding = await self.make_embedding(function_summary)
                
                async with self.rate_limiter_db:
                    async with self.get_session() as session:
                        await (await session.run(
                            "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, project_id: $project_id})",
                            name=func.name, summary=function_summary, embedding=embedding, project_id=project_id
                        )).consume()
        except Exception as e:
            logger.error(f"Error ingesting Python file {inputPath}: {e}")
            raise FileProcessingError(f"Error ingesting Python file {inputPath}: {e}")

    async def summarize_python_class(self, cls) -> str:
        """
        Summarize a Python class.

        Args:
            cls: The class node.

        Returns:
            str: The summary of the class.
        """
        description = f"Class {cls.name} with methods: "
        methods = [func.name for func in cls.body if isinstance(func, ast.FunctionDef)]
        description += ", ".join(methods)
        return await self.do_summarize_text(description)

    async def summarize_python_function(self, func) -> str:
        """
        Summarize a Python function.

        Args:
            func: The function node.

        Returns:
            str: The summary of the function.
        """
        description = f"Function {func.name} with arguments: {', '.join(arg.arg for arg in func.args.args)}. It performs the following tasks: "
        return await self.do_summarize_text(description)

    async def IngestRust(self, inputPath: str, inputLocation: str, inputName: str, localPath: str, currentOutputPath: str, project_id: str) -> None:
        """
        Ingest a Rust file by parsing it, summarizing implementations and functions, and storing them in the database.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            project_id (str): The project ID.
        """
        try:
            async with aiofiles.open(inputPath, 'r') as file:
                content = await file.read()
                tree = syn.parse_file(content)
            functions = [item for item in tree.items if isinstance(item, syn.ItemFn)]
            impls = [item for item in tree.items if isinstance(item, syn.ItemImpl)]

            for impl in impls:
                impl_summary = await self.summarize_rust_impl(impl)
                embedding = await self.make_embedding(impl_summary)
                
                async with self.rate_limiter_db:
                    async with self.get_session() as session:
                        impl_id = await self.run_query_and_get_element_id(session,
                            "CREATE (n:Impl {name: $name, summary: $summary, embedding: $embedding, project_id: $project_id}) RETURN elementId(n)",
                            name=impl.trait_.path.segments[0].ident, summary=impl_summary, embedding=embedding, project_id=project_id
                        )

                for item in impl.items:
                    if isinstance(item, syn.ImplItemMethod):
                        function_summary = await self.summarize_rust_function(item)
                        embedding = await self.make_embedding(function_summary)
                        
                        async with self.rate_limiter_db:
                            async with self.get_session() as session:
                                function_id = await self.run_query_and_get_element_id(session,
                                    "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, project_id: $project_id}) RETURN elementId(n)",
                                    name=item.sig.ident, summary=function_summary, embedding=embedding, project_id=project_id
                                )
                                await (await session.run(
                                    "MATCH (i:Impl {elementId: $impl_id, project_id: $project_id}), "
                                    "MATCH (f:Function {elementId: $function_id, project_id: $project_id}) "
                                    "CREATE (i)-[:HAS_METHOD]->(f)",
                                    impl_id=impl_id, function_id=function_id, project_id=project_id
                                )).consume()

            for func in functions:
                function_summary = await self.summarize_rust_function(func)
                embedding = await self.make_embedding(function_summary)
                
                async with self.rate_limiter_db:
                    async with self.get_session() as session:
                        await (await session.run(
                            "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, project_id: $project_id})",
                            name=func.sig.ident, summary=function_summary, embedding=embedding, project_id=project_id
                        )).consume()
        except Exception as e:
            logger.error(f"Error ingesting Rust file {inputPath}: {e}")
            raise FileProcessingError(f"Error ingesting Rust file {inputPath}: {e}")

    async def summarize_rust_impl(self, impl) -> str:
        """
        Summarize a Rust implementation block.

        Args:
            impl: The implementation node.

        Returns:
            str: The summary of the implementation.
        """
        description = f"Implementation of trait {impl.trait_.path.segments[0].ident} with methods: "
        methods = [item.sig.ident for item in impl.items if isinstance(item, syn.ImplItemMethod)]
        description += ", ".join([str(m) for m in methods])
        return await self.do_summarize_text(description)

    async def summarize_rust_function(self, func) -> str:
        """
        Summarize a Rust function.

        Args:
            func: The function node.

        Returns:
            str: The summary of the function.
        """
        description = f"Function {func.sig.ident} with arguments: {', '.join(arg.pat.ident for arg in func.sig.inputs)}. It performs the following tasks: "
        return await self.do_summarize_text(description)

    async def summarize_js_function(self, func_node) -> str:
        """
        Summarize a JavaScript function.

        Args:
            func_node: The function node.

        Returns:
            str: The summary of the function.
        """
        func_name = func_node.id.name if func_node.id else 'anonymous'
        description = f"Function {func_name} with arguments: {', '.join(param.name for param in func_node.params)}."
        return await self.do_summarize_text(description)

    async def summarize_js_variable(self, var_node) -> str:
        """
        Summarize a JavaScript variable.

        Args:
            var_node: The variable node.

        Returns:
            str: The summary of the variable.
        """
        descriptions = []
        for declaration in var_node.declarations:
            var_name = declaration.id.name
            var_type = var_node.kind
            description = f"Variable {var_name} of type {var_type}."
            descriptions.append(description)
        return await self.do_summarize_text(" ".join(descriptions))

    async def summarize_js_class(self, class_node) -> str:
        """
        Summarize a JavaScript class.

        Args:
            class_node: The class node.

        Returns:
            str: The summary of the class.
        """
        description = f"Class {class_node.id.name} with methods: "
        methods = [method.key.name for method in class_node.body.body if isinstance(method, nodes.MethodDefinition)]
        description += ", ".join(methods)
        return await self.do_summarize_text(description)

    async def IngestJavascript(self, inputPath: str, inputLocation: str, inputName: str, localPath: str, currentOutputPath: str, project_id: str) -> None:
        """
        Ingest a JavaScript file by parsing it, summarizing classes, functions, and variables, and storing them in the database.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            project_id (str): The project ID.
        """
        try:
            async with aiofiles.open(inputPath, 'r') as file:
                content = await file.read()

            ast = esprima.parseModule(content, {'loc': True, 'range': True})
        except Exception as e:
            logger.error(f"Error parsing JavaScript file {inputPath}: {e}")
            raise FileProcessingError(f"Error parsing JavaScript file {inputPath}: {e}")

        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    file_id = await self.run_query_and_get_element_id(session,
                        "CREATE (n:JavaScriptFile {name: $name, path: $path, project_id: $project_id}) RETURN elementId(n)",
                        name=inputName, path=inputPath, project_id=project_id
                    )

            for node in ast.body:
                if isinstance(node, nodes.FunctionDeclaration):
                    await self.process_js_function(node, file_id, project_id)
                elif isinstance(node, nodes.ClassDeclaration):
                    await self.process_js_class(node, file_id, project_id)
                elif isinstance(node, nodes.VariableDeclaration):
                    await self.process_js_variable(node, file_id, project_id)
        except Exception as e:
            logger.error(f"Error ingesting JavaScript file {inputPath}: {e}")
            raise DatabaseError(f"Error ingesting JavaScript file {inputPath}: {e}")

    async def process_js_function(self, func_node, file_id: int, project_id: str) -> None:
        func_name = func_node.id.name if func_node.id else 'anonymous'
        func_summary = await self.summarize_js_function(func_node)
        
        embedding = await self.make_embedding(func_summary)

        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    func_db_id = await self.run_query_and_get_element_id(session,
                        "CREATE (n:JavaScriptFunction {name: $name, summary: $summary, embedding: $embedding, project_id: $project_id}) RETURN elementId(n)",
                        name=func_name, summary=func_summary, embedding=embedding, project_id=project_id
                    )

                    await (await session.run(
                        "MATCH (f:JavaScriptFile {elementId: $file_id, project_id: $project_id}), (func:JavaScriptFunction {elementId: $func_id, project_id: $project_id}) "
                        "CREATE (f)-[:CONTAINS]->(func)",
                        file_id=file_id, func_id=func_db_id, project_id=project_id
                    )).consume()
        except Exception as e:
            logger.error(f"Error processing JavaScript function {func_name}: {e}")
            raise DatabaseError(f"Error processing JavaScript function {func_name}: {e}")

    async def process_js_class(self, class_node, file_id: int, project_id: str) -> None:
        class_name = class_node.id.name
        class_summary = await self.summarize_js_class(class_node)
        
        embedding = await self.make_embedding(class_summary)

        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    class_db_id = await self.run_query_and_get_element_id(session,
                        "CREATE (n:JavaScriptClass {name: $name, summary: $summary, embedding: $embedding, project_id: $project_id}) RETURN elementId(n)",
                        name=class_name, summary=class_summary, embedding=embedding, project_id=project_id
                    )

                    await (await session.run(
                        "MATCH (f:JavaScriptFile {elementId: $file_id, project_id: $project_id}), (c:JavaScriptClass {elementId: $class_id, project_id: $project_id}) "
                        "CREATE (f)-[:CONTAINS]->(c)",
                        file_id=file_id, class_id=class_db_id, project_id=project_id
                    )).consume()

            for method in class_node.body.body:
                if isinstance(method, nodes.MethodDefinition):
                    await self.process_js_method(method, class_db_id, project_id)
        except Exception as e:
            logger.error(f"Error processing JavaScript class {class_name}: {e}")
            raise DatabaseError(f"Error processing JavaScript class {class_name}: {e}")

    async def process_js_variable(self, var_node, file_id: int, project_id: str) -> None:
        for declaration in var_node.declarations:
            var_name = declaration.id.name
            var_type = var_node.kind  # 'var', 'let', or 'const'
            var_summary = await self.summarize_js_variable(var_node)
            
            embedding = await self.make_embedding(var_summary)

            try:
                async with self.rate_limiter_db:
                    async with self.get_session() as session:
                        var_db_id = await self.run_query_and_get_element_id(session,
                            "CREATE (n:JavaScriptVariable {name: $name, type: $type, summary: $summary, embedding: $embedding, project_id: $project_id}) RETURN elementId(n)",
                            name=var_name, type=var_type, summary=var_summary, embedding=embedding, project_id=project_id
                        )

                        await (await session.run(
                            "MATCH (f:JavaScriptFile {elementId: $file_id, project_id: $project_id}), (v:JavaScriptVariable {elementId: $var_id, project_id: $project_id}) "
                            "CREATE (f)-[:CONTAINS]->(v)",
                            file_id=file_id, var_id=var_db_id, project_id=project_id
                        )).consume()
            except Exception as e:
                logger.error(f"Error processing JavaScript variable {var_name}: {e}")
                raise DatabaseError(f"Error processing JavaScript variable {var_name}: {e}")

    async def process_js_method(self, method_node, class_id: int, project_id: str) -> None:
        """
        Process a JavaScript method node and create nodes in the database.

        Args:
            method_node: The method node.
            class_id: The class ID.
            project_id: The project ID.
        """
        method_name = method_node.key.name
        method_summary = await self.summarize_js_function(method_node.value)
        
        embedding = await self.make_embedding(method_summary)

        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    method_db_id = await self.run_query_and_get_element_id(session,
                        "CREATE (n:JavaScriptMethod {name: $name, summary: $summary, embedding: $embedding, project_id: $project_id}) RETURN elementId(n)",
                        name=method_name, summary=method_summary, embedding=embedding, project_id=project_id
                    )

                    await (await session.run(
                        "MATCH (c:JavaScriptClass {elementId: $class_id, project_id: $project_id}), (m:JavaScriptMethod {elementId: $method_id, project_id: $project_id}) "
                        "CREATE (c)-[:HAS_METHOD]->(m)",
                        class_id=class_id, method_id=method_db_id, project_id=project_id
                    )).consume()
        except Exception as e:
            logger.error(f"Error processing JavaScript method {method_name}: {e}")
            raise DatabaseError(f"Error processing JavaScript method {method_name}: {e}")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=10), retry=retry_if_exception_type(ServiceUnavailable))
    async def run_query_and_get_element_id(self, session, query: str, **parameters) -> str:
        try:
            result = await session.run(query, **parameters)  # Unpack parameters as keyword arguments
            record = await result.single()  # Ensure this is awaited
            if record:
                return record['elementId(n)']  # Ensure 'elementId(n)' matches the actual returned field
            else:
                raise FileProcessingError("No elementId returned from query")
        except Exception as e:
            logger.error(f"Error running query: {query}")
            raise DatabaseError(f"Error running query: {e}")

    async def IngestPdf(self, inputPath: str, inputLocation: str, inputName: str, localPath: str, currentOutputPath: str, project_id: str) -> None:
        try:
            reader = PdfReader(inputPath)
            num_pages = len(reader.pages)

            # Extract metadata
            metadata = reader.metadata
            author = metadata.get('/Author', 'Unknown')
            title = metadata.get('/Title', inputName)

            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    pdf = await PDF.create_in_database(session=session, project_id=project_id, full_path=localPath, page_count=num_pages, author=author, title=title)
                    if pdf:
                        pdf_id = pdf.id
                    else:
                        print("Failed to create PDF")
                        return  # Exit the function if PDF creation fails
            await self.process_pdf_pages(reader, num_pages, pdf_id, project_id)

        except Exception as e:
            logger.error(f"Error ingesting PDF file {inputPath}: {e}")
            raise FileProcessingError(f"Error ingesting PDF file {inputPath}: {e}")

    async def process_pdf_pages(self, reader: PdfReader, num_pages: int, pdf_id: str, project_id: str) -> None:
        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()

            medium_chunks = self.chunk_text(text, MEDIUM_CHUNK_SIZE)
            for j, medium_chunk in enumerate(medium_chunks):
                async with self.rate_limiter_db:
                    async with self.get_session() as session:
                        medium_chunk_id = await self.run_query_and_get_element_id(session,
                            "CREATE (n:MediumChunk {content: $content, type: 'medium_chunk', page: $page, pdf_id: $pdf_id, project_id: $project_id}) RETURN elementId(n)",
                            content=medium_chunk, page=i + 1, pdf_id=pdf_id, project_id=project_id
                        )

                        if medium_chunk_id:
                            await (await session.run(
                                "MATCH (p:File:Document:PDF) WHERE p.id = $pdf_id "
                                "MATCH (m:MediumChunk) WHERE elementId(m) = $medium_chunk_id "
                                "MERGE (p)-[:HAS_CHUNK]->(m) "
                                "MERGE (m)-[:BELONGS_TO]->(p)",
                                {"pdf_id": pdf_id, "medium_chunk_id": medium_chunk_id}
                            )).consume()

                small_chunks = self.chunk_text(medium_chunk, SMALL_CHUNK_SIZE)
                
                # Handle edge case: if no small chunks, create one with the entire medium chunk content
                if not small_chunks:
                    small_chunks = [medium_chunk]

                for k, small_chunk in enumerate(small_chunks):
                    embedding = await self.make_embedding(small_chunk)
                    
                    async with self.rate_limiter_db:
                        async with self.get_session() as session:
                            small_chunk_id = await self.run_query_and_get_element_id(session,
                                "CREATE (n:SmallChunk {content: $content, type: 'small_chunk', page: $page, pdf_id: $pdf_id, project_id: $project_id, embedding: $embedding}) RETURN elementId(n)",
                                content=small_chunk, page=i + 1, pdf_id=pdf_id, project_id=project_id, embedding=embedding
                            )

                            if small_chunk_id:
                                await (await session.run(
                                    "MATCH (p:File:Document:PDF) WHERE p.id = $pdf_id "
                                    "MATCH (m:MediumChunk) WHERE elementId(m) = $medium_chunk_id "
                                    "MATCH (s:SmallChunk) WHERE elementId(s) = $small_chunk_id "
                                    "MERGE (m)-[:HAS_CHUNK]->(s) "
                                    "MERGE (s)-[:BELONGS_TO]->(m) "
                                    "MERGE (s)-[:HAS_PARENT_DOC]->(p)",
                                    {"pdf_id": pdf_id, "medium_chunk_id": medium_chunk_id, "small_chunk_id": small_chunk_id}
                                )).consume()
                                
    async def getFileAndCppCount(self) -> tuple[int, int]:
        try:
            total_cpp_count = await self.cpp_processor.get_cpp_count()
            file_count = await self.get_file_paths_length()
            return file_count, total_cpp_count
        except Exception as e:
            logger.error(f"Error getting lengths of local members: {e}")
            raise FileProcessingError(f"Error getting lengths of local members: {e}")

    def chunk_text(self, text: str, chunk_size: int) -> List[str]:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# End of NNeo4JImporter class

