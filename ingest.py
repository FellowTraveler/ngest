# Copyright 2024 Chris Odom
# MIT License

import os
import datetime
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import uuid
import logging
from abc import ABC, abstractmethod
import asyncio
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable
import PIL.Image
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import pipeline, AutoTokenizer, AutoModel, BertModel, BertTokenizer
from clang.cindex import Config
Config.set_library_path("/opt/homebrew/opt/llvm/lib")
import clang.cindex
import ast
import syn
from PyPDF2 import PdfReader
from typing import List, Dict, Any, Union, Optional, Generator
import esprima
from esprima import nodes
import argparse
import configparser
from contextlib import asynccontextmanager
import torch
from tqdm.asyncio import tqdm
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import dotenv
import tempfile
import unittest
import json
import aiofiles
import shutil
from pathlib import Path
import fnmatch
import aiorate
import ollama
from aiolimiter import AsyncLimiter
from typing import AsyncGenerator
from typing import Optional, List, Dict, Any, Union, Tuple
import magic
from collections import defaultdict
import copy
import pprint

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
#DEFAULT_NEO4J_URL = "bolt://localhost:7687"
DEFAULT_NEO4J_URL = "bolt://localhost:7689"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "mynewpassword"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


if not os.path.exists(config_dir):
    os.makedirs(config_dir)

if not os.path.isfile(config_file):
    with open(config_file, 'w') as f:
        f.write(f"[Limits]\nMAX_FILE_SIZE = {DEFAULT_MAX_FILE_SIZE}\n\n"
                f"[Chunks]\nMEDIUM_CHUNK_SIZE = {DEFAULT_MEDIUM_CHUNK_SIZE}\nSMALL_CHUNK_SIZE = {DEFAULT_SMALL_CHUNK_SIZE}\n\n"
                f"[Database]\nNEO4J_URL = {DEFAULT_NEO4J_URL}\nNEO4J_USER = {DEFAULT_NEO4J_USER}\nNEO4J_PASSWORD = {DEFAULT_NEO4J_PASSWORD}\n\n"
                f"[Ollama]\nOLLAMA_URL = {DEFAULT_OLLAMA_URL}\n\n"
                f"[Models]\nEMBEDDING_MODEL = {DEFAULT_EMBEDDING_MODEL}")
else:
    config.read(config_file)

MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', config.get('Limits', 'MAX_FILE_SIZE', fallback=DEFAULT_MAX_FILE_SIZE)))
MEDIUM_CHUNK_SIZE = int(os.getenv('MEDIUM_CHUNK_SIZE', config.get('Chunks', 'MEDIUM_CHUNK_SIZE', fallback=DEFAULT_MEDIUM_CHUNK_SIZE)))
SMALL_CHUNK_SIZE = int(os.getenv('SMALL_CHUNK_SIZE', config.get('Chunks', 'SMALL_CHUNK_SIZE', fallback=DEFAULT_SMALL_CHUNK_SIZE)))
NEO4J_URL = os.getenv('NEO4J_URL', config.get('Database', 'NEO4J_URL', fallback=DEFAULT_NEO4J_URL))
NEO4J_USER = os.getenv('NEO4J_USER', config.get('Database', 'NEO4J_USER', fallback=DEFAULT_NEO4J_USER))
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', config.get('Database', 'NEO4J_PASSWORD', fallback=DEFAULT_NEO4J_PASSWORD))
OLLAMA_URL = os.getenv('OLLAMA_URL', config.get('Ollama', 'OLLAMA_URL', fallback=DEFAULT_OLLAMA_URL))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', config.get('Models', 'EMBEDDING_MODEL', fallback=DEFAULT_EMBEDDING_MODEL))

# Configure logging with different levels
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the logging level based on a condition (e.g., a command-line argument or an environment variable)
hide_info_logs = False  # Set this to True to hide info logs

if hide_info_logs:
    logging.getLogger().setLevel(logging.WARNING)
else:
    logging.getLogger().setLevel(logging.INFO)

def ollama_generate_embedding_sync(prompt: str) -> List[float]:
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=prompt)
    if 'embedding' in response:
        return response['embedding']
    else:
        logger.error(f"Error generating embedding: {response}")
        return []

async def ollama_generate_embedding(prompt: str) -> List[float]:
    return await asyncio.to_thread(ollama_generate_embedding_sync, prompt)

class CustomError(Exception):
    """Base class for custom exceptions"""
    pass

class FileProcessingError(CustomError):
    """Raised when there's an error processing a file"""
    pass

class DatabaseError(CustomError):
    """Raised when there's an error with database operations"""
    pass

class ConfigurationError(CustomError):
    """Raised when there's an error with the configuration"""
    pass

class NBaseImporter(ABC):
    """
    Abstract base class for file importers. Provides methods to determine file types
    and handle file chunking and graph node creation.
    """
    @abstractmethod
    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, topLevelInputPath: str, topLevelOutputPath: str, currentOutputPath: str, projectID: str) -> None:
        pass

    def ascertain_file_type(self, inputPath: str) -> dict:
        try:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(inputPath)
            _, ext = os.path.splitext(inputPath)
            ext = ext.lower() if ext else ''

            # Get file stats
            file_stats = os.stat(inputPath)
            size_in_bytes = file_stats.st_size
            created_date = datetime.datetime.fromtimestamp(file_stats.st_birthtime).isoformat() if hasattr(file_stats, 'st_birthtime') else None
            modified_date = datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat()

            # Determine file type based on MIME type and extension
            if file_type.startswith('text'):
                if ext in ['.cpp', '.hpp', '.h', '.c']:
                    type_name = 'cpp'
                elif ext == '.py':
                    type_name = 'python'
                elif ext == '.rs':
                    type_name = 'rust'
                elif ext == '.js':
                    type_name = 'javascript'
                else:
                    type_name = 'text'
            elif file_type.startswith('image'):
                type_name = 'image'
            elif file_type == 'application/pdf':
                type_name = 'pdf'
            else:
                type_name = 'unknown'

            return {
                'type': type_name,
                'extension': ext,
                'size_in_bytes': size_in_bytes,
                'created_date': created_date,
                'modified_date': modified_date
            }
        except Exception as e:
            logger.error(f"Error determining file type for {inputPath}: {e}")
            return {
                'type': 'unknown',
                'extension': '',
                'size_in_bytes': 0,
                'created_date': None,
                'modified_date': None
            }

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=4, max=10))
    async def create_graph_nodes(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        """
        Create graph nodes in the Neo4j database for a given file.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            projectID (str): The project ID.
        """
        logger.info(f"Creating graph nodes for {inputPath}")
        # Implement the actual graph node creation logic here

    async def chunk_and_create_nodes(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        try:
            chunks = [chunk async for chunk in read_file_in_chunks(inputPath, MEDIUM_CHUNK_SIZE)]
            previous_chunk_id = None

            for i, chunk in enumerate(chunks):
                chunk_id = await self.handle_chunk_creation(chunk, inputPath, i, projectID)
                if previous_chunk_id:
                    await self.create_chunk_relationship(previous_chunk_id, chunk_id, projectID)
                previous_chunk_id = chunk_id
        except Exception as e:
            logger.error(f"Error processing file {inputPath}: {e}")
            raise FileProcessingError(f"Error processing file {inputPath}: {e}")


    def chunk_text(self, text: str, chunk_size: int) -> List[str]:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]



#    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=15, max=30))
    async def create_chunk_node(self, chunk: str, inputPath: str, index: int, projectID: str) -> Optional[str]:
        """
        Create a chunk node in the Neo4j database.

        Args:
            chunk (str): The text chunk.
            inputPath (str): The path to the input file.
            index (int): The index of the chunk.
            projectID (str): The project ID.

        Returns:
            Optional[str]: The chunk ID, or None if creation failed.
        """
        try:
            chunk_id = f"{inputPath}_chunk_{index}"

            # Check if the chunk node already exists
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    existing_node = await self.run_query_and_get_element_id(session,
                        "MATCH (n:Chunk {elementId: $id}) RETURN elementId(n)",
                        id=chunk_id
                    )
                if existing_node:
                    logger.info(f"Chunk node {chunk_id} already exists, skipping creation")
                    return chunk_id

            embedding = await self.make_embedding(chunk)
                
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    node_id = await self.run_query_and_get_element_id(session,
                        "CREATE (n:Chunk {elementId: $id, content: $content, embedding: $embedding, projectID: $projectID}) RETURN elementId(n)",
                        id=chunk_id, content=chunk, embedding=embedding, projectID=projectID
                    )
                    if node_id:
                        logger.info(f"Created chunk node {chunk_id} with embedding")
                        return chunk_id
                    else:
                        logger.error(f"No result returned for chunk node {chunk_id}")
                        return None
        except Exception as e:
            logger.error(f"Error creating chunk node: {e}")
            raise DatabaseError(f"Error creating chunk node: {e}")


    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=4, max=10))
    async def create_chunk_relationship(self, chunk_id1: str, chunk_id2: str, projectID: str) -> None:
        """
        Create a relationship between two chunks in the Neo4j database.

        Args:
            chunk_id1 (str): The ID of the first chunk.
            chunk_id2 (str): The ID of the second chunk.
            projectID (str): The project ID.
        """
        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    result = await (await session.run(
                        "MATCH (c1:Chunk {elementId: $id1, projectID: $projectID}), (c2:Chunk {elementId: $id2, projectID: $projectID}) CREATE (c1)-[:NEXT]->(c2)",
                        id1=chunk_id1, id2=chunk_id2, projectID=projectID
                    )).consume()
                    if result:
                        logger.info(f"Created relationship between {chunk_id1} and {chunk_id2}")
                    else:
                        logger.error(f"No result returned for creating relationship between {chunk_id1} and {chunk_id2}")
        except Exception as e:
            logger.error(f"Error creating chunk relationship: {e}")
            raise DatabaseError(f"Error creating chunk relationship: {e}")


    async def handle_chunk_creation(self, chunk, inputPath, index, projectID):
        """
        Handle the creation of a chunk node.

        Args:
            chunk: The text chunk.
            inputPath: The path to the input file.
            index: The index of the chunk.
            projectID: The project ID.

        Returns:
            The chunk ID.
        """
        try:
            chunk_id = await self.create_chunk_node(chunk, inputPath, index, projectID)
            return chunk_id
        except Exception as e:
            logger.error(f"Error creating chunk node for chunk {index} in file {inputPath}: {e}")
            raise DatabaseError(f"Error creating chunk node for chunk {index} in file {inputPath}: {e}")

class NFilesystemImporter(NBaseImporter):
    """
    A file importer that simply copies files to the output directory.
    """
    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, topLevelInputPath: str, topLevelOutputPath: str, currentOutputPath: str, projectID: str) -> None:
        """
        Ingest a file by copying it to the output directory.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            projectID (str): The project ID.
        """
        if os.path.getsize(inputPath) > MAX_FILE_SIZE:
            logger.info(f"File {inputPath} skipped due to size.")
            open(os.path.join(currentOutputPath, f"{inputName}.skipped"), 'a').close()
            return

        try:
            shutil.copy(inputPath, os.path.join(currentOutputPath, inputName))
            logger.info(f"File {inputPath} ingested successfully.")
        except Exception as e:
            logger.error(f"Error ingesting file {inputPath}: {e}")
            raise FileProcessingError(f"Error ingesting file {inputPath}: {e}")

class NNeo4JImporter(NBaseImporter):
    """
    A file importer that ingests various file types into a Neo4j database.
    """
    def __init__(self, neo4j_url: str = NEO4J_URL, neo4j_user: str = NEO4J_USER, neo4j_password: str = NEO4J_PASSWORD):
        self.neo4j_url = neo4j_url
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = AsyncGraphDatabase.driver(self.neo4j_url, auth=(self.neo4j_user, self.neo4j_password))
        logger.info(f"Neo4J importer initialized with URL {neo4j_url}")

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        logging.getLogger("transformers").setLevel(logging.ERROR)

        self.semaphore_summarizer = asyncio.Semaphore(1)  # Limit to 1 concurrent tasks
        self.semaphore_embedding = asyncio.Semaphore(1)  # Limit to 1 concurrent tasks

        self.lock_summarizer = asyncio.Lock()
        self.summarizer = pipeline("summarization", model="google/pegasus-xsum", device=self.device)
#        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=self.device)
        
        self.image_model = models.densenet121(pretrained=True).to(self.device)
        self.image_model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.code_model = AutoModel.from_pretrained("microsoft/codebert-base").to(self.device)
        self.code_model.eval()

        self.rate_limiter_db = AsyncLimiter(50, 1)  # 1 operations per 2 seconds
        self.rate_limiter_summary = AsyncLimiter(2, 1)  # 2 operations per 1 seconds
        self.rate_limiter_embedding = AsyncLimiter(2, 1)  # 2 operations per 1 seconds

        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_model.eval()
        
        self.classes = defaultdict(dict)
        self.functions = defaultdict(dict)
        self.methods = defaultdict(dict)
        self.header_files = {}

        self.file_paths = []
        self.file_paths_lock = asyncio.Lock()

        self.lock_classes = asyncio.Lock()  # Create a lock instance
        self.lock_functions = asyncio.Lock()  # Create a lock instance
        self.lock_methods = asyncio.Lock()  # Create a lock instance
        self.lock_file_paths = asyncio.Lock()  # Create a lock instance
        self.lock_header_files = asyncio.Lock()  # Create a lock instance

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
        # this only limits the creation of new threads.
        async with self.rate_limiter_embedding:
            return await self.generate_embedding(text)

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=15, max=30))
    async def generate_embedding(self, text: str) -> List[float]:
        async with self.semaphore_embedding:  # Ensure only 5 concurrent tasks
            try:
                loop = asyncio.get_running_loop()
#                logger.info(f"Started generating an embedding.")
                embedding = await loop.run_in_executor(None, self._generate_embedding, text)
#                logger.info(f"Finished generating an embedding.")
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


    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, topLevelInputPath: str, topLevelOutputPath: str, currentOutputPath: str, projectID: str) -> None:
        """
        Ingest a file by determining its type and using the appropriate ingestion method.

        Completed scanning for:
        inputPath:          /Users/au/src/blindsecp/sec2-v2.pdf
        inputLocation:      /Users/au/src/blindsecp
        inputName:          sec2-v2.pdf
        topLevelOutputPath: /Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3
        currentOutputPath:  /Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3/blindsecp
        projectID:          02688845-0e74-4c5f-9b50-af4772dca5e3

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            projectID (str): The project ID.
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
            file_type = self.ascertain_file_type(inputPath)
            ingest_method = getattr(self, f"Ingest{file_type.capitalize()}", None)
            if ingest_method:
                
                await self.add_file_path(inputPath)

                localPath = strip_top_level(inputPath, topLevelInputPath)
            
                # localPath                 == "blindsecp/sec2-v2.pdf"
                ## RESUME
                
#            APPLICATION: "global"
#             1. projectsFolder:          "/Users/au/.ngest/projects"
#                                                                  "/"
#            PROJECT:  "blindsecp"
#                projectID:                                         "02688845-0e74-4c5f-9b50-af4772dca5e3"
#                topLevelOutputPath:      "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3"
#                topLevelOutputPath:           $projectsFolder     "/"           $projectID
#                projectsFolder:          "/Users/au/.ngest/projects"
#                topLevelOutputPath:      "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3"
#             1. projectName: "blindsecp"                                                                "blindsecp"
#                project_root: "blindsecp"                                                               "blindsecp"
#             2. inputLocation: "/Users/au/src/blindsecp"                                  "/Users/au/src/blindsecp"
#                project_input_location: "/Users/au/src"                                   "/Users/au/src"
#                topLevelOutputPath:      "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3"
#                currentOutputPath:       "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3/blindsecp"
#                topLevelOutputPath:      "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3"
#                topLevelOutputPath:           $projectsFolder     "/"           $projectID
#                projectName:                                                                            "blindsecp"
#                currentOutputPath:            $projectsFolder     "/"           $projectID             "/" $project_root
#             3. projectID:                                         "02688845-0e74-4c5f-9b50-af4772dca5e3"
#                topLevelOutputPath:           $projectsFolder     "/"           $projectID
#                projectName:                                                                            "blindsecp"
#             4. currentOutputPath:       "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3/blindsecp"
#                currentOutputPath:            $projectsFolder     "/"           $projectID             "/" $project_root
#             5. topLevelOutputPath:      "/Users/au/.ngest/projects/02688845-0e74-4c5f-9b50-af4772dca5e3"
#                topLevelOutputPath:           $projectsFolder     "/"           $projectID
#                currentOutputPath:            $projectsFolder     "/"           $projectID             "/blindsecp"
#                currentOutputPath:            $projectsFolder     "/"           $projectID             "/" $project_root
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
#                projectID:                                        "02688845-0e74-4c5f-9b50-af4772dca5e3"
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
                        from Document import Document
                        document = Document(
                            filename=inputName,
                            full_path=localPath,
                            size_in_bytes=file_type['size_in_bytes'],
                            project_id=projectID,
                            created_date=file_type['created_date'],
                            modified_date=file_type['modified_date'],
                            extension=file_type['extension'],
                            content_type=file_type['type']
                        )
                        query, element_id = await document.generate_cypher_query(session)
                        document_id = await self.run_query_and_get_element_id(session, query, element_id=element_id)

#                            "CREATE (n:PDF {name: $name, projectID: $projectID}) RETURN elementId(n)",
#                            "CREATE (n:Document:PDF {name: $name, projectID: $projectID}) RETURN elementId(n)"
                            
                
                await ingest_method(inputPath, inputLocation, inputName, currentOutputPath, projectID)
            else:
                logger.warning(f"No ingest method for file type: {file_type}")
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
        # this only limits the creation of new threads.
        async with self.rate_limiter_summary:
            return await self.summarize_text(text, max_length, min_length)

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=15, max=30))
    async def summarize_text(self, text, max_length=50, min_length=25) -> str:
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
        
    async def process_chunks(self, text, parent_node_id, parent_node_type, chunk_size, chunk_type, projectID):
        """
        Process text chunks and create corresponding nodes in the Neo4j database.

        Args:
            text: The text to be chunked.
            parent_node_id: The ID of the parent node.
            parent_node_type: The type of the parent node.
            chunk_size: The size of each chunk.
            chunk_type: The type of the chunk.
            projectID: The project ID.
        """
        chunks = self.chunk_text(text, chunk_size)
        for i, chunk in enumerate(chunks):
            await self.create_chunk_with_embedding(chunk, parent_node_id, parent_node_type, chunk_type, projectID)

    async def create_chunk_with_embedding(self, chunk, parent_node_id, parent_node_type, chunk_type, projectID):
        """
        Create a chunk node with embedding in the Neo4j database.

        Args:
            chunk: The text chunk.
            parent_node_id: The ID of the parent node.
            parent_node_type: The type of the parent node.
            chunk_type: The type of the chunk.
            projectID: The project ID.
        """
        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    chunk_id = await self.run_query_and_get_element_id(session,
                        f"CREATE (n:{chunk_type} {{content: $content, type: $chunk_type, projectID: $projectID}}) RETURN elementId(n)",
                        content=chunk, chunk_type=chunk_type, projectID=projectID
                    )
                    await (await session.run(
                        f"MATCH (p:{parent_node_type} {{elementId: $parent_node_id, projectID: $projectID}}) "
                        f"MATCH (c:{chunk_type} {{elementId: $chunk_id, projectID: $projectID}}) "
                        f"CREATE (p)-[:HAS_CHUNK]->(c), (c)-[:PART_OF]->(p)",
                        parent_node_id=parent_node_id, chunk_id=chunk_id, projectID=projectID
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
                        "CREATE (n:Embedding {embedding: $embedding, type: 'embedding', projectID: $projectID}) RETURN elementId(n)",
                        embedding=embedding, projectID=projectID
                    )
                    await (await session.run(
                        f"MATCH (c:{chunk_type} {{elementId: $chunk_id, projectID: $projectID}}) "
                        f"MATCH (e:Embedding {{elementId: $embedding_id, projectID: $projectID}}) "
                        f"CREATE (c)-[:HAS_EMBEDDING]->(e)",
                        chunk_id=chunk_id, embedding_id=embedding_id, projectID=projectID
                    )).consume()
        except Exception as e:
            logger.error(f"Error storing chunk with embedding: {e}")
            raise DatabaseError(f"Error storing chunk with embedding: {e}")


    async def IngestText(self, input_path: str, input_location: str, input_name: str, current_output_path: str, project_id: str) -> None:
            try:
                # File reading operation - doesn't need rate limiting
                async with aiofiles.open(input_path, 'r') as file:
                    file_content = await file.read()
            except Exception as e:
                logger.error(f"Error reading file {input_path}: {e}")
                raise FileProcessingError(f"Error reading file {input_path}: {e}")

            try:
                # Database operation - needs rate limiting
                async with self.rate_limiter_db:
                    async with self.get_session() as session:
                        parent_doc_id = await self.run_query_and_get_element_id(session,
                            "CREATE (n:Document {name: $name, type: 'text', projectID: $projectID}) RETURN elementId(n)",
                            name=input_name, projectID=project_id
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

    async def IngestImg(self, input_path: str, input_location: str, input_name: str, current_output_path: str, project_id: str) -> None:
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
                        "CREATE (n:Image {name: $name, type: 'image', projectID: $projectID}) RETURN elementId(n)",
                        name=input_name, projectID=project_id
                    )
                    features_id = await self.run_query_and_get_element_id(session,
                        "CREATE (n:ImageFeatures {features: $features, type: 'image_features', projectID: $projectID}) RETURN elementId(n)",
                        features=features, projectID=project_id
                    )
                    await (await session.run(
                        "MATCH (i:Image), (f:ImageFeatures) WHERE elementId(i) = $image_id AND elementId(f) = $features_id AND i.projectID = $projectID AND f.projectID = $projectID "
                        "CREATE (i)-[:HAS_FEATURES]->(f)",
                        image_id=image_id, features_id=features_id, projectID=project_id).consume()
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

    async def summarize_cpp_class_public_interface(self, class_info) -> (str, str, str, str):
        """
        Summarize the public interface of a C++ class, including inherited public members from base classes,
        and indicate whether they are exported.

        Args:
            class_info: The class node's info.

        Returns:
            tuple: A tuple containing the class name, fully-qualified name, and the summary.
        """
        
        try:
            type_name = class_info.get('type', 'Class')
            full_name = class_info.get('name', 'anonymous')
            full_scope = class_info.get('scope', '')
            short_name = class_info.get('short_name', '')
            raw_code = class_info.get('raw_code', '')
            file_path = class_info.get('file_path', '')
            raw_comment = class_info.get('raw_comment', '')
            description = class_info.get('description', '')
            interface_description = class_info.get('interface_description', raw_code)

            background = (
                "Speaking as a senior developer and software architect, describe the purpose and usage of this class in your own words. "
                "Meditate on the provided description and public interface first, before writing your final summary. "
                "Then enclose those meditations in opening and closing <thoughts> tags, and then write your final summary."
            )
            
            # Combine background and description with clear separation
            full_prompt = f"{background}\n\nClass Details:\n{description}. {interface_description}"
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_public_interface accessing dict: {e}")
            raise DatabaseError(f"Error in summarize_cpp_class_public_interface accessing dict: {e}")

        try:
            summary = await self.do_summarize_text(full_prompt, 200, 25)
            return full_name, full_scope, summary, description + ". " + interface_description
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_public_interface: {e}")
            raise DatabaseError(f"Error in summarize_cpp_class_public_interface: {e}")
                

    async def summarize_cpp_class_implementation(self, class_info) -> (str, str):
        """
        Summarize the implementation details of a C++ class, including private methods, properties, static members,
        and inherited protected members.

        Args:
            class_info: The class node's info

        Returns:
            tuple: A tuple containing the class name, fully-qualified name, and the summary.
        """
        
        try:
            type_name = class_info.get('type', 'Class')
            full_name = class_info.get('name', 'anonymous')
            full_scope = class_info.get('scope', '')
            short_name = class_info.get('short_name', '')
            raw_code = class_info.get('raw_code', '')
            file_path = class_info.get('file_path', '')
            raw_comment = class_info.get('raw_comment', '')
            description = class_info.get('description', '')
            implementation_description = class_info.get('implementation_description', raw_code)

            background = (
                "Speaking as a senior developer and software architect, describe the implementation and inner workings of this class in your own words. "
                "Meditate on the provided description below, before writing your final summary. "
                "Then enclose those meditations in opening and closing <thoughts> tags, and then write your final summary."
            )
            
            # Combine background and description with clear separation
            full_prompt = f"{background}\n\nClass Details: {description}. {implementation_description}"
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_implementation accessing dict: {e}")
            raise DatabaseError(f"Error in summarize_cpp_class_implementation accessing dict: {e}")

        try:
            summary = await self.do_summarize_text(full_prompt, 200, 25)
            return summary, description + ". " + implementation_description
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_implementation: {e}")
            raise DatabaseError(f"Error in summarize_cpp_class_implementation: {e}")


    async def summarize_cpp_class(self, class_info) -> (str, str, str, str):
        """
        Summarize a C++ class.

        Args:
            class_info: The class node's info.

        Returns:
            str: The summary of the class.
        """
        
        try:
#            logger.info(f"Summarizing class public interface: {class_info['name']}")
            class_name, full_scope, interface_summary, interface_description = await self.summarize_cpp_class_public_interface(class_info)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_public_interface for {class_name}: {e}")
            raise DatabaseError(f"Error in summarize_cpp_class_public_interface for {class_name}: {e}")
        try:
#            logger.info(f"Summarizing class implementation: {class_info['name']}")
            implementation_summary, implementation_description = await self.summarize_cpp_class_implementation(class_info)
#            logger.info(f"Finished summarizing/embedding CPP class: {class_name}")
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_implementationfor {class_name}: {e}")
            raise DatabaseError(f"Error in summarize_cpp_class_implementation for {class_name}: {e}")

        return class_name, full_scope, interface_summary + "\n\n" + interface_description, implementation_summary + "\n\n" + implementation_description

    async def summarize_cpp_function(self, node_info) -> (str, str, str):
        """
        Summarize a C++ function.

        Args:
            node_info: The function or method node's info.

        Returns:
            str: The summary of the function or method.
        """
        
        type_name = node_info.get('type', 'Function')
        full_name = node_info.get('name', 'anonymous')
        full_scope = node_info.get('scope', '')
        short_name = node_info.get('short_name', '')
        raw_code = node_info.get('raw_code', '')
        file_path = node_info.get('file_path', '')
        raw_comment = node_info.get('raw_comment', '')
        description = node_info.get('description', '')
        is_cpp_file = node_info.get('is_cpp_file', False)

        if type_name is None:
            logger.error(f"Error in summarize_cpp_function - type_name is None")
            raise DatabaseError(f"Error in summarize_cpp_function: type_name is None")

        # Background for summarization
        background = (
            f"Speaking as a senior developer and software architect, describe the implementation and inner workings of this {type_name} in your own words. "
            "Meditate on the provided description below, before writing your final summary. "
            "Then enclose those meditations in opening and closing <thoughts> tags, and then write your final summary."
        )
        
        # Combine background and description with clear separation
        full_prompt = f"{background}\n\n{type_name} Details:\n{description}"
        
        try:
            summary = await self.do_summarize_text(full_prompt)
#            logger.info(f"Finished summarizing/embedding CPP {type_name}: {full_name}")
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function calling do_summarize_text: {e}")
            raise DatabaseError(f"Error in summarize_cpp_function calling do_summarize_text: {e}")
        
        try:
            retval = full_name, full_scope, summary + "\n\n" + description
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function appending strings: {e}\nnode_info contents:\n{pprint.pformat(node_info, indent=4)}")
            raise DatabaseError(f"Error in summarize_cpp_function appending strings: {e}\nnode_info contents:\n{pprint.pformat(node_info, indent=4)}")

        return retval
#
#    async def IngestCpp(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
#        try:
#            async with self.get_session() as session:
#                logger.info(f"Parsing file: {inputPath}")
#                index = clang.cindex.Index.create()
#                translation_unit = index.parse(inputPath)
#                logger.info(f"File parsed successfully: {inputPath}")
#                await self.process_nodes(session, translation_unit.cursor, inputLocation, projectID)
#        except Exception as e:
#            logger.error(f"Error ingesting C++ file {inputPath}: {e}")
#            raise FileProcessingError(f"Error ingesting C++ file {inputPath}: {e}")
#
#    async def process_nodes(self, session, node, project_path, projectID):
#        for child in node.get_children():
#            file_name = child.location.file.name if child.location.file else ''
#            if project_path in file_name:
#                if child.kind in (clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL):
#                    type_name = ""
#                    if child.kind == clang.cindex.CursorKind.CLASS_DECL:
#                        logger.info(f"Class Declaration found: {child.spelling} at {file_name}")
#                        type_name = "Class"
#                    elif child.kind == clang.cindex.CursorKind.STRUCT_DECL:
#                        logger.info(f"Struct Declaration found: {child.spelling} at {file_name}")
#                        type_name = "Struct"
#
#                    cls = child
#                    class_name, class_scope, interface_summary, implementation_summary = await self.summarize_cpp_class(cls)
#                    fully_qualified_name = f"{class_scope}::{class_name}" if class_scope else class_name
#                    interface_embedding = await self.generate_embedding(interface_summary)
#                    implementation_embedding = await self.generate_embedding(implementation_summary)
#                    async with self.rate_limiter:
#                        class_id = await self.run_query_and_get_element_id(session,
#                            f"""
#                            MERGE (n:{type_name} {{name: $name, projectID: $projectID}})
#                            ON CREATE SET n.scope = $scope,
#                                           n.short_name = $short_name,
#                                           n.interface_summary = $interface_summary,
#                                           n.implementation_summary = $implementation_summary,
#                                           n.interface_embedding = $interface_embedding,
#                                           n.implementation_embedding = $implementation_embedding
#                            ON MATCH SET
#                                n.interface_summary = CASE WHEN size($interface_summary) > size(n.interface_summary) THEN $interface_summary ELSE n.interface_summary END,
#                                n.interface_embedding = CASE WHEN size($interface_summary) > size(n.interface_summary) THEN $interface_embedding ELSE n.interface_embedding END,
#                                n.implementation_summary = CASE WHEN size($implementation_summary) > size(n.implementation_summary) THEN $implementation_summary ELSE n.implementation_summary END,
#                                n.implementation_embedding = CASE WHEN size($implementation_summary) > size(n.implementation_summary) THEN $implementation_embedding ELSE n.implementation_embedding END
#                            RETURN elementId(n)
#                            """,
#                            name=fully_qualified_name,
#                            projectID=projectID,
#                            scope=class_scope,
#                            short_name=cls.spelling,
#                            interface_summary=interface_summary,
#                            implementation_summary=implementation_summary,
#                            interface_embedding=interface_embedding,
#                            implementation_embedding=implementation_embedding
#                        )
#                    struct_id = class_id
#                    
#                    for func in cls.get_children():
#                        # Check if the method belongs to a file within the project directory
#                        if func.kind == clang.cindex.CursorKind.CXX_METHOD:
#                            logger.info(f"Method found: {func.spelling} in {cls.spelling} at {file_name}")
#                            function_name, function_scope, function_summary = await self.summarize_cpp_function(func)
#                            full_func_name = f"{function_scope}::{function_name}" if function_scope else function_name
#                            function_embedding = await self.generate_embedding(function_summary)
#                            async with self.rate_limiter:
#                                function_id = await self.run_query_and_get_element_id(session,
#                                    """
#                                    MERGE (n:Method {name: $name, projectID: $projectID})
#                                    ON CREATE SET n.scope = $scope,
#                                                  n.short_name = $short_name,
#                                                  n.summary = $summary,
#                                                  n.embedding = $embedding
#                                    ON MATCH SET
#                                        n.summary = CASE WHEN size($summary) > size(n.summary) THEN $summary ELSE n.summary END,
#                                        n.embedding = CASE WHEN size($summary) > size(n.summary) THEN $embedding ELSE n.embedding END
#                                    RETURN elementId(n)
#                                    """,
#                                    scope=function_scope if function_scope else "",
#                                    short_name=func.spelling,
#                                    name=full_func_name,
#                                    summary=function_summary,
#                                    embedding=function_embedding,
#                                    projectID=projectID
#                                )
#
#                                await session.run(
#                                    "MATCH (c) WHERE (elementId(c) = $class_id AND c:Class AND c.projectID = $projectID) "
#                                    "OR (elementId(c) = $struct_id AND c:Struct AND c.projectID = $projectID) "
#                                    "MATCH (f:Method) WHERE elementId(f) = $function_id AND f.projectID = $projectID "
#                                    "MERGE (c)-[:HAS_METHOD]->(f) "
#                                    "MERGE (f)-[:BELONGS_TO]->(c)",
#                                    {"class_id": class_id, "struct_id": class_id, "function_id": function_id, "projectID": projectID}
#                                )
#
#                    await self.process_nodes(session, child, project_path, projectID)
#                    
#                elif child.kind == clang.cindex.CursorKind.FUNCTION_DECL:
#                    func = child
#                    logger.info(f"Function found: {func.spelling} at {file_name}")
#                    function_name, function_scope, function_summary = await self.summarize_cpp_function(func)
#                    full_func_name = f"{function_scope}::{function_name}" if function_scope else function_name
#                    function_embedding = await self.generate_embedding(function_summary)
#                    async with self.rate_limiter:
#                        function_id = await self.run_query_and_get_element_id(session,
#                            """
#                            MERGE (n:Function {name: $name, projectID: $projectID})
#                            ON CREATE SET n.scope = $scope,
#                                          n.short_name = $short_name,
#                                          n.summary = $summary,
#                                          n.embedding = $embedding
#                            ON MATCH SET
#                                n.summary = CASE WHEN size($summary) > size(n.summary) THEN $summary ELSE n.summary END,
#                                n.embedding = CASE WHEN size($summary) > size(n.summary) THEN $embedding ELSE n.embedding END
#                            RETURN elementId(n)
#                            """,
#                            scope=function_scope if function_scope else "",
#                            short_name=func.spelling,
#                            name=full_func_name,
#                            summary=function_summary,
#                            embedding=function_embedding,
#                            projectID=projectID
#                        )
#                else:
#                    #logger.warning(f"Something found: {child.spelling} at {file_name}")
#                    # TODO?
#                    await self.process_nodes(session, child, project_path, projectID)
    
    async def IngestCpp(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        try:
#            logger.info(f"Parsing file: {inputPath}")
            index = clang.cindex.Index.create()
            translation_unit = index.parse(inputPath)
#            logger.info(f"File parsed successfully: {inputPath}")

            with open(inputPath, 'r') as file:
                file_contents = file.read()

            # Save raw code of header files
            if inputPath.endswith('.hpp') or inputPath.endswith('.h'):
                    async with self.lock_header_files:
                        self.header_files[inputPath] = file_contents
            
            await self.process_nodes(translation_unit.cursor, inputLocation, projectID, inputPath.endswith('.cpp'))
                
        except Exception as e:
            logger.error(f"Error ingesting C++ file {inputPath}: {e}")
            raise FileProcessingError(f"Error ingesting C++ file {inputPath}: {e}")


    async def update_classes(self, full_name, details):
        async with self.lock_classes:  # Acquire the lock
            self.classes[full_name].update(details)

    async def update_methods(self, full_name, details):
        async with self.lock_methods:  # Acquire the lock
            self.methods[full_name].update(details)

    async def update_functions(self, full_name, details):
        async with self.lock_functions:  # Acquire the lock
            self.functions[full_name].update(details)

    async def process_nodes(self, node, project_path, projectID, is_cpp_file):
        def is_exported(node):
            # Example check for __declspec(dllexport) or visibility attribute
            for token in node.get_tokens():
                if token.spelling in ["__declspec(dllexport)", "__attribute__((visibility(\"default\")))"]:
                    return True
            return False

        for child in node.get_children():
            file_name = child.location.file.name if child.location.file else ''
            if project_path in file_name:
                if child.kind in (clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL):
                    type_name = "Class" if child.kind == clang.cindex.CursorKind.CLASS_DECL else "Struct"
                    class_name = child.spelling
                    full_scope = self.get_full_scope(child)
                    fully_qualified_name = f"{full_scope}::{class_name}" if full_scope else class_name
                    interface_description = ""
                    implementation_description = ""

                    description = f"Class {class_name} in scope {full_scope} defined in {file_name}"
                    raw_comment = child.raw_comment if child.raw_comment else ''
                    if raw_comment:
                        description += f" with documentation: {raw_comment.strip()}"

                    # Base classes
                    bases = [base for base in child.get_children() if base.kind == clang.cindex.CursorKind.CXX_BASE_SPECIFIER]
                    if bases:
                        base_names = [f"{base.spelling} in scope {self.get_full_scope(base.type.get_declaration())}" for base in bases]
                        description += f". Inherits from: {', '.join(base_names)}"
                                        
                    is_node_exported = is_exported(child)
                    if is_node_exported:
                        description += ". (EXPORTED)"
                    
                    # Public members of the class
                    members = []

                    def get_public_members(node, inherited=False):
                        for member in node.get_children():
                            if member.access_specifier == clang.cindex.AccessSpecifier.PUBLIC:
                                origin = " (inherited)" if inherited else ""
                                export_status = "exported" if is_node_exported else "not exported"
                                if member.kind == clang.cindex.CursorKind.CXX_METHOD:
                                    members.append(f"public method {member.spelling} in scope {self.get_full_scope(member)} ({export_status}){origin}")
                                elif member.kind == clang.cindex.CursorKind.FIELD_DECL:
                                    members.append(f"public attribute {member.spelling} of type {member.type.spelling} in scope {self.get_full_scope(member)} ({export_status}){origin}")

                    # Get public members of the class itself
                    get_public_members(child, inherited=False)
                    
                    # Get public members of base classes
                    for base in bases:
                        if base.type and base.type.get_declaration().kind == clang.cindex.CursorKind.CLASS_DECL:
                            base_class = base.type.get_declaration()
                            get_public_members(base_class, inherited=True)
                    
                    if members:
                        interface_description = "Public interface: " + ", ".join(members)

                    members = []
                                        
                    def get_implementation_members(node, inherited=False):
                        for member in node.get_children():
                            origin = " (inherited)" if inherited else ""
                            if member.access_specifier == clang.cindex.AccessSpecifier.PRIVATE:
                                if member.kind == clang.cindex.CursorKind.CXX_METHOD:
                                    members.append(f"private method {member.spelling} in scope {self.get_full_scope(member)}{origin}")
                                elif member.kind == clang.cindex.CursorKind.FIELD_DECL:
                                    members.append(f"private attribute {member.spelling} of type {member.type.spelling} in scope {self.get_full_scope(member)}{origin}")
                                elif member.kind == clang.cindex.CursorKind.VAR_DECL:
                                    members.append(f"private static {member.spelling} of type {member.type.spelling} in scope {self.get_full_scope(member)}{origin}")
                            elif member.access_specifier == clang.cindex.AccessSpecifier.PROTECTED and inherited:
                                if member.kind == clang.cindex.CursorKind.CXX_METHOD:
                                    members.append(f"protected method {member.spelling} in scope {self.get_full_scope(member)}{origin}")
                                elif member.kind == clang.cindex.CursorKind.FIELD_DECL:
                                    members.append(f"protected attribute {member.spelling} of type {member.type.spelling} in scope {self.get_full_scope(member)}{origin}")

                    # Get implementation members of the class itself
                    get_implementation_members(child, inherited=False)
                    
                    # Get protected members of base classes
                    for base in bases:
                        if base.type and base.type.get_declaration().kind == clang.cindex.CursorKind.CLASS_DECL:
                            base_class = base.type.get_declaration()
                            get_implementation_members(base_class, inherited=True)
                    
                    if members:
                        implementation_description = "Implementation details: " + ", ".join(members)

                    async with self.lock_header_files:
                        header_code = self.header_files.get(file_name, '')
                        
                    # Cache class information
                    details = {
                        'type': type_name,
                        'name': fully_qualified_name,
                        'scope': full_scope,
                        'short_name': class_name,
                        'description' : description,
                        'interface_description' : interface_description,
                        'implementation_description' : implementation_description,
                        'raw_comment': raw_comment,
                        'file_path': file_name,
                        'raw_code': header_code
                    }
                    await self.update_classes(fully_qualified_name, details)
                    
                    await self.process_nodes(child, project_path, projectID, is_cpp_file)
                    
                elif child.kind == clang.cindex.CursorKind.CXX_METHOD:
                    type_name = "Method"
                    class_name = self.get_full_scope(child)
                    full_scope = class_name
                    method_name = child.spelling
                    fully_qualified_method_name = f"{class_name}::{method_name}" if class_name else method_name
                    
                    description = f"Method {method_name} in class {full_scope} defined in {file_name}"
                    raw_comment = child.raw_comment if child.raw_comment else ''
                    if raw_comment:
                        description += f" with documentation: {raw_comment.strip()}"

                    is_node_exported = is_exported(child)
                    if is_node_exported:
                        description += ". (EXPORTED)"
                        
                    # Parameters and return type
                    params = [f"{param.type.spelling} {param.spelling}" for param in child.get_arguments()]
                    return_type = child.result_type.spelling
                    description += f". Returns {return_type} and takes parameters: {', '.join(params)}"

                    description += "."

                    async with self.lock_methods:
                        is_new_method = True if fully_qualified_method_name not in self.methods else False
                        
                    # Cache method information, prioritize CPP file
                    if is_cpp_file or is_new_method:
                        raw_code = self.get_raw_code(child) if is_cpp_file else ''
                        # Cache method information
                        details = {
                            'type': type_name,
                            'name': fully_qualified_method_name,
                            'scope': full_scope,
                            'short_name': method_name,
                            'return_type': return_type,
                            'description' : description,
                            'raw_comment': raw_comment,
                            'file_path': file_name,
                            'is_cpp_file': is_cpp_file,
                            'raw_code': raw_code
                        }
                        await self.update_methods(fully_qualified_method_name, details)
                                                
                elif child.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                    type_name = "Function"
                    function_name = child.spelling
                    full_scope = self.get_full_scope(child)
                    fully_qualified_function_name = f"{full_scope}::{function_name}" if full_scope else function_name
                    
                    description = f"Function {function_name} in scope {full_scope} defined in {file_name}"
                    raw_comment = child.raw_comment if child.raw_comment else ''
                    if raw_comment:
                        description += f" with documentation: {raw_comment.strip()}"

                    is_node_exported = is_exported(child)
                    if is_node_exported:
                        description += ". (EXPORTED)"
                    
                    # Parameters and return type
                    params = [f"{param.type.spelling} {param.spelling}" for param in child.get_arguments()]
                    return_type = child.result_type.spelling
                    description += f". Returns {return_type} and takes parameters: {', '.join(params)}"
                    
                    description += "."

                    async with self.lock_functions:
                        is_new_function = True if fully_qualified_function_name not in self.functions else False
                    
                    # Store function information, prioritize CPP file
                    if is_cpp_file or is_new_function:
                        raw_code = self.get_raw_code(child) if is_cpp_file else ''
                        details = {
                            'type': type_name,
                            'name': fully_qualified_function_name,
                            'scope': full_scope,
                            'short_name': function_name,
                            'description' : description,
                            'return_type': return_type,
                            'raw_comment': raw_comment,
                            'file_path': file_name,
                            'is_cpp_file': is_cpp_file,
                            'raw_code': raw_code
                        }
                        # Cache function information
                        await self.update_functions(fully_qualified_function_name, details)
                else:
                    await self.process_nodes(child, project_path, projectID, is_cpp_file)

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

    async def summarize_all_cpp(self, projectID):
#        logger.info("Starting summarization of collected data.")
        
        try:
            tasks = []
            async with self.lock_classes:
                for class_name, class_info in self.classes.items():
                    name = class_info.get('name', 'anonymous')
                    class_info['name'] = name
                    info_copy = copy.deepcopy(class_info)
                    tasks.append(self.summarize_cpp_class_prep(name, info_copy))
#                    logger.info(f"Added task for summarizing cpp class {class_info['name']}")

            async with self.lock_methods:
                for method_name, method_info in self.methods.items():
                    name = method_info.get('name', 'anonymous')
                    method_info['name'] = name
                    info_copy = copy.deepcopy(method_info)
                    tasks.append(self.summarize_cpp_method_prep(name, info_copy))
#                    logger.info(f"Added task for summarizing cpp method {method_info['name']}")

            async with self.lock_functions:
                for function_name, function_info in self.functions.items():
                    name = function_info.get('name', 'anonymous')
                    function_info['name'] = name
                    info_copy = copy.deepcopy(function_info)
                    tasks.append(self.summarize_cpp_function_prep(name, info_copy))
#                    logger.info(f"Added task for summarizing cpp function {function_info['name']}")

        except Exception as e:
            logger.error(f"Error while appending tasks to queue: {e}")
            raise FileProcessingError(f"Error while appending tasks to queue: {e}")

#        logger.info("Gathering summarization and embedding tasks...")
        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error while gathering summarization and embedding tasks: {e}")
            raise FileProcessingError(f"Error while gathering summarization and embedding tasks: {e}")

#        logger.info("Finished summarization and embedding tasks. Starting storage phase...")
    

    async def summarize_cpp_class_prep(self, name, class_info):
    
#        logger.info(f"Summarizing class: {class_info['name']}")
        try:
            if name is None:
                logger.info(f"Skipping anonymous class")
                return class_info
                
            class_name, class_scope, interface_summary, implementation_summary = await self.summarize_cpp_class(class_info)
            
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_prep calling summarize_cpp_class: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_class_prep calling summarize_cpp_class: {e}")
#        logger.info(f"Finished summarizing class: {name}")

        try:
            interface_embedding = await self.make_embedding(interface_summary)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_prep calling make_embedding for interface_summary: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_class_prep calling make_embedding for interface_summary: {e}")
#        logger.info(f"Finished generating embedding for class interface: {name}")
        
        try:
            implementation_embedding = await self.make_embedding(implementation_summary)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_prep calling make_embedding for implementation_summary: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_class_prep calling make_embedding for implementation_summary: {e}")
#        logger.info(f"Finished generating embedding for class implementation: {name}")
        
        details = {
            'interface_summary': interface_summary,
            'implementation_summary': implementation_summary,
            'interface_embedding': interface_embedding,
            'implementation_embedding': implementation_embedding
        }
#        logger.info(f"Updating classes at the bottom of summarize_cpp_class_prep.")
        await self.update_classes(name, details)
        await self.update_progress_summarize(1)
        logger.info(f"Finished summarizing/embedding class: {name}")
        return class_info


    async def summarize_cpp_method_prep(self, name, method_info):
#        logger.info(f"Summarizing cpp method: {method_info['name']}")
        try:
            if name is None:
                logger.info(f"Skipping anonymous method")
                return method_info
            function_name, function_scope, function_summary = await self.summarize_cpp_function(method_info)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_method_prep: {e}\nmethod_info contents:\n{pprint.pformat(method_info, indent=4)}")
            raise FileProcessingError(f"Error in summarize_cpp_method_prep calling summarize_cpp_function for method {name}: {e}")
#        logger.info(f"Finished summarizing cpp method: {name}")
        
        try:
            embedding = await self.make_embedding(function_summary)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_method_prep calling make_embedding: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_method_prep calling make_embedding: {e}")
#        logger.info(f"Finished creating embedding for cpp method: {name}")

        details = {
            'summary': function_summary,
            'embedding': embedding
        }
#        logger.info(f"Updating methods at the bottom of summarize_cpp_method_prep.")
        await self.update_methods(name, details)
        await self.update_progress_summarize(1)
        logger.info(f"Finished summarizing/embedding cpp method: {name}")
        return method_info

    async def summarize_cpp_function_prep(self, name, function_info):
        try:
            if name is None:
                logger.info(f"Skipping anonymous function")
                return function_info
            

        except Exception as e:
            logger.error(f"Error in summarize_cpp_function_prep grabbing dict values: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_function_prep grabbing dict values: {e}")

#        logger.info(f"Summarizing cpp function: {name}")
        
        try:
            function_name, function_scope, function_summary = await self.summarize_cpp_function(function_info)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function_prep calling summarize_cpp_function: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_function_prep calling summarize_cpp_function: {e}")
#        logger.info(f"Finished summarizing cpp function: {name}")

        try:
            embedding = await self.make_embedding(function_summary)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function_prep calling make_embedding: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_function_prep calling make_embedding: {e}")
#        logger.info(f"Finished making embedding for cpp function: {name}")
        
        details = {
            'summary': function_summary,
            'embedding': embedding
        }
#        logger.info(f"Updating functions at the bottom of summarize_cpp_function_prep.")
        try:
            await self.update_functions(name, details)
            await self.update_progress_summarize(1)
            logger.info(f"Finished summarizing/embedding cpp function: {name}")
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function_prep calling update_functions: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_function_prep calling update_functions: {e}")

        return function_info




    async def store_all_cpp(self, projectID):
        try:
            queue = asyncio.Queue()

            # Create a task to process the queue
            process_task = asyncio.create_task(self.process_cpp_storage_queue(queue, projectID))

#            tasks = []

            async with self.lock_classes:
                for class_name, class_info in self.classes.items():
                    name = class_info.get('name', 'anonymous')
                    class_info['name'] = name
                    info_copy = copy.deepcopy(class_info)
                    await queue.put(('Class', name, info_copy))

            async with self.lock_methods:
                for method_name, method_info in self.methods.items():
                    name = method_info.get('name', 'anonymous')
                    method_info['name'] = name
                    info_copy = copy.deepcopy(method_info)
                    await queue.put(('Method', name, info_copy))

            async with self.lock_functions:
                for function_name, function_info in self.functions.items():
                    name = function_info.get('name', 'anonymous')
                    function_info['name'] = name
                    info_copy = copy.deepcopy(function_info)
                    await queue.put(('Function', name, info_copy))

        except Exception as e:
            logger.error(f"Error in store_all_cpp: {e}")
            raise FileProcessingError(f"Error in store_all_cpp: {e}")

        try:
#            logger.info("Gathering storage tasks...")
#            await asyncio.gather(*tasks)
#            logger.info("Finished gathering storage tasks.")

            # Signal that all items have been added to the queue
            await queue.put(None)

            # Wait for all items to be processed
            await process_task

        except Exception as e:
            logger.error(f"Error while gathering storage tasks: {e}")
            raise FileProcessingError(f"Error while gathering storage tasks: {e}")



    async def process_cpp_storage_queue(self, queue, projectID):
        while True:
            item = await queue.get()
            if item is None:
                break
            item_type, name, info = item
            if item_type == 'Class':
                await self.store_summary_cpp_class(name, info, projectID)
            elif item_type == 'Method':
                await self.store_summary_cpp_method(name, info, projectID)
            elif item_type == 'Function':
                await self.store_summary_cpp_function(name, info, projectID)
            await self.update_progress_store(1)
            queue.task_done()
        
    """
    ## TODO RESUME
    2024-07-13 07:48:59,056 - ERROR - Error forming query for CPP class Blindsecp: 'name'                                                     | 0/118 [00:00<?, ?data/s]
    {   'description': 'Class Blindsecp in scope  defined in '
                       '/Users/au/src/blindsecp/Blindsecp.hpp',
        'file_path': '/Users/au/src/blindsecp/Blindsecp.hpp',
        'implementation_description': '',
        'implementation_embedding': [   0.0942307785153389,
        'implementation_summary': 'This class is for people who would like to '
        'interface_description': 'Public interface: public method createInstance '
        'interface_embedding': [   -0.20329229533672333,
        'interface_summary': 'This class describes how to create a public '
        'name': 'Blindsecp',
        'raw_code': '#pragma once\n'
        'raw_comment': '',
        'scope': '',
        'short_name': 'Blindsecp',
        'type': 'Class'}

    """

    async def store_summary_cpp_class(self, name, info, projectID):
        try:
#            logger.error(f"In store_summary_cpp_class.\ninfo contents:\n{pprint.pformat(info, indent=4)}")

            type_name = info.get('type', 'Class')
            if not type_name or not type_name.strip():
                type_name = 'Class'

            scope = info.get('scope', '')
            short_name = info.get('short_name', '')
            interface_summary = info.get('interface_summary', '')
            implementation_summary = info.get('implementation_summary', '')
            interface_embedding = info.get('interface_embedding', [])
            implementation_embedding = info.get('implementation_embedding', [])
            file_path = info.get('file_path', '')
            raw_code = info.get('raw_code', '')

#            logger.info(f"store_summary_cpp_class name: {name}")
#            logger.info(f"store_summary_cpp_class projectID: {projectID}")
#            logger.info(f"store_summary_cpp_class scope: {scope}")
#            logger.info(f"store_summary_cpp_class short_name: {short_name}")
#            logger.info(f"store_summary_cpp_class interface_summary:\n{interface_summary}")
#            logger.info(f"store_summary_cpp_class implementation_summary:\n{implementation_summary}")
#            logger.info(f"store_summary_cpp_class file_path: {file_path}")
#            logger.info(f"store_summary_cpp_class raw_code: {raw_code}")

            query = """
                MERGE (n:{type} {{name: $name, projectID: $projectID}})
                ON CREATE SET n.scope = $scope,
                              n.short_name = $short_name,
                              n.interface_summary = $interface_summary,
                              n.implementation_summary = $implementation_summary,
                              n.interface_embedding = $interface_embedding,
                              n.implementation_embedding = $implementation_embedding,
                              n.file_path = $file_path,
                              n.raw_code = $raw_code
                ON MATCH SET
                    n.interface_summary = CASE WHEN size($interface_summary) > size(n.interface_summary) THEN $interface_summary ELSE n.interface_summary END,
                    n.interface_embedding = CASE WHEN size($interface_summary) > size(n.interface_summary) THEN $interface_embedding ELSE n.interface_embedding END,
                    n.implementation_summary = CASE WHEN size($implementation_summary) > size(n.implementation_summary) THEN $implementation_summary ELSE n.implementation_summary END,
                    n.implementation_embedding = CASE WHEN size($implementation_summary) > size(n.implementation_embedding) THEN $implementation_embedding ELSE n.implementation_embedding END,
                    n.raw_code = CASE WHEN size($raw_code) > size(n.raw_code) THEN $raw_code ELSE n.raw_code END
                RETURN elementId(n)
            """.format(type=type_name)
            params = {
                'name': name if name else 'anonymous',
                'projectID': projectID if projectID else 'Unknown',
                'scope': scope,
                'short_name': short_name,
                'interface_summary': interface_summary,
                'implementation_summary': implementation_summary,
                'interface_embedding': interface_embedding,
                'implementation_embedding': implementation_embedding,
                'file_path': file_path,
                'raw_code': raw_code
            }
#            logger.debug(f"Storing cpp class with query:\n\n{query}\n\n")
        except Exception as e:
            logger.error(f"Error forming query for CPP class {name}: {e}")
            raise FileProcessingError(f"Error forming query for CPP class {name}: {e}")
#            logger.error(f"Error forming query for CPP class {name}: {e}\n{pprint.pformat(info, indent=4)}")
#            raise FileProcessingError(f"Error forming query for CPP class {name}: {e}\n{pprint.pformat(info, indent=4)}")

        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
#                    logger.info(f"Storing CPP class {name}...")
                    await self.run_query_and_get_element_id(session, query, **params)
                    logger.info(f"Finished storing CPP class: {name}")
        except Exception as e:
            logger.error(f"Error storing summary for CPP class {name}: {e}")
            raise FileProcessingError(f"Error storing summary for CPP class {name}: {e}")



    async def store_summary_cpp_method(self, name, info, projectID):
        try:
#            logger.error(f"In store_summary_cpp_method.\ninfo contents:\n{pprint.pformat(info, indent=4)}")
            
            type_name = info.get('type', 'Method')
            if not type_name or not type_name.strip():
                type_name = 'Method'

            scope = info.get('scope', '')
            short_name = info.get('short_name', '')
            summary = info.get('summary', '')
            embedding = info.get('embedding', [])
            file_path = info.get('file_path', '')
            raw_code = info.get('raw_code', '')

            query = """
                MERGE (n:{type} {{name: $name, projectID: $projectID}})
                ON CREATE SET n.scope = $scope,
                              n.short_name = $short_name,
                              n.summary = $summary,
                              n.embedding = $embedding,
                              n.file_path = $file_path,
                              n.raw_code = $raw_code
                ON MATCH SET
                    n.summary = CASE WHEN size($raw_code) > size(n.raw_code) THEN $summary ELSE n.summary END,
                    n.embedding = CASE WHEN size($raw_code) > size(n.raw_code) THEN $embedding ELSE n.embedding END
                RETURN elementId(n)
            """.format(type=type_name)
            params = {
                'name': name if name is not None else 'Unknown',
                'projectID': projectID if projectID is not None else 'Unknown',
                'scope': scope,
                'short_name': short_name,
                'summary': summary,
                'embedding': embedding,
                'file_path': file_path,
                'raw_code': raw_code
            }
        except Exception as e:
            logger.error(f"Error forming query for CPP method {name}: {e}\n{pprint.pformat(info, indent=4)}")
            raise FileProcessingError(f"Error forming query for CPP method {name}: {e}\n{pprint.pformat(info, indent=4)}")

        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    element_id = await self.run_query_and_get_element_id(session, query, **params)

                    # Here we create the relationship (edge) between the class and its method.
                    if element_id:
                        await (await session.run(
                            "MATCH (c) "
                            "WHERE (c.name = $scope AND c:Class AND c.projectID = $projectID) "
                            "OR (c.name = $scope AND c:Struct AND c.projectID = $projectID) "
                            "MATCH (f:Method) WHERE elementId(f) = $element_id AND f.projectID = $projectID "
                            "MERGE (c)-[:HAS_METHOD]->(f) "
                            "MERGE (f)-[:BELONGS_TO]->(c)",
                            {"name": name, "scope": scope, "element_id": element_id, "projectID": projectID}
                        )).consume()
                       
                        logger.info(f"Finished storing CPP method: {name}")

#                    logger.info(f"Finished creating CPP method: '{name}' Relationships to class {scope}")

        except Exception as e:
            logger.error(f"Error storing summary for CPP method {name}: {e}")
            raise FileProcessingError(f"Error storing summary for CPP method {name}: {e}")



    async def store_summary_cpp_function(self, name, info, projectID):
        try:
#            logger.error(f"In store_summary_cpp_function.\ninfo contents:\n{pprint.pformat(info, indent=4)}")

            type_name = info.get('type', 'Function')
            if not type_name or not type_name.strip():
                type_name = 'Function'

            scope = info.get('scope', '')
            short_name = info.get('short_name', '')
            summary = info.get('summary', '')
            embedding = info.get('embedding', [])
            file_path = info.get('file_path', '')
            raw_code = info.get('raw_code', '')

            query = """
                MERGE (n:{type} {{name: $name, projectID: $projectID}})
                ON CREATE SET n.scope = $scope,
                              n.short_name = $short_name,
                              n.summary = $summary,
                              n.embedding = $embedding,
                              n.file_path = $file_path,
                              n.raw_code = $raw_code
                ON MATCH SET
                    n.summary = CASE WHEN size($raw_code) > size(n.raw_code) THEN $summary ELSE n.summary END,
                    n.embedding = CASE WHEN size($raw_code) > size(n.raw_code) THEN $embedding ELSE n.embedding END
                RETURN elementId(n)
            """.format(type=type_name)
            params = {
                'name': name if name is not None else 'anonymous',
                'projectID': projectID if projectID is not None else 'Unknown',
                'scope': scope,
                'short_name': short_name,
                'summary': summary,
                'embedding': embedding,
                'file_path': file_path,
                'raw_code': raw_code
            }
        except Exception as e:
            logger.error(f"Error forming query for CPP function {name}: {e}\n{pprint.pformat(info, indent=4)}")
            raise FileProcessingError(f"Error forming query for CPP function {name}: {e}\n{pprint.pformat(info, indent=4)}")

        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    await self.run_query_and_get_element_id(session, query, **params)
                    logger.info(f"Finished storing CPP function: {name}")

        except Exception as e:
            logger.error(f"Error storing summary for CPP function {name}: {e}")
            raise FileProcessingError(f"Error storing summary for CPP function {name}: {e}")


    def get_full_scope(self, node):
        scopes = []
        current = node.semantic_parent
        while current and current.kind != clang.cindex.CursorKind.TRANSLATION_UNIT:
            scopes.append(current.spelling)
            current = current.semantic_parent
        return "::".join(reversed(scopes))

    
    async def IngestPython(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        """
        Ingest a Python file by parsing it, summarizing classes and functions, and storing them in the database.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            projectID (str): The project ID.
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
                            "CREATE (n:Class {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN elementId(n)",
                            name=cls.name, summary=class_summary, embedding=embedding, projectID=projectID
                        )

                for func in cls.body:
                    if isinstance(func, ast.FunctionDef):
                        function_summary = await self.summarize_python_function(func)
                        embedding = await self.make_embedding(function_summary)
                        
                        async with self.rate_limiter_db:
                            async with self.get_session() as session:
                                function_id = await self.run_query_and_get_element_id(session,
                                    "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN elementId(n)",
                                    name=func.name, summary=function_summary, embedding=embedding, projectID=projectID
                                )
                                await (await session.run(
                                    "MATCH (c:Class {elementId: $class_id, projectID: $projectID}) "
                                    "MATCH (f:Function {elementId: $function_id, projectID: $projectID}) "
                                    "CREATE (c)-[:HAS_METHOD]->(f)",
                                    class_id=class_id, function_id=function_id, projectID=projectID
                                )).consume()

            for func in functions:
                function_summary = await self.summarize_python_function(func)
                embedding = await self.make_embedding(function_summary)
                
                async with self.rate_limiter_db:
                    async with self.get_session() as session:
                        await (await session.run(
                            "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID})",
                            name=func.name, summary=function_summary, embedding=embedding, projectID=projectID
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

    async def IngestRust(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        """
        Ingest a Rust file by parsing it, summarizing implementations and functions, and storing them in the database.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            projectID (str): The project ID.
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
                            "CREATE (n:Impl {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN elementId(n)",
                            name=impl.trait_.path.segments[0].ident, summary=impl_summary, embedding=embedding, projectID=projectID
                        )

                for item in impl.items:
                    if isinstance(item, syn.ImplItemMethod):
                        function_summary = await self.summarize_rust_function(item)
                        embedding = await self.make_embedding(function_summary)
                        
                        async with self.rate_limiter_db:
                            async with self.get_session() as session:
                                function_id = await self.run_query_and_get_element_id(session,
                                    "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN elementId(n)",
                                    name=item.sig.ident, summary=function_summary, embedding=embedding, projectID=projectID
                                )
                                await (await session.run(
                                    "MATCH (i:Impl {elementId: $impl_id, projectID: $projectID}), "
                                    "MATCH (f:Function {elementId: $function_id, projectID: $projectID}) "
                                    "CREATE (i)-[:HAS_METHOD]->(f)",
                                    impl_id=impl_id, function_id=function_id, projectID=projectID
                                )).consume()

            for func in functions:
                function_summary = await self.summarize_rust_function(func)
                embedding = await self.make_embedding(function_summary)
                
                async with self.rate_limiter_db:
                    async with self.get_session() as session:
                        await (await session.run(
                            "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID})",
                            name=func.sig.ident, summary=function_summary, embedding=embedding, projectID=projectID
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

    async def IngestJavascript(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        """
        Ingest a JavaScript file by parsing it, summarizing classes, functions, and variables, and storing them in the database.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            projectID (str): The project ID.
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
                            "CREATE (n:JavaScriptFile {name: $name, path: $path, projectID: $projectID}) RETURN elementId(n)",
                            name=inputName, path=inputPath, projectID=projectID
                        )

                for node in ast.body:
                    if isinstance(node, nodes.FunctionDeclaration):
                        await self.process_js_function(node, file_id, projectID)
                    elif isinstance(node, nodes.ClassDeclaration):
                        await self.process_js_class(node, file_id, projectID)
                    elif isinstance(node, nodes.VariableDeclaration):
                        await self.process_js_variable(node, file_id, projectID)
            except Exception as e:
                logger.error(f"Error ingesting JavaScript file {inputPath}: {e}")
                raise DatabaseError(f"Error ingesting JavaScript file {inputPath}: {e}")

    async def process_js_function(self, func_node, file_id: int, projectID: str) -> None:
        func_name = func_node.id.name if func_node.id else 'anonymous'
        func_summary = await self.summarize_js_function(func_node)
        
        embedding = await self.make_embedding(func_summary)

        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    func_db_id = await self.run_query_and_get_element_id(session,
                        "CREATE (n:JavaScriptFunction {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN elementId(n)",
                        name=func_name, summary=func_summary, embedding=embedding, projectID=projectID
                    )

                    await (await session.run(
                        "MATCH (f:JavaScriptFile {elementId: $file_id, projectID: $projectID}), (func:JavaScriptFunction {elementId: $func_id, projectID: $projectID}) "
                        "CREATE (f)-[:CONTAINS]->(func)",
                        file_id=file_id, func_id=func_db_id, projectID=projectID
                    )).consume()
        except Exception as e:
            logger.error(f"Error processing JavaScript function {func_name}: {e}")
            raise DatabaseError(f"Error processing JavaScript function {func_name}: {e}")

    async def process_js_class(self, class_node, file_id: int, projectID: str) -> None:
        class_name = class_node.id.name
        class_summary = await self.summarize_js_class(class_node)
        
        embedding = await self.make_embedding(class_summary)

        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    class_db_id = await self.run_query_and_get_element_id(session,
                        "CREATE (n:JavaScriptClass {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN elementId(n)",
                        name=class_name, summary=class_summary, embedding=embedding, projectID=projectID
                    )

                    await (await session.run(
                        "MATCH (f:JavaScriptFile {elementId: $file_id, projectID: $projectID}), (c:JavaScriptClass {elementId: $class_id, projectID: $projectID}) "
                        "CREATE (f)-[:CONTAINS]->(c)",
                        file_id=file_id, class_id=class_db_id, projectID=projectID
                    )).consume()

            for method in class_node.body.body:
                if isinstance(method, nodes.MethodDefinition):
                    await self.process_js_method(method, class_db_id, projectID)
        except Exception as e:
            logger.error(f"Error processing JavaScript class {class_name}: {e}")
            raise DatabaseError(f"Error processing JavaScript class {class_name}: {e}")

    async def process_js_variable(self, var_node, file_id: int, projectID: str) -> None:
        for declaration in var_node.declarations:
            var_name = declaration.id.name
            var_type = var_node.kind  # 'var', 'let', or 'const'
            var_summary = await self.summarize_js_variable(var_node)
            
            embedding = await self.make_embedding(var_summary)

            try:
                async with self.rate_limiter_db:
                    async with self.get_session() as session:
                        var_db_id = await self.run_query_and_get_element_id(session,
                            "CREATE (n:JavaScriptVariable {name: $name, type: $type, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN elementId(n)",
                            name=var_name, type=var_type, summary=var_summary, embedding=embedding, projectID=projectID
                        )

                        await (await session.run(
                            "MATCH (f:JavaScriptFile {elementId: $file_id, projectID: $projectID}), (v:JavaScriptVariable {elementId: $var_id, projectID: $projectID}) "
                            "CREATE (f)-[:CONTAINS]->(v)",
                            file_id=file_id, var_id=var_db_id, projectID=projectID
                        )).consume()
            except Exception as e:
                logger.error(f"Error processing JavaScript variable {var_name}: {e}")
                raise DatabaseError(f"Error processing JavaScript variable {var_name}: {e}")

    async def process_js_method(self, method_node, class_id: int, projectID: str) -> None:
        """
        Process a JavaScript method node and create nodes in the database.

        Args:
            method_node: The method node.
            class_id: The class ID.
            projectID: The project ID.
        """
        method_name = method_node.key.name
        method_summary = await self.summarize_js_function(method_node.value)
        
        embedding = await self.make_embedding(method_summary)

        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    method_db_id = await self.run_query_and_get_element_id(session,
                        "CREATE (n:JavaScriptMethod {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN elementId(n)",
                        name=method_name, summary=method_summary, embedding=embedding, projectID=projectID
                    )

                    await (await session.run(
                        "MATCH (c:JavaScriptClass {elementId: $class_id, projectID: $projectID}), (m:JavaScriptMethod {elementId: $method_id, projectID: $projectID}) "
                        "CREATE (c)-[:HAS_METHOD]->(m)",
                        class_id=class_id, method_id=method_db_id, projectID=projectID
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



## RESUME
    async def IngestPdf(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        try:
            reader = PdfReader(inputPath)
            num_pages = len(reader.pages)

            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    pdf_id = await self.run_query_and_get_element_id(session,
                        "CREATE (n:PDF {name: $name, pages: $pages, projectID: $projectID}) RETURN elementId(n)",
                        name=inputName, pages=num_pages, projectID=projectID
                    )

            for i in range(num_pages):
                page = reader.pages[i]
                text = page.extract_text()

                medium_chunks = self.chunk_text(text, MEDIUM_CHUNK_SIZE)
                for j, medium_chunk in enumerate(medium_chunks):
                    async with self.rate_limiter_db:
                        async with self.get_session() as session:
                            medium_chunk_id = await self.run_query_and_get_element_id(session,
                                "CREATE (n:MediumChunk {content: $content, type: 'medium_chunk', page: $page, projectID: $projectID}) RETURN elementId(n)",
                                content=medium_chunk, page=i + 1, projectID=projectID
                            )

                            if medium_chunk_id:
                                await (await session.run(
                                    "MATCH (p:PDF) WHERE (elementId(p) = $pdf_id AND p.projectID = $projectID) "
                                    "MATCH (m:MediumChunk) WHERE (elementId(m) = $medium_chunk_id AND m.projectID = $projectID) "
                                    "MERGE (p)-[:HAS_CHUNK]->(m) "
                                    "MERGE (m)-[:BELONGS_TO]->(p)",
                                    {"pdf_id": pdf_id, "medium_chunk_id": medium_chunk_id, "projectID": projectID}
                                )).consume()

                    small_chunks = self.chunk_text(medium_chunk, SMALL_CHUNK_SIZE)
                    for k, small_chunk in enumerate(small_chunks):
                        async with self.rate_limiter_db:
                            async with self.get_session() as session:
                                small_chunk_id = await self.run_query_and_get_element_id(session,
                                    "CREATE (n:SmallChunk {content: $content, type: 'small_chunk', page: $page, projectID: $projectID}) RETURN elementId(n)",
                                    content=small_chunk, page=i + 1, projectID=projectID
                                )

                                if small_chunk_id:
                                    await (await session.run(
                                        "MATCH (m:MediumChunk) WHERE (elementId(m) = $medium_chunk_id AND m.projectID = $projectID) "
                                        "MATCH (s:SmallChunk) WHERE (elementId(s) = $small_chunk_id AND s.projectID = $projectID) "
                                        "MERGE (m)-[:HAS_CHUNK]->(s) "
                                        "MERGE (s)-[:BELONGS_TO]->(m)",
                                        {"small_chunk_id": small_chunk_id, "medium_chunk_id": medium_chunk_id, "projectID": projectID}
                                    )).consume()

                        embedding = await self.make_embedding(small_chunk)
                            
                        async with self.rate_limiter_db:
                            async with self.get_session() as session:
                                embedding_id = await self.run_query_and_get_element_id(session,
                                    "CREATE (n:Embedding {embedding: $embedding, type: 'embedding', projectID: $projectID}) RETURN elementId(n)",
                                    embedding=embedding, projectID=projectID
                                )

                                if embedding_id:
                                    await (await session.run(
                                        "MATCH (p:PDF) WHERE (elementId(p) = $pdf_id AND p.projectID = $projectID) "
                                        "MATCH (m:MediumChunk) WHERE (elementId(m) = $medium_chunk_id AND m.projectID = $projectID) "
                                        "MATCH (s:SmallChunk) WHERE (elementId(s) = $small_chunk_id AND s.projectID = $projectID) "
                                        "MATCH (e:Embedding) WHERE (elementId(e) = $embedding_id AND e.projectID = $projectID) "
                                        "MERGE (s)-[:HAS_EMBEDDING]->(e) "
                                        "MERGE (e)-[:IS_EMBEDDING_OF]->(s) "
                                        "MERGE (e)-[:HAS_PARENT_CHUNK]->(m) "
                                        "MERGE (e)-[:HAS_PARENT_DOC]->(p) "
                                        "",
                                        {"pdf_id": pdf_id, "medium_chunk_id": medium_chunk_id, "small_chunk_id": small_chunk_id, "embedding_id": embedding_id, "projectID": projectID}
                                    )).consume()

        except Exception as e:
            logger.error(f"Error ingesting PDF file {inputPath}: {e}")
            raise FileProcessingError(f"Error ingesting PDF file {inputPath}: {e}")

    async def getFileAndCppCount(self) -> tuple[int, int]:
        try:
            async with self.lock_classes:
                len_classes = len(self.classes)
            async with self.lock_methods:
                len_methods = len(self.methods)
            async with self.lock_functions:
                len_functions = len(self.functions)
            # We increment the methods twice each, once for the method itself and once for it as a function.
            total_cpp_count = len_classes + len_methods + len_functions
            
            # Set the progress bar here for summarization and storing.
            # We set the total value to all files + all cpp classes + all cpp methods * 2 + all cpp functions
            # Therefore we must increment each one after every file PLUS every Cpp class, every method, and every function.
            file_count = await self.get_file_paths_length()
            
            return file_count, total_cpp_count
            
        except Exception as e:
            logger.error(f"Error getting lengths of local members: {e}")
            raise FileProcessingError(f"Error getting lengths of local members: {e}")
    # -------------------------------------------------------------------


class NIngest:
    """
    Manages the ingestion process, counting files, handling directories, and updating or deleting projects.
    """
    def __init__(self, projectID: str, importer: NBaseImporter = NNeo4JImporter()):
        self.projectID = projectID
        self.currentOutputPath = os.path.expanduser(f"~/.ngest/projects/{self.projectID}")
        
        self.progress_bar_scan = None
        self.progress_bar_summarize = None
        self.progress_bar_store = None
        
        self.progress_queue_scan = asyncio.Queue()
        self.progress_queue_summarize = asyncio.Queue()
        self.progress_queue_store = asyncio.Queue()
        
        self.progress_updater_task_scan = None
        self.progress_updater_task_summarize = None
        self.progress_updater_task_store = None
        
        self.started_ingestion = False
        self.summarized_cpp_started = False
        self.summarized_cpp_finished = False
        self.stored_cpp_started = False
        self.stored_cpp_finished = False
        self.start_ingest_semaphore = asyncio.Semaphore(1)
        self.summarize_semaphore = asyncio.Semaphore(1)
        self.store_semaphore = asyncio.Semaphore(1)

        self.progress_lock_scan = asyncio.Lock()
        self.progress_lock_summarize = asyncio.Lock()
        self.progress_lock_store = asyncio.Lock()

        self.importer_ = importer
        self.importer_.set_progress_callback_scan(self.update_progress_scan)
        self.importer_.set_progress_callback_summarize(self.update_progress_summarize)
        self.importer_.set_progress_callback_store(self.update_progress_store)

        self.total_files = 0

        if not os.path.exists(self.currentOutputPath):
            os.makedirs(self.currentOutputPath, exist_ok=True)
            open(os.path.join(self.currentOutputPath, '.ngest_index'), 'a').close()
            logger.info(f"Created new project directory at {self.currentOutputPath}")

    def validate_input_path(self, input_path: str) -> bool:
        if not os.path.exists(input_path):
            logger.error(f"Input path does not exist: {input_path}")
            return False
        if os.path.isfile(input_path) and os.path.getsize(input_path) > MAX_FILE_SIZE:
            logger.error(f"File {input_path} exceeds the maximum allowed size of {MAX_FILE_SIZE} bytes.")
            return False
        return True

    async def update_progress_scan(self, increment=1):
        await self.progress_queue_scan.put(increment)

    async def progress_updater_scan(self):
        try:
            while True:
                async with self.progress_lock_scan:
                    increment = await self.progress_queue_scan.get()
                    self.progress_bar_scan.update(increment)
                    self.progress_queue_scan.task_done()
        except Exception as e:
            logger.error(f"Error in progress_updater_scan: {e}")

    async def update_progress_summarize(self, increment=1):
        await self.progress_queue_summarize.put(increment)

    async def progress_updater_summarize(self):
        try:
            while True:
                async with self.progress_lock_summarize:
                    increment = await self.progress_queue_summarize.get()
                    self.progress_bar_summarize.update(increment)
                    self.progress_queue_summarize.task_done()
        except Exception as e:
            logger.error(f"Error in progress_updater_summarize: {e}")

    async def update_progress_store(self, increment=1):
        await self.progress_queue_store.put(increment)

    async def progress_updater_store(self):
        try:
            while True:
                async with self.progress_lock_store:
                    increment = await self.progress_queue_store.get()
                    self.progress_bar_store.update(increment)
                    self.progress_queue_store.task_done()
        except Exception as e:
            logger.error(f"Error in progress_updater_store: {e}")

    async def update_total_progress_scan(self, new_total):
        try:
            async with self.progress_lock_scan:
                if self.progress_bar_scan is not None:
                    self.progress_bar_scan.total = new_total
                    self.progress_bar_scan.refresh()
        except Exception as e:
            logger.error(f"Error in update_total_progress_scan: {e}")

    async def update_total_progress_summarize(self, new_total):
        try:
            async with self.progress_lock_summarize:
                if self.progress_bar_summarize is not None:
                    self.progress_bar_summarize.total = new_total
                    self.progress_bar_summarize.refresh()
        except Exception as e:
            logger.error(f"Error in update_total_progress_summarize: {e}")

    async def update_total_progress_store(self, new_total):
        try:
            async with self.progress_lock_store:
                if self.progress_bar_store is not None:
                    self.progress_bar_store.total = new_total
                    self.progress_bar_store.refresh()
        except Exception as e:
            logger.error(f"Error in update_total_progress_store: {e}")

    async def start_progress_scan(self, total):
        try:
            async with self.progress_lock_scan:
                if self.progress_bar_scan is None:
                    self.progress_bar_scan = tqdm(total=total, desc="Scanning / Ingesting", unit="files")
                    self.progress_updater_task_scan = asyncio.create_task(self.progress_updater_scan())
        except Exception as e:
            logger.error(f"Error in start_progress_scan: {e}")

    async def start_progress_summarize(self, total):
        try:
            async with self.progress_lock_summarize:
                if self.progress_bar_summarize is None:
                    self.progress_bar_summarize = tqdm(total=total, desc="Summarizing / Embedding", unit="data")
                    self.progress_updater_task_summarize = asyncio.create_task(self.progress_updater_summarize())
        except Exception as e:
            logger.error(f"Error in start_progress_summarize: {e}")

    async def start_progress_store(self, total):
        try:
            async with self.progress_lock_store:
                if self.progress_bar_store is None:
                    self.progress_bar_store = tqdm(total=total, desc="Storing", unit="data")
                    self.progress_updater_task_store = asyncio.create_task(self.progress_updater_store())
        except Exception as e:
            logger.error(f"Error in start_progress_store: {e}")



    async def start_ingestion(self, inputPath: str) -> int:
        async with self.start_ingest_semaphore:
            if self.started_ingestion == True:
                return 0
            else:
                started_ingestion = True
        
            if not self.validate_input_path(inputPath):
                return -1
            
            async def split_path(topLevelInputPath, topLevelOutputPath):
                project_input_location, project_root = os.path.split(topLevelInputPath)
                project_input_location = project_input_location.rstrip('/')
                project_output_location = topLevelOutputPath
                return project_input_location, project_root, project_output_location

            project_input_location, project_root, topLevelOutputPath = await split_path(inputPath, self.currentOutputPath)
            logger.info(f"project_input_location: {project_input_location},  project_root: {project_root}, topLevelOutputPath: {topLevelOutputPath}")

            logger.info(f"Starting ingestion from inputPath: {inputPath}")
            # Starting ingestion from inputPath: /Users/au/src/blindsecp

    
            try:
                self.gitignore_patterns = self.load_gitignore_patterns(inputPath)

                if not os.path.exists(self.currentOutputPath):
                    os.makedirs(self.currentOutputPath, exist_ok=True)
                    open(os.path.join(self.currentOutputPath, '.ngest_index'), 'a').close()
                    logger.info(f"Created new project ingestion directory at {self.currentOutputPath}")
                    # Created new project ingestion directory at /Users/au/.ngest/projects/03e419c4-5b33-402f-8087-fcf084a08af5

                self.total_files = await self.count_files(inputPath)
                result = 0

                await self.start_progress_scan(self.total_files)
                await self.start_progress_summarize(self.total_files)
                await self.start_progress_store(self.total_files)

                result = await self.Ingest(inputPath, project_input_location, topLevelOutputPath, self.currentOutputPath)
                
            except Exception as e:
                result = -1
                logger.error(f"Error during ingestion in start_ingestion: {e}")
            finally:
                try:
                    # Cancel all progress updater tasks
                    if self.progress_updater_task_scan:
                        self.progress_updater_task_scan.cancel()
                    if self.progress_updater_task_summarize:
                        self.progress_updater_task_summarize.cancel()
                    if self.progress_updater_task_store:
                        self.progress_updater_task_store.cancel()

                    async with self.progress_lock_scan:
                        if self.progress_bar_scan:
                            self.progress_bar_scan.close()
                            
                    async with self.progress_lock_summarize:
                        if self.progress_bar_summarize:
                            self.progress_bar_summarize.close()
                            
                    async with self.progress_lock_store:
                        if self.progress_bar_store:
                            self.progress_bar_store.close()

                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")
                return result


    async def count_files(self, path: str) -> int:
        """
        Count the total number of files in a directory.

        Args:
            path (str): The path to the directory.

        Returns:
            int: The total number of files.
        """
        total = 0
        if self.should_ignore_file(path):
            return total
            
        for entry in os.scandir(path):
            if self.should_ignore_file(entry.path):
                continue
            if entry.is_file():
                total += 1
            elif entry.is_dir():
                total += await self.count_files(entry.path)
        return total
    
    async def cleanup_partial_ingestion(self, projectID: str):
        """
        Clean up partial ingestion by removing all nodes and relationships related to the project.

        Args:
            projectID (str): The project ID.
        """
        async with self.importer_.get_session() as session:
            try:
                await (await session.run(
                    "MATCH (n) WHERE n.projectID = $projectID "
                    "DETACH DELETE n",
                    projectID=self.projectID
                )).consume()
                logger.info(f"Cleaned up partial ingestion for project {self.projectID}")
            except Exception as e:
                logger.error(f"Error during cleanup for project {self.projectID}: {e}")


## RESUME

    # This is where the root input directory is passed. Everyhing scanned / ingested happens from here on in.
    # Recursively starting here.
    async def Ingest(self, inputPath: str, topLevelInputPath: str, topLevelOutputPath: str, currentOutputPath: str) -> int:
        """
        Ingest files from the input path into the output path.

        Args:
            inputPath (str): The path to the input file or directory.
            currentOutputPath (str): The current output path.

        Returns:
            int: The result of the ingestion process.
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

#        async with self.ingest_semaphore:
        try:
            # Phase 1: Scanning / Ingestion
            # Currently all non-CPP files are also summarized / stored here.
            if not os.path.exists(inputPath):
                logger.error(f"Invalid path: {inputPath}")
                return -1

            inputType = 'd' if os.path.isdir(inputPath) else 'f'
            index_file_path = os.path.join(topLevelOutputPath, '.ngest_index')

            localPath = strip_top_level(inputPath, topLevelInputPath)

            with open(index_file_path, 'a') as index_file:
                index_file.write(f"{inputType},{localPath}\n")
#            logger.info(f"Scanning: {inputPath}")

            inputLocation, inputName = os.path.split(inputPath)
            tasks = []


## RESUME need to pass localPath to IngestDirectory

            if inputType == 'd':
                tasks.append(self.IngestDirectory(inputPath=inputPath, inputLocation=inputLocation, inputName=inputName, topLevelInputPath=topLevelInputPath, topLevelOutputPath=topLevelOutputPath, currentOutputPath=currentOutputPath, projectID=self.projectID))
            else:
                if not self.should_ignore_file(inputPath):
                    tasks.append(self.IngestFile(inputPath=inputPath, inputLocation=inputLocation, inputName=inputName, topLevelInputPath=topLevelInputPath, topLevelOutputPath=topLevelOutputPath, currentOutputPath=currentOutputPath, projectID=self.projectID))

#            logger.info("Gathering scanning tasks...")
            # Wait for all ingestion tasks to complete
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error during ingestion in Ingest: {e}")
            await self.cleanup_partial_ingestion(self.projectID)
            return -1
        # -------------------------------------------------------------------
        # Phase 2: Summarizing and Embedding CPP files
        if self.summarized_cpp_started == False:
            self.summarized_cpp_started = True
#                logger.info("Finished gathering scanning tasks.")
            # -------------------------------------------------------------------
            async with self.summarize_semaphore:
#            try:
#                file_count, total_cpp_count = await self.importer_.getFileAndCppCount()
#                total_progress = file_count + total_cpp_count
#            except Exception as e:
#                logger.error(f"Error getting file and cpp count: {e}")
#                raise FileProcessingError(f"Error getting file and cpp count: {e}")
#
#            try:
#                await self.update_total_progress_scan(file_count)
#                await self.update_total_progress_summarize(total_progress)
#                await self.update_total_progress_store(total_progress)
#            except Exception as e:
#                logger.error(f"Error updating summarization and storing progress bars: {e}")
#                raise FileProcessingError(f"Error updating summarization and storing progress bars: {e}")
            # -------------------------------------------------------------------
#                logger.info("Starting summarization of CPP files...")
                try:
                    await self.importer_.summarize_all_cpp(self.projectID)
                    self.summarized_cpp_finished = True
                except Exception as e:
                    logger.error(f"Error during CPP summarization in Ingest: {e}")
                    await self.cleanup_partial_ingestion(self.projectID)
                    return -1
    #                logger.info("CPP summarization complete. Next, storing to database...")
        # -------------------------------------------------------------------
        # Phase 3: Batching into the Database
        if self.stored_cpp_finished is not True:
            if self.summarized_cpp_finished == True and self.stored_cpp_started == False:
                self.stored_cpp_started = True
#                logger.info("Starting DB importation of CPP files.")
                async with self.store_semaphore:

#                try:
#                    file_count, total_cpp_count = await self.importer_.getFileAndCppCount()
#                    total_progress = file_count + total_cpp_count
#                except Exception as e:
#                    logger.error(f"Error getting file and cpp count: {e}")
#                    raise FileProcessingError(f"Error getting file and cpp count: {e}")

#                try:
#                    await self.update_total_progress_scan(file_count)
#                    await self.update_total_progress_summarize(total_progress)
#                    await self.update_total_progress_store(total_progress)
#                except Exception as e:
#                    logger.error(f"Error updating summarization and storing progress bars: {e}")
#                    raise FileProcessingError(f"Error updating summarization and storing progress bars: {e}")

                    try:
                        await self.importer_.store_all_cpp(self.projectID)
                        self.stored_cpp_finished = True
    #                        logger.info("Batching to DB of CPP files is complete.")
                        return 0
                    except Exception as e:
                        logger.error(f"Error during CPP storing in Ingest: {e}")
                        await self.cleanup_partial_ingestion(self.projectID)
                        return -1
        # -------------------------------------------------------------------
        return 0
            
#    async def Ingest(self, inputPath: str, topLevelOutputPath: str, currentOutputPath: str) -> int:
#        """
#        Ingest files from the input path into the output path.
#
#        Args:
#            inputPath (str): The path to the input file or directory.
#            currentOutputPath (str): The current output path.
#
#        Returns:
#            int: The result of the ingestion process.
#        """
#        try:
#            if not os.path.exists(inputPath):
#                logger.error(f"Invalid path: {inputPath}")
#                return -1
#
#            inputType = 'd' if os.path.isdir(inputPath) else 'f'
#            index_file_path = os.path.join(currentOutputPath, '.ngest_index')
#
#            with open(index_file_path, 'a') as index_file:
#                index_file.write(f"{inputType},{inputPath}\n")
#            logger.info(f"Scanning: {inputPath}")
#
#            inputLocation, inputName = os.path.split(inputPath)
#            if inputType == 'd':
#                await self.IngestDirectory(inputPath=inputPath, inputLocation=inputLocation, inputName=inputName, currentOutputPath=currentOutputPath, projectID=self.projectID)
#            else:
#                if not self.should_ignore_file(inputPath):
#                    await self.IngestFile(inputPath=inputPath, inputLocation=inputLocation, inputName=inputName, currentOutputPath=currentOutputPath, projectID=self.projectID)
##                    if self.progress_bar:
##                        self.progress_bar.update(1)
#                        
#            await self.importer_.finalize_ingestion('projectID')
#            return 0
#        
#        except Exception as e:
#            logger.error(f"Error during ingestion in Ingest: {e}")
#            await self.cleanup_partial_ingestion(self.projectID)
#            return -1

    async def IngestDirectory(self, inputPath: str, inputLocation: str, inputName: str, topLevelInputPath: str, topLevelOutputPath: str, currentOutputPath: str, projectID: str) -> None:
        """
        Ingest a directory by recursively processing its contents.

        Args:
            inputPath (str): The path to the input directory.
            inputLocation (str): The location of the input directory.
            inputName (str): The name of the input directory.
            currentOutputPath (str): The current output path.
            projectID (str): The project ID.
        """
        newOutputPath = os.path.join(currentOutputPath, inputName)
        os.makedirs(newOutputPath, exist_ok=True)

        tasks = []
        for item in os.listdir(inputPath):
            itemPath = os.path.join(inputPath, item)
            if not self.should_ignore_file(itemPath):
                tasks.append(self.Ingest(itemPath, topLevelInputPath, topLevelOutputPath, newOutputPath))
        await asyncio.gather(*tasks)

    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, topLevelInputPath: str, topLevelOutputPath: str, currentOutputPath: str, projectID: str) -> None:
        """
        Ingest a single file.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            projectID (str): The project ID.
        """
        await self.importer_.IngestFile(inputPath, inputLocation, inputName, topLevelInputPath, topLevelOutputPath, currentOutputPath, projectID)

    def should_ignore_file(self, file_path: str) -> bool:
        """
        Determine if a file should be ignored based on .gitignore patterns.

        Args:
            file_path (str): The path to the file.

        Returns:
            bool: True if the file should be ignored, False otherwise.
        """
        for pattern in self.gitignore_patterns:
            if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(os.path.basename(file_path), pattern):
                return True
        # Explicitly ignore .git directory
        if '.git/' in file_path or file_path.endswith('.git'):
            return True
        if 'doxyfile' in file_path:
            return True
        return False

    def load_gitignore_patterns(self, directory: str) -> List[str]:
        """
        Load .gitignore patterns from a directory.

        Args:
            directory (str): The directory to load patterns from.

        Returns:
            List[str]: A list of .gitignore patterns.
        """
        patterns = []
        gitignore_path = os.path.join(directory, '.gitignore')
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return patterns

    async def update_project(self, projectID: str, inputPath: str) -> int:
        """
        Update a project with new files.

        Args:
            projectID (str): The project ID.
            inputPath (str): The path to the new files.

        Returns:
            int: The result of the update process.
        """
        # Implement project update logic here
        pass

    async def delete_project(self, projectID: str) -> int:
        """
        Delete a project by removing all related data and files.

        Args:
            projectID (str): The project ID.

        Returns:
            int: The result of the deletion process.
        """
        try:
            # Delete from database
            await self.cleanup_partial_ingestion(projectID)
            
            # Delete project directory
            project_path = os.path.expanduser(f"~/.ngest/projects/{projectID}")
            if os.path.exists(project_path):
                shutil.rmtree(project_path)
            
            logger.info(f"Project {projectID} deleted successfully")
            return 0
        except Exception as e:
            logger.error(f"Error deleting project {projectID}: {e}")
            return -1

    async def export_project(self, projectID: str, outputPath: str) -> int:
        """
        Export a project to a JSON file.

        Args:
            projectID (str): The project ID.
            outputPath (str): The path to the output file.

        Returns:
            int: The result of the export process.
        """
        try:
            async with self.importer_.get_session() as session:
                result = await (await session.run(
                    "MATCH (n) WHERE n.projectID = $projectID "
                    "RETURN n",
                    projectID=projectID
                )).consume()
                
                nodes = await result.data()
                
                with open(outputPath, 'w') as f:
                    json.dump(nodes, f)
                
            logger.info(f"Project {projectID} exported successfully to {outputPath}")
            return 0
        except Exception as e:
            logger.error(f"Error exporting project {projectID}: {e}")
            return -1

def preprocess_text(text: str) -> str:
    """
    Preprocess the text by stripping leading and trailing whitespace.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    return text.strip()

async def read_file_in_chunks(file_path: str, chunk_size: int = 1024) -> AsyncGenerator[str, None]:
    """
    Read a file in chunks.

    Args:
        file_path (str): The path to the file.
        chunk_size (int): The size of each chunk.

    Yields:
        str: A chunk of the file.
    """
    try:
        async with aiofiles.open(file_path, 'r') as file:
            while True:
                data = await file.read(chunk_size)
                if not data:
                    break
                yield data
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return

class ProjectManager:

    def __init__(self):
        self.ingest_semaphore = asyncio.Semaphore(1)

    """
    Provides a high-level interface for managing projects, including creating, updating, deleting, and exporting projects.
    """
    async def create_project(self, input_path: str) -> str:
        """
        Create a new project.

        Args:
            input_path (str): The path to the input files.

        Returns:
            str: The project ID.
        """
        async with self.ingest_semaphore:
            project_id = str(uuid.uuid4())
            ingest_instance = NIngest(projectID=project_id)
            result = await ingest_instance.start_ingestion(input_path)
            if result == 0:
#                await ingest_instance.importer_.closeDB()
                return project_id

            else:
                raise Exception("Failed to create project")

    async def update_project(self, project_id: str, input_path: str) -> int:
        """
        Update an existing project with new files.

        Args:
            project_id (str): The project ID.
            input_path (str): The path to the new files.

        Returns:
            int: The result of the update process.
        """
        async with self.ingest_semaphore:
            ingest_instance = NIngest(projectID=project_id)
            return await ingest_instance.update_project(project_id, input_path)

    async def delete_project(self, project_id: str) -> int:
        """
        Delete a project.

        Args:
            project_id (str): The project ID.

        Returns:
            int: The result of the deletion process.
        """
        async with self.ingest_semaphore:
            ingest_instance = NIngest(projectID=project_id)
            return await ingest_instance.delete_project(project_id)

    async def export_project(self, project_id: str, export_path: str) -> int:
        """
        Export a project to a JSON file.

        Args:
            project_id (str): The project ID.
            export_path (str): The path to the output file.

        Returns:
            int: The result of the export process.
        """
        async with self.ingest_semaphore:
            ingest_instance = NIngest(projectID=project_id)
            return await ingest_instance.export_project(project_id, export_path)

async def main():
    """
    Main function to handle command-line arguments and execute the appropriate action.
    """
    parser = argparse.ArgumentParser(description='Ingest files into Neo4j database')
    parser.add_argument('action', choices=['create', 'update', 'delete', 'export'], help='Action to perform')
    parser.add_argument('--input_path', type=str, help='Path to file or directory to ingest')
    parser.add_argument('--project_id', type=str, help='Project ID for update, delete, or export actions')
    parser.add_argument('--export_path', type=str, help='Output path for export action')
    args = parser.parse_args()

    project_manager = ProjectManager()

    try:
        if args.action == 'create':
            if not args.input_path:
                raise ValueError("input_path is required for create action")
            project_id = await project_manager.create_project(args.input_path)
            print(f"Project created successfully. Project ID: {project_id}")
        elif args.action == 'update':
            if not args.project_id or not args.input_path:
                raise ValueError("Both project_id and input_path are required for update action")
            result = await project_manager.update_project(args.project_id, args.input_path)
            print(f"Project update {'succeeded' if result == 0 else 'failed'}")
        elif args.action == 'delete':
            if not args.project_id:
                raise ValueError("project_id is required for delete action")
            result = await project_manager.delete_project(args.project_id)
            print(f"Project deletion {'succeeded' if result == 0 else 'failed'}")
        elif args.action == 'export':
            if not args.project_id or not args.export_path:
                raise ValueError("Both project_id and export_path are required for export action")
            result = await project_manager.export_project(args.project_id, args.export_path)
            print(f"Project export {'succeeded' if result == 0 else 'failed'}")
    except Exception as e:
        print(f"An error occurred: {e}")

class TestNIngest(unittest.TestCase):
    """
    Unit tests for the NIngest class.
    """
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.project_id = str(uuid.uuid4())
        self.ningest = NIngest(projectID=self.project_id)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

#    def test_ingest_file(self):
#        """
#        Test ingesting a single file.
#        """
#        test_file = os.path.join(self.test_dir, 'test.txt')
#        with open(test_file, 'w') as f:
#            f.write("Test content")
#        
#        async def run_test():
#            result = await self.ningest.Ingest(test_file, self.test_dir)
#            self.assertEqual(result, 0)
#
#        asyncio.run(run_test())

#    def test_ingest_directory(self):
#        """
#        Test ingesting a directory.
#        """
#        test_subdir = os.path.join(self.test_dir, 'subdir')
#        os.makedirs(test_subdir)
#        test_file = os.path.join(test_subdir, 'test.txt')
#        with open(test_file, 'w') as f:
#            f.write("Test content")
#        
#        async def run_test():
#            result = await self.ningest.Ingest(self.test_dir, self.test_dir)
#            self.assertEqual(result, 0)
#
#        asyncio.run(run_test())

#    def test_gitignore(self):
#        """
#        Test ignoring files based on .gitignore patterns.
#        """
#        with open(os.path.join(self.test_dir, '.gitignore'), 'w') as f:
#            f.write("*.log\n")
#        
#        test_file = os.path.join(self.test_dir, 'test.txt')
#        with open(test_file, 'w') as f:
#            f.write("Test content")
#        
#        ignored_file = os.path.join(self.test_dir, 'ignored.log')
#        with open(ignored_file, 'w') as f:
#            f.write("Ignored content")
#        
#        async def run_test():
#            result = await self.ningest.Ingest(self.test_dir, self.test_dir)
#            self.assertEqual(result, 0)
#            self.assertTrue(os.path.exists(os.path.join(self.ningest.currentOutputPath, 'test.txt')))
#            self.assertFalse(os.path.exists(os.path.join(self.ningest.currentOutputPath, 'ignored.log')))
#
#        asyncio.run(run_test())

if __name__ == "__main__":
    asyncio.run(main())
