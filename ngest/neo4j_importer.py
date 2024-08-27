# Copyright 2024 Chris Odom
# MIT License

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
#from clang.cindex import Config
#Config.set_library_path("/opt/homebrew/opt/llvm/lib")
import clang.cindex
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

from ngest.base_importer import NBaseImporter
from ngest.project import Project
from ngest.file import File
from ngest.document import Document
from ngest.pdf import PDF

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

class NNeo4JImporter(NBaseImporter):
    """
    A file importer that ingests various file types into a Neo4j database.
    """
    def __init__(self, neo4j_url: str = NEO4J_URL, neo4j_user: str = NEO4J_USER, neo4j_password: str = NEO4J_PASSWORD):
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

        self.rate_limiter_db = AsyncLimiter(50, 1)  # 3 operations per 1 seconds
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
                logger.error(f"Error reading file {input_path}: {e}")
                raise FileProcessingError(f"Error reading file {input_path}: {e}")

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
            class_name, full_scope, interface_summary, interface_description = await self.summarize_cpp_class_public_interface(class_info)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_public_interface for {class_name}: {e}")
            raise DatabaseError(f"Error in summarize_cpp_class_public_interface for {class_name}: {e}")
        try:
            implementation_summary, implementation_description = await self.summarize_cpp_class_implementation(class_info)
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
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function calling do_summarize_text: {e}")
            raise DatabaseError(f"Error in summarize_cpp_function calling do_summarize_text: {e}")
        
        try:
            retval = full_name, full_scope, summary + "\n\n" + description
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function appending strings: {e}\nnode_info contents:\n{pprint.pformat(node_info, indent=4)}")
            raise DatabaseError(f"Error in summarize_cpp_function appending strings: {e}\nnode_info contents:\n{pprint.pformat(node_info, indent=4)}")

        return retval

    async def IngestCpp(self, inputPath: str, inputLocation: str, inputName: str, localPath: str, currentOutputPath: str, project_id: str):
        try:
            index = clang.cindex.Index.create()
            translation_unit = index.parse(inputPath)

            with open(inputPath, 'r') as file:
                file_contents = file.read()

            # Save raw code of header files
            if inputPath.endswith('.hpp') or inputPath.endswith('.h'):
                    async with self.lock_header_files:
                        self.header_files[inputPath] = file_contents
            
            await self.process_nodes(translation_unit.cursor, inputLocation, project_id, inputPath.endswith('.cpp'))
                
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

    async def process_nodes(self, node, project_path, project_id, is_cpp_file):
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
                    
                    await self.process_nodes(child, project_path, project_id, is_cpp_file)
                    
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
                    await self.process_nodes(child, project_path, project_id, is_cpp_file)

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

    async def summarize_all_cpp(self, project_id):
        try:
            tasks = []
            async with self.lock_classes:
                for class_name, class_info in self.classes.items():
                    name = class_info.get('name', 'anonymous')
                    class_info['name'] = name
                    info_copy = copy.deepcopy(class_info)
                    tasks.append(self.summarize_cpp_class_prep(name, info_copy))

            async with self.lock_methods:
                for method_name, method_info in self.methods.items():
                    name = method_info.get('name', 'anonymous')
                    method_info['name'] = name
                    info_copy = copy.deepcopy(method_info)
                    tasks.append(self.summarize_cpp_method_prep(name, info_copy))

            async with self.lock_functions:
                for function_name, function_info in self.functions.items():
                    name = function_info.get('name', 'anonymous')
                    function_info['name'] = name
                    info_copy = copy.deepcopy(function_info)
                    tasks.append(self.summarize_cpp_function_prep(name, info_copy))

        except Exception as e:
            logger.error(f"Error while appending tasks to queue: {e}")
            raise FileProcessingError(f"Error while appending tasks to queue: {e}")

        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error while gathering summarization and embedding tasks: {e}")
            raise FileProcessingError(f"Error while gathering summarization and embedding tasks: {e}")

    async def summarize_cpp_class_prep(self, name, class_info):
        try:
            if name is None:
                logger.info(f"Skipping anonymous class")
                return class_info
                
            class_name, class_scope, interface_summary, implementation_summary = await self.summarize_cpp_class(class_info)
            
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_prep calling summarize_cpp_class: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_class_prep calling summarize_cpp_class: {e}")

        try:
            interface_embedding = await self.make_embedding(interface_summary)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_prep calling make_embedding for interface_summary: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_class_prep calling make_embedding for interface_summary: {e}")
        
        try:
            implementation_embedding = await self.make_embedding(implementation_summary)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_prep calling make_embedding for implementation_summary: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_class_prep calling make_embedding for implementation_summary: {e}")
        
        details = {
            'interface_summary': interface_summary,
            'implementation_summary': implementation_summary,
            'interface_embedding': interface_embedding,
            'implementation_embedding': implementation_embedding
        }
        await self.update_classes(name, details)
        await self.update_progress_summarize(1)
        logger.info(f"Finished summarizing/embedding class: {name}")
        return class_info

    async def summarize_cpp_method_prep(self, name, method_info):
        try:
            if name is None:
                logger.info(f"Skipping anonymous method")
                return method_info
            function_name, function_scope, function_summary = await self.summarize_cpp_function(method_info)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_method_prep: {e}\nmethod_info contents:\n{pprint.pformat(method_info, indent=4)}")
            raise FileProcessingError(f"Error in summarize_cpp_method_prep calling summarize_cpp_function for method {name}: {e}")
        
        try:
            embedding = await self.make_embedding(function_summary)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_method_prep calling make_embedding: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_method_prep calling make_embedding: {e}")

        details = {
            'summary': function_summary,
            'embedding': embedding
        }
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
        
        try:
            function_name, function_scope, function_summary = await self.summarize_cpp_function(function_info)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function_prep calling summarize_cpp_function: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_function_prep calling summarize_cpp_function: {e}")

        try:
            embedding = await self.make_embedding(function_summary)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function_prep calling make_embedding: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_function_prep calling make_embedding: {e}")
        
        details = {
            'summary': function_summary,
            'embedding': embedding
        }
        try:
            await self.update_functions(name, details)
            await self.update_progress_summarize(1)
            logger.info(f"Finished summarizing/embedding cpp function: {name}")
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function_prep calling update_functions: {e}")
            raise FileProcessingError(f"Error in summarize_cpp_function_prep calling update_functions: {e}")

        return function_info

    async def store_all_cpp(self, project_id):
        try:
            queue = asyncio.Queue()

            # Create a task to process the queue
            process_task = asyncio.create_task(self.process_cpp_storage_queue(queue, project_id))

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
            # Signal that all items have been added to the queue
            await queue.put(None)

            # Wait for all items to be processed
            await process_task

        except Exception as e:
            logger.error(f"Error while gathering storage tasks: {e}")
            raise FileProcessingError(f"Error while gathering storage tasks: {e}")

    async def process_cpp_storage_queue(self, queue, project_id):
        while True:
            item = await queue.get()
            if item is None:
                break
            item_type, name, info = item
            if item_type == 'Class':
                await self.store_summary_cpp_class(name, info, project_id)
            elif item_type == 'Method':
                await self.store_summary_cpp_method(name, info, project_id)
            elif item_type == 'Function':
                await self.store_summary_cpp_function(name, info, project_id)
            await self.update_progress_store(1)
            queue.task_done()

    async def store_summary_cpp_class(self, name, info, project_id):
        try:
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

            query = """
                MERGE (n:{type} {{name: $name, project_id: $project_id}})
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
                'project_id': project_id if project_id else 'Unknown',
                'scope': scope,
                'short_name': short_name,
                'interface_summary': interface_summary,
                'implementation_summary': implementation_summary,
                'interface_embedding': interface_embedding,
                'implementation_embedding': implementation_embedding,
                'file_path': file_path,
                'raw_code': raw_code
            }
        except Exception as e:
            logger.error(f"Error forming query for CPP class {name}: {e}")
            raise FileProcessingError(f"Error forming query for CPP class {name}: {e}")

        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    await self.run_query_and_get_element_id(session, query, **params)
                    logger.info(f"Finished storing CPP class: {name}")
        except Exception as e:
            logger.error(f"Error storing summary for CPP class {name}: {e}")
            raise FileProcessingError(f"Error storing summary for CPP class {name}: {e}")

    async def store_summary_cpp_method(self, name, info, project_id):
        try:
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
                MERGE (n:{type} {{name: $name, project_id: $project_id}})
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
                'project_id': project_id if project_id is not None else 'Unknown',
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
                            "WHERE (c.name = $scope AND c:Class AND c.project_id = $project_id) "
                            "OR (c.name = $scope AND c:Struct AND c.project_id = $project_id) "
                            "MATCH (f:Method) WHERE elementId(f) = $element_id AND f.project_id = $project_id "
                            "MERGE (c)-[:HAS_METHOD]->(f) "
                            "MERGE (f)-[:BELONGS_TO]->(c)",
                            {"name": name, "scope": scope, "element_id": element_id, "project_id": project_id}
                        )).consume()
                       
                        logger.info(f"Finished storing CPP method: {name}")

        except Exception as e:
            logger.error(f"Error storing summary for CPP method {name}: {e}")
            raise FileProcessingError(f"Error storing summary for CPP method {name}: {e}")

    async def store_summary_cpp_function(self, name, info, project_id):
        try:
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
                MERGE (n:{type} {{name: $name, project_id: $project_id}})
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
                'project_id': project_id if project_id is not None else 'Unknown',
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

    def chunk_text(self, text: str, chunk_size: int) -> List[str]:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# End of NNeo4JImporter class

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

def preprocess_text(text: str) -> str:
    """
    Preprocess the text by stripping leading and trailing whitespace.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    return text.strip()

