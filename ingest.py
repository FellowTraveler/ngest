# Copyright 2024 Chris Odom
# MIT License

import os
import uuid
import logging
from abc import ABC, abstractmethod
import asyncio
from neo4j import AsyncGraphDatabase
import PIL.Image
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import pipeline, AutoTokenizer, AutoModel
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
from tqdm import tqdm
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
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
DEFAULT_NEO4J_URL = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "password"
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
    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        pass

    def ascertain_file_type(self, filename: str) -> str:
        import magic
        try:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(filename)
        except Exception as e:
            logger.error(f"Error determining file type for {filename}: {e}")
            return 'unknown'
        
        if file_type.startswith('text'):
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.cpp', '.hpp', '.h', '.c']:
                return 'cpp'
            elif ext == '.py':
                return 'python'
            elif ext == '.rs':
                return 'rust'
            elif ext == '.js':
                return 'javascript'
            else:
                return 'text'
        elif file_type.startswith('image'):
            return 'image'
        elif file_type == 'application/pdf':
            return 'pdf'
        else:
            return 'unknown'

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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
        async with self.driver.session() as session:
            try:
                chunks = [chunk async for chunk in read_file_in_chunks(inputPath, MEDIUM_CHUNK_SIZE)]
                previous_chunk_id = None

                for i, chunk in enumerate(chunks):
                    chunk_id = await self.handle_chunk_creation(session, chunk, inputPath, i, projectID)
                    if previous_chunk_id:
                        await self.create_chunk_relationship(session, previous_chunk_id, chunk_id, projectID)
                    previous_chunk_id = chunk_id
            except Exception as e:
                logger.error(f"Error processing file {inputPath}: {e}")
                raise FileProcessingError(f"Error processing file {inputPath}: {e}")

    def chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Chunk the given text into smaller pieces.

        Args:
            text (str): The text to be chunked.
            chunk_size (int): The size of each chunk.

        Returns:
            List[str]: A list of text chunks.
        """
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def create_chunk_node(self, session, chunk: str, inputPath: str, index: int, projectID: str) -> Optional[str]:
        """
        Create a chunk node in the Neo4j database.

        Args:
            session: The Neo4j session.
            chunk (str): The text chunk.
            inputPath (str): The path to the input file.
            index (int): The index of the chunk.
            projectID (str): The project ID.

        Returns:
            Optional[str]: The chunk ID, or None if creation failed.
        """
        embedding = self.model.embed_text(chunk)
        chunk_id = f"{inputPath}_chunk_{index}"
        try:
            result = await session.run(
                "CREATE (n:Chunk {id: $id, content: $content, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                id=chunk_id, content=chunk, embedding=embedding, projectID=projectID
            )
            node = result.single()
            if node:
                node_id = node['id']
                logger.info(f"Created chunk node {chunk_id} with embedding")
                return chunk_id
            else:
                logger.error(f"No result returned for chunk node {chunk_id}")
                return None
        except Exception as e:
            logger.error(f"Error creating chunk node: {e}")
            raise DatabaseError(f"Error creating chunk node: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def create_chunk_relationship(self, session, chunk_id1: str, chunk_id2: str, projectID: str) -> None:
        """
        Create a relationship between two chunks in the Neo4j database.

        Args:
            session: The Neo4j session.
            chunk_id1 (str): The ID of the first chunk.
            chunk_id2 (str): The ID of the second chunk.
            projectID (str): The project ID.
        """
        try:
            result = await session.run(
                "MATCH (c1:Chunk {id: $id1, projectID: $projectID}), (c2:Chunk {id: $id2, projectID: $projectID}) "
                "CREATE (c1)-[:NEXT]->(c2)",
                id1=chunk_id1, id2=chunk_id2, projectID=projectID
            )
            if result:
                logger.info(f"Created relationship between {chunk_id1} and {chunk_id2}")
            else:
                logger.error(f"No result returned for creating relationship between {chunk_id1} and {chunk_id2}")
        except Exception as e:
            logger.error(f"Error creating chunk relationship: {e}")
            raise DatabaseError(f"Error creating chunk relationship: {e}")

    async def handle_chunk_creation(self, session, chunk, inputPath, index, projectID):
        """
        Handle the creation of a chunk node.

        Args:
            session: The Neo4j session.
            chunk: The text chunk.
            inputPath: The path to the input file.
            index: The index of the chunk.
            projectID: The project ID.

        Returns:
            The chunk ID.
        """
        try:
            chunk_id = await self.create_chunk_node(session, chunk, inputPath, index, projectID)
            return chunk_id
        except Exception as e:
            logger.error(f"Error creating chunk node for chunk {index} in file {inputPath}: {e}")
            raise DatabaseError(f"Error creating chunk node for chunk {index} in file {inputPath}: {e}")

class NFilesystemImporter(NBaseImporter):
    """
    A file importer that simply copies files to the output directory.
    """
    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
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
        self.index = clang.cindex.Index.create()
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=self.device)
        
        self.image_model = models.densenet121(pretrained=True).to(self.device)
        self.image_model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.code_model = AutoModel.from_pretrained("microsoft/codebert-base").to(self.device)
        self.code_model.eval()
        
        self.model = ollama.Model(EMBEDDING_MODEL)  # Reuse the model instance
        
        self.rate_limiter = aiorate.Limiter(10, 1)  # 10 operations per second
        
        logger.info(f"Neo4J importer initialized with URL {neo4j_url}")
        
    @asynccontextmanager
    async def get_session(self):
        """
        Get a session for Neo4j database operations.

        Yields:
            The Neo4j session.
        """
        async with self.driver.session() as session:
            try:
                yield session
            finally:
                await session.close()

    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        """
        Ingest a file by determining its type and using the appropriate ingestion method.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            projectID (str): The project ID.
        """
        try:
            file_type = self.ascertain_file_type(inputPath)
        except Exception as e:
            logger.error(f"Error ascertaining file type for {inputPath}: {e}")
            raise FileProcessingError(f"Error ascertaining file type for {inputPath}: {e}")

        ingest_method = getattr(self, f"Ingest{file_type.capitalize()}", None)
        if ingest_method:
            try:
                await ingest_method(inputPath, inputLocation, inputName, currentOutputPath, projectID)
            except Exception as e:
                logger.error(f"Error ingesting {file_type} file {inputPath}: {e}")
                raise FileProcessingError(f"Error ingesting {file_type} file {inputPath}: {e}")
        else:
            logger.warning(f"No ingest method for file type: {file_type}")

    async def summarize_text(self, text: str) -> str:
        """
        Summarize the given text using a pre-trained model.

        Args:
            text (str): The text to be summarized.

        Returns:
            str: The summary of the text.
        """
        try:
            return (await asyncio.to_thread(self.summarizer, text, max_length=50, min_length=25, do_sample=False))[0]['summary_text']
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return ""

    async def process_chunks(self, session, text, parent_node_id, parent_node_type, chunk_size, chunk_type, projectID):
        """
        Process text chunks and create nodes in the database.

        Args:
            session: The Neo4j session.
            text: The text to be chunked and processed.
            parent_node_id: The parent node ID.
            parent_node_type: The parent node type.
            chunk_size: The size of each chunk.
            chunk_type: The type of chunk.
            projectID: The project ID.
        """
        chunks = self.chunk_text(text, chunk_size)
        for i, chunk in enumerate(chunks):
            async with self.rate_limiter:
                await self.create_chunk_with_embedding(session, chunk, parent_node_id, parent_node_type, chunk_type, projectID)

    async def create_chunk_with_embedding(self, session, chunk, parent_node_id, parent_node_type, chunk_type, projectID):
        """
        Create a chunk node with an embedding in the database.

        Args:
            session: The Neo4j session.
            chunk: The text chunk.
            parent_node_id: The parent node ID.
            parent_node_type: The parent node type.
            chunk_type: The type of chunk.
            projectID: The project ID.
        """
        try:
            chunk_node = await session.run(
                f"CREATE (n:{chunk_type} {{content: $content, type: $chunk_type, projectID: $projectID}}) RETURN id(n)",
                content=chunk, chunk_type=chunk_type, projectID=projectID
            )
            chunk_id = chunk_node.single()['id']
            await session.run(
                f"MATCH (p:{parent_node_type}), (c:{chunk_type}) WHERE id(p) = $parent_node_id AND id(c) = $chunk_id AND p.projectID = $projectID AND c.projectID = $projectID "
                f"CREATE (p)-[:HAS_CHUNK]->(c), (c)-[:PART_OF]->(p)",
                parent_node_id=parent_node_id, chunk_id=chunk_id, projectID=projectID
            )
            embedding = self.model.embed_text(chunk)
            embedding_node = await session.run(
                "CREATE (n:Embedding {embedding: $embedding, type: 'embedding', projectID: $projectID}) RETURN id(n)",
                embedding=embedding, projectID=projectID
            )
            embedding_id = embedding_node.single()['id']
            await session.run(
                f"MATCH (c:{chunk_type}), (e:Embedding) WHERE id(c) = $chunk_id AND id(e) = $embedding_id AND c.projectID = $projectID AND e.projectID = $projectID "
                f"CREATE (c)-[:HAS_EMBEDDING]->(e)",
                chunk_id=chunk_id, embedding_id=embedding_id, projectID=projectID
            )
        except Exception as e:
            logger.error(f"Error creating chunk with embedding: {e}")
            raise DatabaseError(f"Error creating chunk with embedding: {e}")

    async def IngestTxt(self, input_path: str, input_location: str, input_name: str, current_output_path: str, project_id: str) -> None:
        """
        Ingest a text file by reading its content and creating document and chunk nodes in the database.

        Args:
            input_path (str): The path to the input file.
            input_location (str): The location of the input file.
            input_name (str): The name of the input file.
            current_output_path (str): The current output path.
            project_id (str): The project ID.
        """
        async with self.get_session() as session:
            try:
                async with aiofiles.open(input_path, 'r') as file:
                    file_content = await file.read()
            except Exception as e:
                logger.error(f"Error reading file {input_path}: {e}")
                raise FileProcessingError(f"Error reading file {input_path}: {e}")

            try:
                parent_doc_node = await session.run(
                    "CREATE (n:Document {name: $name, type: 'text', projectID: $projectID}) RETURN id(n)",
                    name=input_name, projectID=project_id
                )
                parent_doc_id = parent_doc_node.single()['id']

                await self.process_chunks(session, file_content, parent_doc_id, "Document", MEDIUM_CHUNK_SIZE, "MediumChunk", project_id)

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

        async with self.get_session() as session:
            await self.store_image_features(session, features, input_name, project_id)

    async def store_image_features(self, session, features: List[float], input_name: str, project_id: str):
        """
        Store the extracted image features in the database.

        Args:
            session: The Neo4j session.
            features (List[float]): The extracted features.
            input_name (str): The name of the input file.
            project_id (str): The project ID.
        """
        try:
            async with self.rate_limiter:
                image_node = await session.run(
                    "CREATE (n:Image {name: $name, type: 'image', projectID: $projectID}) RETURN id(n)",
                    name=input_name, projectID=project_id
                )
                image_id = image_node.single()['id']
                features_node = await session.run(
                    "CREATE (n:ImageFeatures {features: $features, type: 'image_features', projectID: $projectID}) RETURN id(n)",
                    features=features, projectID=project_id
                )
                features_id = features_node.single()['id']
                await session.run(
                    "MATCH (i:Image), (f:ImageFeatures) WHERE id(i) = $image_id AND id(f) = $features_id AND i.projectID = $projectID AND f.projectID = $projectID "
                    "CREATE (i)-[:HAS_FEATURES]->(f)",
                    image_id=image_id, features_id=features_id, projectID=project_id
                )
        except Exception as e:
            logger.error(f"Error storing image features: {e}")
            raise DatabaseError(f"Error storing image features: {e}")

    async def summarize_cpp_class(self, cls) -> str:
        """
        Summarize a C++ class.

        Args:
            cls: The class node.

        Returns:
            str: The summary of the class.
        """
        description = f"Class {cls.spelling} with public methods: "
        public_methods = [c.spelling for c in cls.get_children() if c.access_specifier == clang.cindex.AccessSpecifier.PUBLIC]
        description += ", ".join(public_methods)
        return await self.summarize_text(description)

    async def summarize_cpp_function(self, func) -> str:
        """
        Summarize a C++ function.

        Args:
            func: The function node.

        Returns:
            str: The summary of the function.
        """
        description = f"Function {func.spelling} in namespace {func.semantic_parent.spelling}. It performs the following tasks: "
        return await self.summarize_text(description)

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

    async def IngestCpp(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        """
        Ingest a C++ file by parsing it, summarizing classes and functions, and storing them in the database.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            projectID (str): The project ID.
        """
        try:
            tu = self.index.parse(inputPath)
            functions = []
            classes = []

            for node in tu.cursor.get_children():
                if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                    functions.append(node)
                elif node.kind in [clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL]:
                    classes.append(node)

            async with self.get_session() as session:
                for cls in classes:
                    class_summary = await self.summarize_cpp_class(cls)
                    embedding = self.model.embed_text(class_summary)
                    async with self.rate_limiter:
                        class_node = await session.run(
                            "CREATE (n:Class {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                            name=cls.spelling, summary=class_summary, embedding=embedding, projectID=projectID
                        )
                        class_id = class_node.single()['id']
                    
                    for func in cls.get_children():
                        if func.kind == clang.cindex.CursorKind.CXX_METHOD:
                            function_summary = await self.summarize_cpp_function(func)
                            embedding = self.model.embed_text(function_summary)
                            async with self.rate_limiter:
                                function_node = await session.run(
                                    "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                                    name=func.spelling, summary=function_summary, embedding=embedding, projectID=projectID
                                )
                                function_id = function_node.single()['id']
                                await session.run(
                                    "MATCH (c:Class), (f:Function) WHERE id(c) = $class_id AND id(f) = $function_id AND c.projectID = $projectID AND f.projectID = $projectID "
                                    "CREATE (c)-[:HAS_METHOD]->(f)",
                                    class_id=class_id, function_id=function_id, projectID=projectID
                                )

                for func in functions:
                    function_summary = await self.summarize_cpp_function(func)
                    embedding = self.model.embed_text(function_summary)
                    async with self.rate_limiter:
                        await session.run(
                            "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID})",
                            name=func.spelling, summary=function_summary, embedding=embedding, projectID=projectID
                        )
        except Exception as e:
            logger.error(f"Error ingesting C++ file {inputPath}: {e}")
            raise FileProcessingError(f"Error ingesting C++ file {inputPath}: {e}")

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

            async with self.get_session() as session:
                for cls in classes:
                    class_summary = await self.summarize_python_class(cls)
                    embedding = self.model.embed_text(class_summary)
                    async with self.rate_limiter:
                        class_node = await session.run(
                            "CREATE (n:Class {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                            name=cls.name, summary=class_summary, embedding=embedding, projectID=projectID
                        )
                        class_id = class_node.single()['id']
                    
                    for func in cls.body:
                        if isinstance(func, ast.FunctionDef):
                            function_summary = await self.summarize_python_function(func)
                            embedding = self.model.embed_text(function_summary)
                            async with self.rate_limiter:
                                function_node = await session.run(
                                    "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                                    name=func.name, summary=function_summary, embedding=embedding, projectID=projectID
                                )
                                function_id = function_node.single()['id']
                                await session.run(
                                    "MATCH (c:Class), (f:Function) WHERE id(c) = $class_id AND id(f) = $function_id AND c.projectID = $projectID AND f.projectID = $projectID "
                                    "CREATE (c)-[:HAS_METHOD]->(f)",
                                    class_id=class_id, function_id=function_id, projectID=projectID
                                )

                for func in functions:
                    function_summary = await self.summarize_python_function(func)
                    embedding = self.model.embed_text(function_summary)
                    async with self.rate_limiter:
                        await session.run(
                            "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID})",
                            name=func.name, summary=function_summary, embedding=embedding, projectID=projectID
                        )
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
        return await self.summarize_text(description)

    async def summarize_python_function(self, func) -> str:
        """
        Summarize a Python function.

        Args:
            func: The function node.

        Returns:
            str: The summary of the function.
        """
        description = f"Function {func.name} with arguments: {', '.join(arg.arg for arg in func.args.args)}. It performs the following tasks: "
        return await self.summarize_text(description)

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

            async with self.get_session() as session:
                for impl in impls:
                    impl_summary = await self.summarize_rust_impl(impl)
                    embedding = self.model.embed_text(impl_summary)
                    async with self.rate_limiter:
                        impl_node = await session.run(
                            "CREATE (n:Impl {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                            name=impl.trait_.path.segments[0].ident, summary=impl_summary, embedding=embedding, projectID=projectID
                        )
                        impl_id = impl_node.single()['id']
                    
                    for item in impl.items:
                        if isinstance(item, syn.ImplItemMethod):
                            function_summary = await self.summarize_rust_function(item)
                            embedding = self.model.embed_text(function_summary)
                            async with self.rate_limiter:
                                function_node = await session.run(
                                    "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                                    name=item.sig.ident, summary=function_summary, embedding=embedding, projectID=projectID
                                )
                                function_id = function_node.single()['id']
                                await session.run(
                                    "MATCH (i:Impl), (f:Function) WHERE id(i) = $impl_id AND id(f) = $function_id AND i.projectID = $projectID AND f.projectID = $projectID "
                                    "CREATE (i)-[:HAS_METHOD]->(f)",
                                    impl_id=impl_id, function_id=function_id, projectID=projectID
                                )

                for func in functions:
                    function_summary = await self.summarize_rust_function(func)
                    embedding = self.model.embed_text(function_summary)
                    async with self.rate_limiter:
                        await session.run(
                            "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID})",
                            name=func.sig.ident, summary=function_summary, embedding=embedding, projectID=projectID
                        )
        except Exception as e:
            logger.error(f"Error ingesting Rust file {inputPath}: {e}")
            raise FileProcessingError(f"Error ingesting Rust file {inputPath}: {e}")

    async def summarize_rust_impl(self, impl) -> str:
        """
        Summarize a Rust implementation.

        Args:
            impl: The implementation node.

        Returns:
            str: The summary of the implementation.
        """
        description = f"Implementation of trait {impl.trait_.path.segments[0].ident} with methods: "
        methods = [item.sig.ident for item in impl.items if isinstance(item, syn.ImplItemMethod)]
        description += ", ".join([str(m) for m in methods])
        return await self.summarize_text(description)

    async def summarize_rust_function(self, func) -> str:
        """
        Summarize a Rust function.

        Args:
            func: The function node.

        Returns:
            str: The summary of the function.
        """
        description = f"Function {func.sig.ident} with arguments: {', '.join(arg.pat.ident for arg in func.sig.inputs)}. It performs the following tasks: "
        return await self.summarize_text(description)

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

        async with self.get_session() as session:
            try:
                async with self.rate_limiter:
                    file_node = await session.run(
                        "CREATE (n:JavaScriptFile {name: $name, path: $path, projectID: $projectID}) RETURN id(n)",
                        name=inputName, path=inputPath, projectID=projectID
                    )
                    file_id = file_node.single()['id']

                for node in ast.body:
                    if isinstance(node, nodes.FunctionDeclaration):
                        await self.process_js_function(session, node, file_id, projectID)
                    elif isinstance(node, nodes.ClassDeclaration):
                        await self.process_js_class(session, node, file_id, projectID)
                    elif isinstance(node, nodes.VariableDeclaration):
                        await self.process_js_variable(session, node, file_id, projectID)
            except Exception as e:
                logger.error(f"Error ingesting JavaScript file {inputPath}: {e}")
                raise DatabaseError(f"Error ingesting JavaScript file {inputPath}: {e}")

    async def process_js_function(self, session, func_node, file_id: int, projectID: str) -> None:
        """
        Process a JavaScript function node and create nodes in the database.

        Args:
            session: The Neo4j session.
            func_node: The function node.
            file_id: The file ID.
            projectID: The project ID.
        """
        func_name = func_node.id.name if func_node.id else 'anonymous'
        func_summary = await self.summarize_js_function(func_node)
        embedding = self.model.embed_text(func_summary)

        try:
            async with self.rate_limiter:
                func_db_node = await session.run(
                    "CREATE (n:JavaScriptFunction {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                    name=func_name, summary=func_summary, embedding=embedding, projectID=projectID
                )
                func_db_id = func_db_node.single()['id']

                await session.run(
                    "MATCH (f:JavaScriptFile), (func:JavaScriptFunction) WHERE id(f) = $file_id AND id(func) = $func_id AND f.projectID = $projectID AND func.projectID = $projectID "
                    "CREATE (f)-[:CONTAINS]->(func)",
                    file_id=file_id, func_id=func_db_id, projectID=projectID
                )
        except Exception as e:
            logger.error(f"Error processing JavaScript function {func_name}: {e}")
            raise DatabaseError(f"Error processing JavaScript function {func_name}: {e}")

    async def process_js_class(self, session, class_node, file_id: int, projectID: str) -> None:
        """
        Process a JavaScript class node and create nodes in the database.

        Args:
            session: The Neo4j session.
            class_node: The class node.
            file_id: The file ID.
            projectID: The project ID.
        """
        class_name = class_node.id.name
        class_summary = await self.summarize_js_class(class_node)
        embedding = self.model.embed_text(class_summary)

        try:
            async with self.rate_limiter:
                class_db_node = await session.run(
                    "CREATE (n:JavaScriptClass {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                    name=class_name, summary=class_summary, embedding=embedding, projectID=projectID
                )
                class_db_id = class_db_node.single()['id']

                await session.run(
                    "MATCH (f:JavaScriptFile), (c:JavaScriptClass) WHERE id(f) = $file_id AND id(c) = $class_id AND f.projectID = $projectID AND c.projectID = $projectID "
                    "CREATE (f)-[:CONTAINS]->(c)",
                    file_id=file_id, class_id=class_db_id, projectID=projectID
                )

            for method in class_node.body.body:
                if isinstance(method, nodes.MethodDefinition):
                    await self.process_js_method(session, method, class_db_id, projectID)
        except Exception as e:
            logger.error(f"Error processing JavaScript class {class_name}: {e}")
            raise DatabaseError(f"Error processing JavaScript class {class_name}: {e}")

    async def process_js_variable(self, session, var_node, file_id: int, projectID: str) -> None:
        """
        Process a JavaScript variable node and create nodes in the database.

        Args:
            session: The Neo4j session.
            var_node: The variable node.
            file_id: The file ID.
            projectID: The project ID.
        """
        for declaration in var_node.declarations:
            var_name = declaration.id.name
            var_type = var_node.kind  # 'var', 'let', or 'const'
            var_summary = await self.summarize_js_variable(var_node)
            embedding = self.model.embed_text(var_summary)

            try:
                async with self.rate_limiter:
                    var_db_node = await session.run(
                        "CREATE (n:JavaScriptVariable {name: $name, type: $type, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                        name=var_name, type=var_type, summary=var_summary, embedding=embedding, projectID=projectID
                    )
                    var_db_id = var_db_node.single()['id']

                    await session.run(
                        "MATCH (f:JavaScriptFile), (v:JavaScriptVariable) WHERE id(f) = $file_id AND id(v) = $var_id AND f.projectID = $projectID AND v.projectID = $projectID "
                        "CREATE (f)-[:CONTAINS]->(v)",
                        file_id=file_id, var_id=var_db_id, projectID=projectID
                    )
            except Exception as e:
                logger.error(f"Error processing JavaScript variable {var_name}: {e}")
                raise DatabaseError(f"Error processing JavaScript variable {var_name}: {e}")

    async def process_js_method(self, session, method_node, class_id: int, projectID: str) -> None:
        """
        Process a JavaScript method node and create nodes in the database.

        Args:
            session: The Neo4j session.
            method_node: The method node.
            class_id: The class ID.
            projectID: The project ID.
        """
        method_name = method_node.key.name
        method_summary = await self.summarize_js_function(method_node.value)
        embedding = self.model.embed_text(method_summary)

        try:
            async with self.rate_limiter:
                method_db_node = await session.run(
                    "CREATE (n:JavaScriptMethod {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                    name=method_name, summary=method_summary, embedding=embedding, projectID=projectID
                )
                method_db_id = method_db_node.single()['id']

                await session.run(
                    "MATCH (c:JavaScriptClass), (m:JavaScriptMethod) WHERE id(c) = $class_id AND id(m) = $method_id AND c.projectID = $projectID AND m.projectID = $projectID "
                    "CREATE (c)-[:HAS_METHOD]->(m)",
                    class_id=class_id, method_id=method_db_id, projectID=projectID
                )
        except Exception as e:
            logger.error(f"Error processing JavaScript method {method_name}: {e}")
            raise DatabaseError(f"Error processing JavaScript method {method_name}: {e}")

    async def summarize_js_function(self, func) -> str:
        """
        Summarize a JavaScript function.

        Args:
            func: The function node.

        Returns:
            str: The summary of the function.
        """
        description = f"Function {func.id.name if func.id else 'anonymous'} with arguments: {', '.join(param.name for param in func.params)}. It performs the following tasks: "
        return await self.summarize_text(description)

    async def summarize_js_class(self, cls) -> str:
        """
        Summarize a JavaScript class.

        Args:
            cls: The class node.

        Returns:
            str: The summary of the class.
        """
        description = f"Class {cls.id.name} with methods: "
        methods = [method.key.name for method in cls.body.body if isinstance(method, nodes.MethodDefinition)]
        description += ", ".join(methods)
        return await self.summarize_text(description)

    async def summarize_js_variable(self, var) -> str:
        """
        Summarize a JavaScript variable.

        Args:
            var: The variable node.

        Returns:
            str: The summary of the variable.
        """
        description = f"Variable {var.declarations[0].id.name} of type {var.kind}. It is initialized as: "
        return await self.summarize_text(description)

    async def IngestPdf(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        """
        Ingest a PDF file by reading its content, chunking it, and storing the chunks and their embeddings in the database.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            projectID (str): The project ID.
        """
        try:
            reader = PdfReader(inputPath)
            num_pages = len(reader.pages)

            async with self.get_session() as session:
                async with self.rate_limiter:
                    pdf_node = await session.run(
                        "CREATE (n:PDF {name: $name, pages: $pages, projectID: $projectID}) RETURN id(n)",
                        name=inputName, pages=num_pages, projectID=projectID
                    )
                    pdf_id = pdf_node.single()['id']

                for i in range(num_pages):
                    page = reader.pages[i]
                    text = page.extract_text()

                    medium_chunks = self.chunk_text(text, MEDIUM_CHUNK_SIZE)
                    for j, medium_chunk in enumerate(medium_chunks):
                        async with self.rate_limiter:
                            medium_chunk_node = await session.run(
                                "CREATE (n:MediumChunk {content: $content, type: 'medium_chunk', page: $page, projectID: $projectID}) RETURN id(n)",
                                content=medium_chunk, page=i+1, projectID=projectID
                            )
                            medium_chunk_id = medium_chunk_node.single()['id']

                            await session.run(
                                "MATCH (p:PDF), (c:MediumChunk) WHERE id(p) = $pdf_id AND id(c) = $chunk_id AND p.projectID = $projectID AND c.projectID = $projectID "
                                "CREATE (p)-[:HAS_CHUNK]->(c), (c)-[:PART_OF]->(p)",
                                pdf_id=pdf_id, chunk_id=medium_chunk_id, projectID=projectID
                            )

                        small_chunks = self.chunk_text(medium_chunk, SMALL_CHUNK_SIZE)
                        for k, small_chunk in enumerate(small_chunks):
                            async with self.rate_limiter:
                                small_chunk_node = await session.run(
                                    "CREATE (n:SmallChunk {content: $content, type: 'small_chunk', page: $page, projectID: $projectID}) RETURN id(n)",
                                    content=small_chunk, page=i+1, projectID=projectID
                                )
                                small_chunk_id = small_chunk_node.single()['id']

                                await session.run(
                                    "MATCH (mc:MediumChunk), (sc:SmallChunk) WHERE id(mc) = $medium_chunk_id AND id(sc) = $small_chunk_id AND mc.projectID = $projectID AND sc.projectID = $projectID "
                                    "CREATE (mc)-[:HAS_CHUNK]->(sc), (sc)-[:PART_OF]->(mc)",
                                    medium_chunk_id=medium_chunk_id, small_chunk_id=small_chunk_id, projectID=projectID
                                )

                                embedding = self.model.embed_text(small_chunk)
                                embedding_node = await session.run(
                                    "CREATE (n:Embedding {embedding: $embedding, type: 'embedding', projectID: $projectID}) RETURN id(n)",
                                    embedding=embedding, projectID=projectID
                                )
                                embedding_id = embedding_node.single()['id']

                                await session.run(
                                    "MATCH (sc:SmallChunk), (e:Embedding) WHERE id(sc) = $small_chunk_id AND id(e) = $embedding_id AND sc.projectID = $projectID AND e.projectID = $projectID "
                                    "CREATE (sc)-[:HAS_EMBEDDING]->(e)",
                                    small_chunk_id=small_chunk_id, embedding_id=embedding_id, projectID=projectID
                                )
        except Exception as e:
            logger.error(f"Error ingesting PDF file {inputPath}: {e}")
            raise FileProcessingError(f"Error ingesting PDF file {inputPath}: {e}")

class NIngest:
    """
    Manages the ingestion process, counting files, handling directories, and updating or deleting projects.
    """
    def __init__(self, projectID: str = None, importer: NBaseImporter = NNeo4JImporter()):
        self.total_files = 0
        self.progress_bar = None

        self.projectID = projectID or str(uuid.uuid4())
        self.currentOutputPath = os.path.expanduser(f"~/.ngest/projects/{self.projectID}")
        self.importer_ = importer

        if not projectID:
            os.makedirs(self.currentOutputPath, exist_ok=True)
            open(os.path.join(self.currentOutputPath, '.ngest_index'), 'a').close()
            logger.info(f"Created new project directory at {self.currentOutputPath}")
            
    async def start_ingestion(self, inputPath: str) -> int:
        """
        Start the ingestion process.

        Args:
            inputPath (str): The path to the input file or directory.

        Returns:
            int: The result of the ingestion process.
        """
        if not validate_input_path(inputPath):
            return -1

        self.total_files = await self.count_files(inputPath)
        self.progress_bar = tqdm(total=self.total_files, desc="Ingesting files", unit="file")
        try:
            result = await self.Ingest(inputPath, self.currentOutputPath)
            self.progress_bar.close()
            return result
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            if self.progress_bar:
                self.progress_bar.close()
            return -1

    async def count_files(self, path: str) -> int:
        """
        Count the total number of files in a directory.

        Args:
            path (str): The path to the directory.

        Returns:
            int: The total number of files.
        """
        total = 0
        for entry in os.scandir(path):
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
                # Remove all nodes and relationships related to this project
                await session.run(
                    "MATCH (n) WHERE n.projectID = $projectID "
                    "DETACH DELETE n",
                    projectID=projectID
                )
                logger.info(f"Cleaned up partial ingestion for project {projectID}")
            except Exception as e:
                logger.error(f"Error during cleanup for project {projectID}: {e}")
                
    async def Ingest(self, inputPath: str, currentOutputPath: str) -> int:
        """
        Ingest files from the input path into the output path.

        Args:
            inputPath (str): The path to the input file or directory.
            currentOutputPath (str): The current output path.

        Returns:
            int: The result of the ingestion process.
        """
        try:
            if not os.path.exists(inputPath):
                logger.error(f"Invalid path: {inputPath}")
                return -1

            inputType = 'd' if os.path.isdir(inputPath) else 'f'
            with open(os.path.join(currentOutputPath, '.ngest_index'), 'a') as index_file:
                index_file.write(f"{inputType},{inputPath}\n")
            logger.info(f"INGESTING: {inputPath}")

            inputLocation, inputName = os.path.split(inputPath)
            if inputType == 'd':
                await self.IngestDirectory(inputPath, inputLocation, inputName, currentOutputPath, self.projectID)
            else:
                if not self.should_ignore_file(inputPath):
                    await self.IngestFile(inputPath, inputLocation, inputName, currentOutputPath, self.projectID)
                    if self.progress_bar:
                        self.progress_bar.update(1)
            return 0
        
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            await self.cleanup_partial_ingestion(self.projectID)
            return -1
            
    async def IngestDirectory(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        """
        Ingest all files in a directory.

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
                tasks.append(self.Ingest(itemPath, newOutputPath))
        await asyncio.gather(*tasks)

    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        """
        Ingest a single file.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            projectID (str): The project ID.
        """
        await self.importer_.IngestFile(inputPath, inputLocation, inputName, currentOutputPath, projectID)

    def should_ignore_file(self, file_path: str) -> bool:
        """
        Determine if a file should be ignored based on .gitignore patterns.

        Args:
            file_path (str): The path to the file.

        Returns:
            bool: True if the file should be ignored, False otherwise.
        """
        gitignore_patterns = self.load_gitignore_patterns(os.path.dirname(file_path))
        return any(fnmatch.fnmatch(os.path.basename(file_path), pattern) for pattern in gitignore_patterns)

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
                result = await session.run(
                    "MATCH (n) WHERE n.projectID = $projectID "
                    "RETURN n",
                    projectID=projectID
                )
                
                nodes = await result.data()
                
                with open(outputPath, 'w') as f:
                    json.dump(nodes, f)
                
            logger.info(f"Project {projectID} exported successfully to {outputPath}")
            return 0
        except Exception as e:
            logger.error(f"Error exporting project {projectID}: {e}")
            return -1

def validate_input_path(input_path: str) -> bool:
    """
    Validate the input path.

    Args:
        input_path (str): The path to validate.

    Returns:
        bool: True if the path is valid, False otherwise.
    """
    if not os.path.exists(input_path):
        logger.error(f"Input path does not exist: {input_path}")
        return False
    if os.path.isfile(input_path) and os.path.getsize(input_path) > MAX_FILE_SIZE:
        logger.error(f"File {input_path} exceeds the maximum allowed size of {MAX_FILE_SIZE} bytes.")
        return False
    return True

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
    """
    Provides a high-level interface for managing projects, including creating, updating, deleting, and exporting projects.
    """
    def __init__(self):
        self.ingest = NIngest()

    async def create_project(self, input_path: str) -> str:
        """
        Create a new project.

        Args:
            input_path (str): The path to the input files.

        Returns:
            str: The project ID.
        """
        project_id = str(uuid.uuid4())
        ingest_instance = NIngest(projectID=project_id)
        result = await ingest_instance.start_ingestion(input_path)
        if result == 0:
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
        ingest_instance = NIngest(projectID=project_id)
        return await ingest_instance.delete_project(project_id)

    async def export_project(self, project_id: str, output_path: str) -> int:
        """
        Export a project to a JSON file.

        Args:
            project_id (str): The project ID.
            output_path (str): The path to the output file.

        Returns:
            int: The result of the export process.
        """
        ingest_instance = NIngest(projectID=project_id)
        return await ingest_instance.export_project(project_id, output_path)

async def main():
    """
    Main function to handle command-line arguments and execute the appropriate action.
    """
    parser = argparse.ArgumentParser(description='Ingest files into Neo4j database')
    parser.add_argument('action', choices=['create', 'update', 'delete', 'export'], help='Action to perform')
    parser.add_argument('--input_path', type=str, help='Path to file or directory to ingest')
    parser.add_argument('--project_id', type=str, help='Project ID for update, delete, or export actions')
    parser.add_argument('--output_path', type=str, help='Output path for export action')
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
            if not args.project_id or not args.output_path:
                raise ValueError("Both project_id and output_path are required for export action")
            result = await project_manager.export_project(args.project_id, args.output_path)
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

    def test_ingest_file(self):
        """
        Test ingesting a single file.
        """
        test_file = os.path.join(self.test_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        async def run_test():
            result = await self.ningest.Ingest(test_file, self.test_dir)
            self.assertEqual(result, 0)

        asyncio.run(run_test())

    def test_ingest_directory(self):
        """
        Test ingesting a directory.
        """
        test_subdir = os.path.join(self.test_dir, 'subdir')
        os.makedirs(test_subdir)
        test_file = os.path.join(test_subdir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        async def run_test():
            result = await self.ningest.Ingest(self.test_dir, self.test_dir)
            self.assertEqual(result, 0)

        asyncio.run(run_test())

    def test_gitignore(self):
        """
        Test ignoring files based on .gitignore patterns.
        """
        with open(os.path.join(self.test_dir, '.gitignore'), 'w') as f:
            f.write("*.log\n")
        
        test_file = os.path.join(self.test_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        ignored_file = os.path.join(self.test_dir, 'ignored.log')
        with open(ignored_file, 'w') as f:
            f.write("Ignored content")
        
        async def run_test():
            result = await self.ningest.Ingest(self.test_dir, self.test_dir)
            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(os.path.join(self.ningest.currentOutputPath, 'test.txt')))
            self.assertFalse(os.path.exists(os.path.join(self.ningest.currentOutputPath, 'ignored.log')))

        asyncio.run(run_test())

if __name__ == "__main__":
    asyncio.run(main())
