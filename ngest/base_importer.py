# Copyright 2024 Chris Odom
# MIT License

import os
import datetime
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import logging
from abc import ABC, abstractmethod
import asyncio
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable
import PIL.Image
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import pipeline, AutoTokenizer, AutoModel, BertModel, BertTokenizer
#from clang.cindex import Config
#Config.set_library_path("/opt/homebrew/opt/llvm/lib")
#import clang.cindex
import ast
import syn
from PyPDF2 import PdfReader
from typing import List, Dict, Any, Union, Optional, Generator
import esprima
from esprima import nodes
import argparse
import configparser
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
from typing import Optional, List, Dict, Any, Union, Tuple
import magic
from collections import defaultdict
import copy
import pprint

logger = logging.getLogger(__name__)

class NBaseImporter(ABC):
    """
    Abstract base class for file importers. Provides methods to determine file types
    and handle file chunking and graph node creation.
    """
    @abstractmethod
    async def ParseFile(self, inputPath: str, inputLocation: str, inputName: str, topLevelInputPath: str, topLevelOutputPath: str, currentOutputPath: str, project_id: str) -> None:
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
            elif ext == '.txt':
                type_name = 'text'
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
    async def create_graph_nodes(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, project_id: str) -> None:
        """
        Create graph nodes in the Neo4j database for a given file.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            project_id (str): The project ID.
        """
        logger.info(f"Creating graph nodes for {inputPath}")
        # Implement the actual graph node creation logic here

    async def chunk_and_create_nodes(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, project_id: str) -> None:
        try:
            chunks = [chunk async for chunk in read_file_in_chunks(inputPath, MEDIUM_CHUNK_SIZE)]
            previous_chunk_id = None

            for i, chunk in enumerate(chunks):
                chunk_id = await self.handle_chunk_creation(chunk, inputPath, i, project_id)
                if previous_chunk_id:
                    await self.create_chunk_relationship(previous_chunk_id, chunk_id, project_id)
                previous_chunk_id = chunk_id
        except Exception as e:
            logger.error(f"Error processing file {inputPath}: {e}")
            raise FileProcessingError(f"Error processing file {inputPath}: {e}")


    def chunk_text(self, text: str, chunk_size: int) -> List[str]:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]



#    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=15, max=30))
    async def create_chunk_node(self, chunk: str, inputPath: str, index: int, project_id: str) -> Optional[str]:
        """
        Create a chunk node in the Neo4j database.

        Args:
            chunk (str): The text chunk.
            inputPath (str): The path to the input file.
            index (int): The index of the chunk.
            project_id (str): The project ID.

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
                        "CREATE (n:Chunk {elementId: $id, content: $content, embedding: $embedding, project_id: $project_id}) RETURN elementId(n)",
                        id=chunk_id, content=chunk, embedding=embedding, project_id=project_id
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
    async def create_chunk_relationship(self, chunk_id1: str, chunk_id2: str, project_id: str) -> None:
        """
        Create a relationship between two chunks in the Neo4j database.

        Args:
            chunk_id1 (str): The ID of the first chunk.
            chunk_id2 (str): The ID of the second chunk.
            project_id (str): The project ID.
        """
        try:
            async with self.rate_limiter_db:
                async with self.get_session() as session:
                    result = await (await session.run(
                        "MATCH (c1:Chunk {elementId: $id1, project_id: $project_id}), (c2:Chunk {elementId: $id2, project_id: $project_id}) CREATE (c1)-[:NEXT]->(c2)",
                        id1=chunk_id1, id2=chunk_id2, project_id=project_id
                    )).consume()
                    if result:
                        logger.info(f"Created relationship between {chunk_id1} and {chunk_id2}")
                    else:
                        logger.error(f"No result returned for creating relationship between {chunk_id1} and {chunk_id2}")
        except Exception as e:
            logger.error(f"Error creating chunk relationship: {e}")
            raise DatabaseError(f"Error creating chunk relationship: {e}")


    async def handle_chunk_creation(self, chunk, inputPath, index, project_id):
        """
        Handle the creation of a chunk node.

        Args:
            chunk: The text chunk.
            inputPath: The path to the input file.
            index: The index of the chunk.
            project_id: The project ID.

        Returns:
            The chunk ID.
        """
        try:
            chunk_id = await self.create_chunk_node(chunk, inputPath, index, project_id)
            return chunk_id
        except Exception as e:
            logger.error(f"Error creating chunk node for chunk {index} in file {inputPath}: {e}")
            raise DatabaseError(f"Error creating chunk node for chunk {index} in file {inputPath}: {e}")

class NFilesystemImporter(NBaseImporter):
    """
    A file importer that simply copies files to the output directory.
    """
    async def ParseFile(self, inputPath: str, inputLocation: str, inputName: str, topLevelInputPath: str, topLevelOutputPath: str, currentOutputPath: str, project_id: str) -> None:
        """
        Ingest a file by copying it to the output directory.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            currentOutputPath (str): The current output path.
            project_id (str): The project ID.
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
