# Copyright 2024 Chris Odom
# MIT License

import os
import uuid
import logging
from abc import ABC, abstractmethod
import asyncio
import ollama
from neo4j import AsyncGraphDatabase
import PIL.Image
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import pipeline, AutoTokenizer, AutoModel
import clang.cindex
import ast
import syn
from PyPDF2 import PdfReader
from typing import List, Dict, Any, Union
import esprima
from esprima import nodes
import argparse
import configparser
from contextlib import asynccontextmanager
import torch
from tqdm import tqdm

# Load configuration
config = configparser.ConfigParser()
config_file = 'config.ini'

if not os.path.isfile(config_file):
    raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

config.read(config_file)

# Constants from config
MAX_FILE_SIZE = config.getint('Limits', 'MAX_FILE_SIZE', fallback=30 * 1024 * 1024)
MEDIUM_CHUNK_SIZE = config.getint('Chunks', 'MEDIUM_CHUNK_SIZE', fallback=10000)
SMALL_CHUNK_SIZE = config.getint('Chunks', 'SMALL_CHUNK_SIZE', fallback=1000)
NEO4J_URL = config.get('Database', 'NEO4J_URL', fallback="bolt://localhost:7687")
OLLAMA_URL = config.get('Ollama', 'OLLAMA_URL', fallback="http://localhost:11434")
EMBEDDING_MODEL = config.get('Models', 'EMBEDDING_MODEL', fallback="nomic-embed-text")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NBaseImporter(ABC):
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

    async def create_graph_nodes(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        logger.info(f"Creating graph nodes for {inputPath}")
        # Implement the actual graph node creation logic here

    async def chunk_and_create_nodes(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        async with self.driver.session() as session:
            try:
                with open(inputPath, 'r') as file:
                    text = file.read()
            except Exception as e:
                logger.error(f"Error reading file {inputPath}: {e}")
                return

            chunks = self.chunk_text(text, MEDIUM_CHUNK_SIZE)
            previous_chunk_id = None

            for i, chunk in enumerate(chunks):
                chunk_id = await self.create_chunk_node(session, chunk, inputPath, i, projectID)
                if previous_chunk_id:
                    await self.create_chunk_relationship(session, previous_chunk_id, chunk_id, projectID)
                previous_chunk_id = chunk_id

    def chunk_text(self, text: str, chunk_size: int) -> List[str]:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    async def create_chunk_node(self, session, chunk: str, inputPath: str, index: int, projectID: str) -> str:
        embedding = self.model.embed_text(chunk)
        chunk_id = f"{inputPath}_chunk_{index}"
        try:
            result = await session.run(
                "CREATE (n:Chunk {id: $id, content: $content, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                id=chunk_id, content=chunk, embedding=embedding, projectID=projectID
            )
            node_id = result.single()['id']
            logger.info(f"Created chunk node {chunk_id} with embedding")
            return chunk_id
        except Exception as e:
            logger.error(f"Error creating chunk node: {e}")
            return None

    async def create_chunk_relationship(self, session, chunk_id1: str, chunk_id2: str, projectID: str) -> None:
        try:
            await session.run(
                "MATCH (c1:Chunk {id: $id1, projectID: $projectID}), (c2:Chunk {id: $id2, projectID: $projectID}) CREATE (c1)-[:NEXT]->(c2)",
                id1=chunk_id1, id2=chunk_id2, projectID=projectID
            )
            logger.info(f"Created relationship between {chunk_id1} and {chunk_id2}")
        except Exception as e:
            logger.error(f"Error creating chunk relationship: {e}")

class NFilesystemImporter(NBaseImporter):
    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        if os.path.getsize(inputPath) > MAX_FILE_SIZE:
            logger.info(f"File {inputPath} skipped due to size.")
            open(os.path.join(currentOutputPath, f"{inputName}.skipped"), 'a').close()
            return

        import shutil
        try:
            shutil.copy(inputPath, os.path.join(currentOutputPath, inputName))
            logger.info(f"File {inputPath} ingested successfully.")
        except Exception as e:
            logger.error(f"Error ingesting file {inputPath}: {e}")

class NNeo4JImporter(NBaseImporter):
    def __init__(self, neo4j_url: str = NEO4J_URL):
        self.neo4j_url = neo4j_url
        self.driver = AsyncGraphDatabase.driver(self.neo4j_url)
        self.index = clang.cindex.Index.create()
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=self.device)
        
        self.image_model = models.densenet121(pretrained=True).to(self.device)
        self.image_model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.code_model = AutoModel.from_pretrained("microsoft/codebert-base").to(self.device)
        self.code_model.eval()
        
        self.model = ollama.Model(EMBEDDING_MODEL)  # Reuse the model instance
        
        logger.info(f"Neo4J importer initialized with URL {neo4j_url}")
        
    @asynccontextmanager
    async def get_session(self):
        async with self.driver.session() as session:
            try:
                yield session
            finally:
                await session.close()

    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        file_type = self.ascertain_file_type(inputPath)
        ingest_method = getattr(self, f"Ingest{file_type.capitalize()}", None)
        if ingest_method:
            await ingest_method(inputPath, inputLocation, inputName, currentOutputPath, projectID)
        else:
            logger.warning(f"No ingest method for file type: {file_type}")

    async def summarize_text(self, text: str) -> str:
        return self.summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
    
    async def IngestTxt(self, input_path: str, input_location: str, input_name: str, current_output_path: str, project_id: str) -> None:
        async with self.get_session() as session:
            try:
                with open(input_path, 'r') as file:
                    file_content = file.read()
            except Exception as e:
                logger.error(f"Error reading file {input_path}: {e}")
                return

            try:
                parent_doc_node = await session.run(
                    "CREATE (n:Document {name: $name, type: 'text', projectID: $projectID}) RETURN id(n)",
                    name=input_name, projectID=project_id
                )
                parent_doc_id = parent_doc_node.single()['id']

                medium_chunks = self.chunk_text(file_content, MEDIUM_CHUNK_SIZE)
                for i, medium_chunk in enumerate(medium_chunks):
                    medium_chunk_node = await session.run(
                        "CREATE (n:MediumChunk {content: $content, type: 'medium_chunk', projectID: $projectID}) RETURN id(n)",
                        content=medium_chunk, projectID=project_id
                    )
                    medium_chunk_id = medium_chunk_node.single()['id']

                    await session.run(
                        "MATCH (d:Document), (c:MediumChunk) WHERE id(d) = $doc_id AND id(c) = $chunk_id AND d.projectID = $projectID AND c.projectID = $projectID "
                        "CREATE (d)-[:HAS_CHUNK]->(c), (c)-[:PART_OF]->(d)",
                        doc_id=parent_doc_id, chunk_id=medium_chunk_id, projectID=project_id
                    )

                    small_chunks = self.chunk_text(medium_chunk, SMALL_CHUNK_SIZE)
                    for j, small_chunk in enumerate(small_chunks):
                        small_chunk_node = await session.run(
                            "CREATE (n:SmallChunk {content: $content, type: 'small_chunk', projectID: $projectID}) RETURN id(n)",
                            content=small_chunk, projectID=project_id
                        )
                        small_chunk_id = small_chunk_node.single()['id']

                        await session.run(
                            "MATCH (mc:MediumChunk), (sc:SmallChunk) WHERE id(mc) = $medium_chunk_id AND id(sc) = $small_chunk_id AND mc.projectID = $projectID AND sc.projectID = $projectID "
                            "CREATE (mc)-[:HAS_CHUNK]->(sc), (sc)-[:PART_OF]->(mc)",
                            medium_chunk_id=medium_chunk_id, small_chunk_id=small_chunk_id, projectID=project_id
                        )

                        embedding = self.model.embed_text(small_chunk)
                        embedding_node = await session.run(
                            "CREATE (n:Embedding {embedding: $embedding, type: 'embedding', projectID: $projectID}) RETURN id(n)",
                            embedding=embedding, projectID=project_id
                        )
                        embedding_id = embedding_node.single()['id']

                        await session.run(
                            "MATCH (sc:SmallChunk), (e:Embedding) WHERE id(sc) = $small_chunk_id AND id(e) = $embedding_id AND sc.projectID = $projectID AND e.projectID = $projectID "
                            "CREATE (sc)-[:HAS_EMBEDDING]->(e)",
                            small_chunk_id=small_chunk_id, embedding_id=embedding_id, projectID=project_id
                        )
            except Exception as e:
                logger.error(f"Error ingesting text file {input_path}: {e}")

    async def extract_image_features(self, image: PIL.Image.Image) -> List[float]:
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
            return []

    async def IngestImg(self, input_path: str, input_location: str, input_name: str, current_output_path: str, project_id: str) -> None:
        try:
            image = PIL.Image.open(input_path)
            features = await self.extract_image_features(image)

            async with self.get_session() as session:
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
            logger.error(f"Error ingesting image: {e}")

    async def summarize_cpp_class(self, cls) -> str:
        description = f"Class {cls.spelling} with public methods: "
        public_methods = [c.spelling for c in cls.get_children() if c.access_specifier == clang.cindex.AccessSpecifier.PUBLIC]
        description += ", ".join(public_methods)
        return await self.summarize_text(description)

    async def summarize_cpp_function(self, func) -> str:
        description = f"Function {func.spelling} in namespace {func.semantic_parent.spelling}. It performs the following tasks: "
        return await self.summarize_text(description)

    def get_code_features(self, code: str) -> List[float]:
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.code_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
        
    async def IngestCpp(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
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
                    class_node = await session.run(
                        "CREATE (n:Class {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                        name=cls.spelling, summary=class_summary, embedding=embedding, projectID=projectID
                    )
                    class_id = class_node.single()['id']
                    
                    for func in cls.get_children():
                        if func.kind == clang.cindex.CursorKind.CXX_METHOD:
                            function_summary = await self.summarize_cpp_function(func)
                            embedding = self.model.embed_text(function_summary)
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
                    await session.run(
                        "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID})",
                        name=func.spelling, summary=function_summary, embedding=embedding, projectID=projectID
                    )
        except Exception as e:
            logger.error(f"Error ingesting C++ file {inputPath}: {e}")

    async def IngestPython(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        try:
            with open(inputPath, 'r') as file:
                tree = ast.parse(file.read(), filename=inputPath)

            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

            async with self.get_session() as session:
                for cls in classes:
                    class_summary = await self.summarize_python_class(cls)
                    embedding = self.model.embed_text(class_summary)
                    class_node = await session.run(
                        "CREATE (n:Class {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                        name=cls.name, summary=class_summary, embedding=embedding, projectID=projectID
                    )
                    class_id = class_node.single()['id']
                    
                    for func in cls.body:
                        if isinstance(func, ast.FunctionDef):
                            function_summary = await self.summarize_python_function(func)
                            embedding = self.model.embed_text(function_summary)
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
                    await session.run(
                        "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID})",
                        name=func.name, summary=function_summary, embedding=embedding, projectID=projectID
                    )
        except Exception as e:
            logger.error(f"Error ingesting Python file {inputPath}: {e}")

    async def summarize_python_class(self, cls) -> str:
        description = f"Class {cls.name} with methods: "
        methods = [func.name for func in cls.body if isinstance(func, ast.FunctionDef)]
        description += ", ".join(methods)
        return await self.summarize_text(description)

    async def summarize_python_function(self, func) -> str:
        description = f"Function {func.name} with arguments: {', '.join(arg.arg for arg in func.args.args)}. It performs the following tasks: "
        return await self.summarize_text(description)
    
    async def IngestRust(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        try:
            tree = syn.parse_file(inputPath)
            functions = [item for item in tree.items if isinstance(item, syn.ItemFn)]
            impls = [item for item in tree.items if isinstance(item, syn.ItemImpl)]

            async with self.get_session() as session:
                for impl in impls:
                    impl_summary = await self.summarize_rust_impl(impl)
                    embedding = self.model.embed_text(impl_summary)
                    impl_node = await session.run(
                        "CREATE (n:Impl {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID}) RETURN id(n)",
                        name=impl.trait_.path.segments[0].ident, summary=impl_summary, embedding=embedding, projectID=projectID
                    )
                    impl_id = impl_node.single()['id']
                    
                    for item in impl.items:
                        if isinstance(item, syn.ImplItemMethod):
                            function_summary = await self.summarize_rust_function(item)
                            embedding = self.model.embed_text(function_summary)
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
                    await session.run(
                        "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding, projectID: $projectID})",
                        name=func.sig.ident, summary=function_summary, embedding=embedding, projectID=projectID
                    )
        except Exception as e:
            logger.error(f"Error ingesting Rust file {inputPath}: {e}")

    async def summarize_rust_impl(self, impl) -> str:
        description = f"Implementation of trait {impl.trait_.path.segments[0].ident} with methods: "
        methods = [item.sig.ident for item in impl.items if isinstance(item, syn.ImplItemMethod)]
        description += ", ".join([str(m) for m in methods])
        return await self.summarize_text(description)

    async def summarize_rust_function(self, func) -> str:
        description = f"Function {func.sig.ident} with arguments: {', '.join(arg.pat.ident for arg in func.sig.inputs)}. It performs the following tasks: "
        return await self.summarize_text(description)

    async def IngestJavascript(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        try:
            with open(inputPath, 'r') as file:
                content = file.read()

            ast = esprima.parseModule(content, {'loc': True, 'range': True})
        except Exception as e:
            logger.error(f"Error parsing JavaScript file {inputPath}: {e}")
            return

        async with self.get_session() as session:
            try:
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

    async def process_js_function(self, session, func_node, file_id: int, projectID: str) -> None:
        func_name = func_node.id.name if func_node.id else 'anonymous'
        func_summary = await self.summarize_js_function(func_node)
        embedding = self.model.embed_text(func_summary)

        try:
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

    async def process_js_class(self, session, class_node, file_id: int, projectID: str) -> None:
        class_name = class_node.id.name
        class_summary = await self.summarize_js_class(class_node)
        embedding = self.model.embed_text(class_summary)

        try:
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

    async def process_js_variable(self, session, var_node, file_id: int, projectID: str) -> None:
        for declaration in var_node.declarations:
            var_name = declaration.id.name
            var_type = var_node.kind  # 'var', 'let', or 'const'
            var_summary = await self.summarize_js_variable(var_node)
            embedding = self.model.embed_text(var_summary)

            try:
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
                
    async def IngestPdf(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        try:
            reader = PdfReader(inputPath)
            num_pages = len(reader.pages)

            async with self.get_session() as session:
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

    async def process_js_method(self, session, method_node, class_id: int, projectID: str) -> None:
        method_name = method_node.key.name
        method_summary = await self.summarize_js_function(method_node.value)
        embedding = self.model.embed_text(method_summary)

        try:
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

class NIngest:
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
        self.total_files = await self.count_files(inputPath)
        self.progress_bar = tqdm(total=self.total_files, desc="Ingesting files")
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
        total = 0
        for entry in os.scandir(path):
            if entry.is_file():
                total += 1
            elif entry.is_dir():
                total += await self.count_files(entry.path)
        return total
    
    async def cleanup_partial_ingestion(self, projectID: str):
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
                await self.IngestFile(inputPath, inputLocation, inputName, currentOutputPath, self.projectID)
                if self.progress_bar:
                    self.progress_bar.update(1)
            return 0
        
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            await self.cleanup_partial_ingestion(self.projectID)
            return -1
            
    async def IngestDirectory(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        newOutputPath = os.path.join(currentOutputPath, inputName)
        os.makedirs(newOutputPath, exist_ok=True)

        tasks = []
        for item in os.listdir(inputPath):
            itemPath = os.path.join(inputPath, item)
            tasks.append(self.Ingest(itemPath, newOutputPath))
        await asyncio.gather(*tasks)

    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        await self.importer_.IngestFile(inputPath, inputLocation, inputName, currentOutputPath, projectID)

async def main():
    parser = argparse.ArgumentParser(description='Ingest files into Neo4j database')
    parser.add_argument('input_path', type=str, help='Path to file or directory to ingest')
    parser.add_argument('--project_id', type=str, help='Project ID (optional)')
    args = parser.parse_args()

    ingest_instance = NIngest(projectID=args.project_id, importer=NNeo4JImporter())
    result = await ingest_instance.start_ingestion(args.input_path)
    print(f"Ingestion {'succeeded' if result == 0 else 'failed'}")

if __name__ == "__main__":
    asyncio.run(main())
