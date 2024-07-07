# Copyright 2024 Chris Odom
# MIT License

import os
import uuid
import logging
from abc import ABC, abstractmethod
import asyncio
import ollama
from neo4j import GraphDatabase, AsyncGraphDatabase
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 30 * 1024 * 1024  # 30 MB
MEDIUM_CHUNK_SIZE = 10000
SMALL_CHUNK_SIZE = 1000
NEO4J_URL = "bolt://localhost:7687"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"

class NBaseImporter(ABC):
    @abstractmethod
    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        pass

    def ascertain_file_type(self, filename: str) -> str:
        import magic
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(filename)
        
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

    async def chunk_and_create_nodes(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        async with self.driver.session() as session:
            with open(inputPath, 'r') as file:
                text = file.read()

            chunks = self.chunk_text(text, MEDIUM_CHUNK_SIZE)
            previous_chunk_id = None

            for i, chunk in enumerate(chunks):
                chunk_id = await self.create_chunk_node(session, chunk, inputPath, i, projectID)
                if previous_chunk_id:
                    await self.create_chunk_relationship(session, previous_chunk_id, chunk_id)
                previous_chunk_id = chunk_id

    def chunk_text(self, text: str, chunk_size: int) -> List[str]:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    async def create_chunk_node(self, session, chunk: str, inputPath: str, index: int, projectID: str) -> str:
        model = ollama.Model(EMBEDDING_MODEL)
        embedding = model.embed_text(chunk)
        chunk_id = f"{inputPath}_chunk_{index}"
        await session.run(
            "CREATE (n:Chunk {id: $id, content: $content, embedding: $embedding}) RETURN id(n)",
            id=chunk_id, content=chunk, embedding=embedding
        )
        logger.info(f"Created chunk node {chunk_id} with embedding")
        return chunk_id

    async def create_chunk_relationship(self, session, chunk_id1: str, chunk_id2: str) -> None:
        await session.run(
            "MATCH (c1:Chunk {id: $id1}), (c2:Chunk {id: $id2}) CREATE (c1)-[:NEXT]->(c2)",
            id1=chunk_id1, id2=chunk_id2
        )
        logger.info(f"Created relationship between {chunk_id1} and {chunk_id2}")

class NFilesystemImporter(NBaseImporter):
    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        if os.path.getsize(inputPath) > MAX_FILE_SIZE:
            logger.info(f"File {inputPath} skipped due to size.")
            open(os.path.join(currentOutputPath, f"{inputName}.skipped"), 'a').close()
            return

        import shutil
        shutil.copy(inputPath, os.path.join(currentOutputPath, inputName))
        logger.info(f"File {inputPath} ingested successfully.")

class NNeo4JImporter(NBaseImporter):
    def __init__(self, neo4j_url: str = NEO4J_URL):
        self.neo4j_url = neo4j_url
        self.driver = AsyncGraphDatabase.driver(self.neo4j_url)
        self.index = clang.cindex.Index.create()
        self.summarizer = pipeline("summarization")
        self.image_model = models.densenet121(pretrained=True)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.code_model = AutoModel.from_pretrained("microsoft/codebert-base")
        logger.info(f"Neo4J importer initialized with URL {neo4j_url}")

    async def IngestFile(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        file_type = self.ascertain_file_type(inputPath)
        ingest_method = getattr(self, f"Ingest{file_type.capitalize()}", None)
        if ingest_method:
            await ingest_method(inputPath, inputLocation, inputName, currentOutputPath, projectID)
        else:
            logger.warning(f"No ingest method for file type: {file_type}")

    async def IngestTxt(self, input_path: str, input_location: str, input_name: str, current_output_path: str, project_id: str) -> None:
        async with self.driver.session() as session:
            with open(input_path, 'r') as file:
                file_content = file.read()
            
            parent_doc_node = await session.run(
                "CREATE (n:Document {name: $name, type: 'text'}) RETURN id(n)",
                name=input_name
            )
            parent_doc_id = parent_doc_node.single()['id']

            medium_chunks = self.chunk_text(file_content, MEDIUM_CHUNK_SIZE)
            for i, medium_chunk in enumerate(medium_chunks):
                medium_chunk_node = await session.run(
                    "CREATE (n:MediumChunk {content: $content, type: 'medium_chunk'}) RETURN id(n)",
                    content=medium_chunk
                )
                medium_chunk_id = medium_chunk_node.single()['id']

                await session.run(
                    "MATCH (d:Document), (c:MediumChunk) WHERE id(d) = $doc_id AND id(c) = $chunk_id "
                    "CREATE (d)-[:HAS_CHUNK]->(c), (c)-[:PART_OF]->(d)",
                    doc_id=parent_doc_id, chunk_id=medium_chunk_id
                )

                small_chunks = self.chunk_text(medium_chunk, SMALL_CHUNK_SIZE)
                for j, small_chunk in enumerate(small_chunks):
                    small_chunk_node = await session.run(
                        "CREATE (n:SmallChunk {content: $content, type: 'small_chunk'}) RETURN id(n)",
                        content=small_chunk
                    )
                    small_chunk_id = small_chunk_node.single()['id']

                    await session.run(
                        "MATCH (mc:MediumChunk), (sc:SmallChunk) WHERE id(mc) = $medium_chunk_id AND id(sc) = $small_chunk_id "
                        "CREATE (mc)-[:HAS_CHUNK]->(sc), (sc)-[:PART_OF]->(mc)",
                        medium_chunk_id=medium_chunk_id, small_chunk_id=small_chunk_id
                    )

                    embedding = self.generate_embedding(small_chunk)
                    embedding_node = await session.run(
                        "CREATE (n:Embedding {embedding: $embedding, type: 'embedding'}) RETURN id(n)",
                        embedding=embedding
                    )
                    embedding_id = embedding_node.single()['id']

                    await session.run(
                        "MATCH (sc:SmallChunk), (e:Embedding) WHERE id(sc) = $small_chunk_id AND id(e) = $embedding_id "
                        "CREATE (sc)-[:HAS_EMBEDDING]->(e)",
                        small_chunk_id=small_chunk_id, embedding_id=embedding_id
                    )

    def generate_embedding(self, text_chunk: str) -> List[float]:
        model = ollama.Model(EMBEDDING_MODEL)
        embedding = model.embed_text(text_chunk)
        return embedding

    async def extract_image_features(self, image: PIL.Image.Image) -> List[float]:
        try:
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor = preprocess(image).unsqueeze(0)
            features = self.image_model(tensor)
            return features.squeeze().tolist()
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return []

    async def IngestImg(self, input_path: str, input_location: str, input_name: str, current_output_path: str, project_id: str) -> None:
        try:
            image = PIL.Image.open(input_path)
            features = await self.extract_image_features(image)

            async with self.driver.session() as session:
                image_node = await session.run(
                    "CREATE (n:Image {name: $name, type: 'image'}) RETURN id(n)",
                    name=input_name
                )
                image_id = image_node.single()['id']

                features_node = await session.run(
                    "CREATE (n:ImageFeatures {features: $features, type: 'image_features'}) RETURN id(n)",
                    features=features
                )
                features_id = features_node.single()['id']

                await session.run(
                    "MATCH (i:Image), (f:ImageFeatures) WHERE id(i) = $image_id AND id(f) = $features_id "
                    "CREATE (i)-[:HAS_FEATURES]->(f)",
                    image_id=image_id, features_id=features_id
                )
        except Exception as e:
            logger.error(f"Error ingesting image: {e}")

    async def summarize_cpp_class(self, cls) -> str:
        description = f"Class {cls.spelling} with public methods: "
        public_methods = [c.spelling for c in cls.get_children() if c.access_specifier == clang.cindex.AccessSpecifier.PUBLIC]
        description += ", ".join(public_methods)
        summary = self.summarizer(description, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        return summary

    async def summarize_cpp_function(self, func) -> str:
        description = f"Function {func.spelling} in namespace {func.semantic_parent.spelling}. It performs the following tasks: "
        summary = self.summarizer(description, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        return summary

    async def IngestCpp(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        tu = self.index.parse(inputPath)
        functions = []
        classes = []

        for node in tu.cursor.get_children():
            if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                functions.append(node)
            elif node.kind in [clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL]:
                classes.append(node)

        async with self.driver.session() as session:
            for cls in classes:
                class_summary = await self.summarize_cpp_class(cls)
                embedding = self.generate_embedding(class_summary)
                class_node = await session.run(
                    "CREATE (n:Class {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                    name=cls.spelling, summary=class_summary, embedding=embedding
                )
                class_id = class_node.single()['id']
                
                for func in cls.get_children():
                    if func.kind == clang.cindex.CursorKind.CXX_METHOD:
                        function_summary = await self.summarize_cpp_function(func)
                        embedding = self.generate_embedding(function_summary)
                        function_node = await session.run(
                            "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                            name=func.spelling, summary=function_summary, embedding=embedding
                        )
                        function_id = function_node.single()['id']
                        await session.run(
                            "MATCH (c:Class), (f:Function) WHERE id(c) = $class_id AND id(f) = $function_id "
                            "CREATE (c)-[:HAS_METHOD]->(f)",
                            class_id=class_id, function_id=function_id
                        )

            for func in functions:
                function_summary = await self.summarize_cpp_function(func)
                embedding = self.generate_embedding(function_summary)
                await session.run(
                    "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding})",
                    name=func.spelling, summary=function_summary, embedding=embedding
                )

    async def IngestPython(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        with open(inputPath, 'r') as file:
            tree = ast.parse(file.read(), filename=inputPath)

        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        async with self.driver.session() as session:
            for cls in classes:
                class_summary = await self.summarize_python_class(cls)
                embedding = self.generate_embedding(class_summary)
                class_node = await session.run(
                    "CREATE (n:Class {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                    name=cls.name, summary=class_summary, embedding=embedding
                )
                class_id = class_node.single()['id']
                
                for func in cls.body:
                    if isinstance(func, ast.FunctionDef):
                        function_summary = await self.summarize_python_function(func)
                        embedding = self.generate_embedding(function_summary)
                        function_node = await session.run(
                            "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                            name=func.name, summary=function_summary, embedding=embedding
                        )
                        function_id = function_node.single()['id']
                        await session.run(
                            "MATCH (c:Class), (f:Function) WHERE id(c) = $class_id AND id(f) = $function_id "
                            "CREATE (c)-[:HAS_METHOD]->(f)",
                            class_id=class_id, function_id=function_id
                        )

            for func in functions:
                function_summary = await self.summarize_python_function(func)
                embedding = self.generate_embedding(function_summary)
                await session.run(
                    "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding})",
                    name=func.name, summary=function_summary, embedding=embedding
                )

    async def summarize_python_class(self, cls) -> str:
        description = f"Class {cls.name} with methods: "
        methods = [func.name for func in cls.body if isinstance(func, ast.FunctionDef)]
        description += ", ".join(methods)
        summary = self.summarizer(description, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        return summary

    async def summarize_python_function(self, func) -> str:
        description = f"Function {func.name} with arguments: {', '.join(arg.arg for arg in func.args.args)}. It performs the following tasks: "
        summary = self.summarizer(description, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        return summary

    async def IngestRust(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        tree = syn.parse_file(inputPath)
        functions = [item for item in tree.items if isinstance(item, syn.ItemFn)]
        impls = [item for item in tree.items if isinstance(item, syn.ItemImpl)]

        async with self.driver.session() as session:
            for impl in impls:
                impl_summary = await self.summarize_rust_impl(impl)
                embedding = self.generate_embedding(impl_summary)
                impl_node = await session.run(
                    "CREATE (n:Impl {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                    name=impl.trait_.path.segments[0].ident, summary=impl_summary, embedding=embedding
                )
                impl_id = impl_node.single()['id']
                
                for item in impl.items:
                    if isinstance(item, syn.ImplItemMethod):
                        function_summary = await self.summarize_rust_function(item)
                        embedding = self.generate_embedding(function_summary)
                        function_node = await session.run(
                            "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                            name=item.sig.ident, summary=function_summary, embedding=embedding
                        )
                        function_id = function_node.single()['id']
                        await session.run(
                            "MATCH (i:Impl), (f:Function) WHERE id(i) = $impl_id AND id(f) = $function_id "
                            "CREATE (i)-[:HAS_METHOD]->(f)",
                            impl_id=impl_id, function_id=function_id
                        )

            for func in functions:
                function_summary = await self.summarize_rust_function(func)
                embedding = self.generate_embedding(function_summary)
                await session.run(
                    "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding})",
                    name=func.sig.ident, summary=function_summary, embedding=embedding
                )

    async def summarize_rust_impl(self, impl) -> str:
        description = f"Implementation of trait {impl.trait_.path.segments[0].ident} with methods: "
        methods = [item.sig.ident for item in impl.items if isinstance(item, syn.ImplItemMethod)]
        description += ", ".join([str(m) for m in methods])
        summary = self.summarizer(description, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        return summary

    async def summarize_rust_function(self, func) -> str:
        description = f"Function {func.sig.ident} with arguments: {', '.join(arg.pat.ident for arg in func.sig.inputs)}. It performs the following tasks: "
        summary = self.summarizer(description, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        return summary

    async def IngestJavascript(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        with open(inputPath, 'r') as file:
            content = file.read()

        try:
            ast = esprima.parseModule(content, {'loc': True, 'range': True})
        except Exception as e:
            logger.error(f"Error parsing JavaScript file {inputPath}: {e}")
            return

        async with self.driver.session() as session:
            file_node = await session.run(
                "CREATE (n:JavaScriptFile {name: $name, path: $path}) RETURN id(n)",
                name=inputName, path=inputPath
            )
            file_id = file_node.single()['id']

            for node in ast.body:
                if isinstance(node, nodes.FunctionDeclaration):
                    await self.process_js_function(session, node, file_id)
                elif isinstance(node, nodes.ClassDeclaration):
                    await self.process_js_class(session, node, file_id)
                elif isinstance(node, nodes.VariableDeclaration):
                    await self.process_js_variable(session, node, file_id)

    async def process_js_function(self, session, func_node, file_id: int) -> None:
        func_name = func_node.id.name if func_node.id else 'anonymous'
        params = [param.name for param in func_node.params]
        func_summary = f"Function {func_name} with parameters: {', '.join(params)}"
        embedding = self.generate_embedding(func_summary)

        func_db_node = await session.run(
            "CREATE (n:JavaScriptFunction {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
            name=func_name, summary=func_summary, embedding=embedding
        )
        func_db_id = func_db_node.single()['id']

        await session.run(
            "MATCH (f:JavaScriptFile), (func:JavaScriptFunction) WHERE id(f) = $file_id AND id(func) = $func_id "
            "CREATE (f)-[:CONTAINS]->(func)",
            file_id=file_id, func_id=func_db_id
        )

    async def process_js_class(self, session, class_node, file_id: int) -> None:
        class_name = class_node.id.name
        methods = [method.key.name for method in class_node.body.body if isinstance(method, nodes.MethodDefinition)]
        class_summary = f"Class {class_name} with methods: {', '.join(methods)}"
        embedding = self.generate_embedding(class_summary)

        class_db_node = await session.run(
            "CREATE (n:JavaScriptClass {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
            name=class_name, summary=class_summary, embedding=embedding
        )
        class_db_id = class_db_node.single()['id']

        await session.run(
            "MATCH (f:JavaScriptFile), (c:JavaScriptClass) WHERE id(f) = $file_id AND id(c) = $class_id "
            "CREATE (f)-[:CONTAINS]->(c)",
            file_id=file_id, class_id=class_db_id
        )

        for method in class_node.body.body:
            if isinstance(method, nodes.MethodDefinition):
                await self.process_js_method(session, method, class_db_id)

    async def process_js_method(self, session, method_node, class_id: int) -> None:
        method_name = method_node.key.name
        params = [param.name for param in method_node.value.params]
        method_summary = f"Method {method_name} with parameters: {', '.join(params)}"
        embedding = self.generate_embedding(method_summary)

        method_db_node = await session.run(
            "CREATE (n:JavaScriptMethod {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
            name=method_name, summary=method_summary, embedding=embedding
        )
        method_db_id = method_db_node.single()['id']

        await session.run(
            "MATCH (c:JavaScriptClass), (m:JavaScriptMethod) WHERE id(c) = $class_id AND id(m) = $method_id "
            "CREATE (c)-[:HAS_METHOD]->(m)",
            class_id=class_id, method_id=method_db_id
        )

    async def process_js_variable(self, session, var_node, file_id: int) -> None:
        for declaration in var_node.declarations:
            var_name = declaration.id.name
            var_type = var_node.kind  # 'var', 'let', or 'const'
            var_summary = f"{var_type.capitalize()} variable {var_name}"
            embedding = self.generate_embedding(var_summary)

            var_db_node = await session.run(
                "CREATE (n:JavaScriptVariable {name: $name, type: $type, summary: $summary, embedding: $embedding}) RETURN id(n)",
                name=var_name, type=var_type, summary=var_summary, embedding=embedding
            )
            var_db_id = var_db_node.single()['id']

            await session.run(
                "MATCH (f:JavaScriptFile), (v:JavaScriptVariable) WHERE id(f) = $file_id AND id(v) = $var_id "
                "CREATE (f)-[:CONTAINS]->(v)",
                file_id=file_id, var_id=var_db_id
            )
    async def IngestPdf(self, inputPath: str, inputLocation: str, inputName: str, currentOutputPath: str, projectID: str) -> None:
        reader = PdfReader(inputPath)
        num_pages = len(reader.pages)

        async with self.driver.session() as session:
            pdf_node = await session.run(
                "CREATE (n:PDF {name: $name, pages: $pages}) RETURN id(n)",
                name=inputName, pages=num_pages
            )
            pdf_id = pdf_node.single()['id']

            for i in range(num_pages):
                page = reader.pages[i]
                text = page.extract_text()

                medium_chunks = self.chunk_text(text, MEDIUM_CHUNK_SIZE)
                for j, medium_chunk in enumerate(medium_chunks):
                    medium_chunk_node = await session.run(
                        "CREATE (n:MediumChunk {content: $content, type: 'medium_chunk', page: $page}) RETURN id(n)",
                        content=medium_chunk, page=i+1
                    )
                    medium_chunk_id = medium_chunk_node.single()['id']

                    await session.run(
                        "MATCH (p:PDF), (c:MediumChunk) WHERE id(p) = $pdf_id AND id(c) = $chunk_id "
                        "CREATE (p)-[:HAS_CHUNK]->(c), (c)-[:PART_OF]->(p)",
                        pdf_id=pdf_id, chunk_id=medium_chunk_id
                    )

                    small_chunks = self.chunk_text(medium_chunk, SMALL_CHUNK_SIZE)
                    for k, small_chunk in enumerate(small_chunks):
                        small_chunk_node = await session.run(
                            "CREATE (n:SmallChunk {content: $content, type: 'small_chunk', page: $page}) RETURN id(n)",
                            content=small_chunk, page=i+1
                        )
                        small_chunk_id = small_chunk_node.single()['id']

                        await session.run(
                            "MATCH (mc:MediumChunk), (sc:SmallChunk) WHERE id(mc) = $medium_chunk_id AND id(sc) = $small_chunk_id "
                            "CREATE (mc)-[:HAS_CHUNK]->(sc), (sc)-[:PART_OF]->(mc)",
                            medium_chunk_id=medium_chunk_id, small_chunk_id=small_chunk_id
                        )

                        embedding = self.generate_embedding(small_chunk)
                        embedding_node = await session.run(
                            "CREATE (n:Embedding {embedding: $embedding, type: 'embedding'}) RETURN id(n)",
                            embedding=embedding
                        )
                        embedding_id = embedding_node.single()['id']

                        await session.run(
                            "MATCH (sc:SmallChunk), (e:Embedding) WHERE id(sc) = $small_chunk_id AND id(e) = $embedding_id "
                            "CREATE (sc)-[:HAS_EMBEDDING]->(e)",
                            small_chunk_id=small_chunk_id, embedding_id=embedding_id
                        )

class NIngest:
    def __init__(self, projectID: str = None, importer: NBaseImporter = NNeo4JImporter()):
        self.projectID = projectID or str(uuid.uuid4())
        self.currentOutputPath = os.path.expanduser(f"~/.ngest/projects/{self.projectID}")
        self.importer_ = importer

        if not projectID:
            os.makedirs(self.currentOutputPath, exist_ok=True)
            open(os.path.join(self.currentOutputPath, '.ngest_index'), 'a').close()
            logger.info(f"Created new project directory at {self.currentOutputPath}")

    async def Ingest(self, inputPath: str, currentOutputPath: str) -> int:
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
        return 0

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

# Example Usage:
# async def main():
#     ingest_instance = NIngest(importer=NNeo4JImporter())
#     await ingest_instance.Ingest('/path/to/directory_or_file', ingest_instance.currentOutputPath)
#
# if __name__ == "__main__":
#     asyncio.run(main())
