# Copyright 2024 Chris Odom
# MIT License

import os
import shutil
import uuid
import logging
from abc import ABC, abstractmethod
import ollama
from neo4j import GraphDatabase
import PIL.Image
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import pipeline, AutoTokenizer, AutoModel
import clang.cindex
import ast
import astor
import syn


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NBaseImporter(ABC):
    @abstractmethod
    def IngestFile(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        pass

    def ascertain_file_type(self, filename):
        if filename.lower() in ["doxyfile", "makefile"]:
            return 'text'
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.cpp', '.hpp', '.h', '.c']:
            return 'cpp'
        elif ext in ['.py']:
            return 'python'
        elif ext in ['.rs']:
            return 'rust'
        elif ext in ['.js']:
            return 'javascript'
        elif ext in ['.txt', '.toml']:
            return 'text'
        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            return 'image'
        return 'unknown'

    def IngestTxt(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        self.create_graph_nodes(inputPath, inputLocation, inputName, currentOutputPath, projectID)
        self.chunk_and_create_nodes(inputPath, inputLocation, inputName, currentOutputPath, projectID)

    def IngestCpp(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        self.IngestTxt(inputPath, inputLocation, inputName, currentOutputPath, projectID)

    def IngestPython(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        self.IngestTxt(inputPath, inputLocation, inputName, currentOutputPath, projectID)

    def IngestRust(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        self.IngestTxt(inputPath, inputLocation, inputName, currentOutputPath, projectID)

    def IngestJavascript(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        self.IngestTxt(inputPath, inputLocation, inputName, currentOutputPath, projectID)

    def IngestImg(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        # todo: add logging here.
        pass

    def create_graph_nodes(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        logger.info(f"Creating graph nodes for {inputPath}")

    def chunk_and_create_nodes(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        with open(inputPath, 'r') as file:
            text = file.read()

        chunks = self.chunk_text(text, 512)
        previous_chunk_id = None

        for i, chunk in enumerate(chunks):
            chunk_id = self.create_chunk_node(chunk, inputPath, i, projectID)
            if previous_chunk_id:
                self.create_chunk_relationship(previous_chunk_id, chunk_id)
            previous_chunk_id = chunk_id

    def chunk_text(self, text, chunk_size):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def create_chunk_node(self, chunk, inputPath, index, projectID):
        model = ollama.Model("nomic-embed-text")
        embedding = model.embed_text(chunk)
        chunk_id = f"{inputPath}_chunk_{index}"
        logger.info(f"Creating chunk node {chunk_id} with embedding {embedding}")
        return chunk_id

    def create_chunk_relationship(self, chunk_id1, chunk_id2):
        logger.info(f"Creating relationship between {chunk_id1} and {chunk_id2}")

class NFilesystemImporter(NBaseImporter):
    MAX_FILE_SIZE = 30 * 1024 * 1024  # 30 MB

    def IngestFile(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        if os.path.getsize(inputPath) > self.MAX_FILE_SIZE:
            logger.info(f"File {inputPath} skipped due to size.")
            open(os.path.join(currentOutputPath, f"{inputName}.skipped"), 'a').close()
            return

        shutil.copy(inputPath, os.path.join(currentOutputPath, inputName))
        logger.info(f"File {inputPath} ingested successfully.")
        file_type = self.ascertain_file_type(inputName)
        if file_type == 'text':
            self.IngestTxt(inputPath, inputLocation, inputName, currentOutputPath, projectID)
        elif file_type == 'cpp':
            self.IngestCpp(inputPath, inputLocation, inputName, currentOutputPath, projectID)
        elif file_type == 'python':
            self.IngestPython(inputPath, inputLocation, inputName, currentOutputPath, projectID)
        elif file_type == 'rust':
            self.IngestRust(inputPath, inputLocation, inputName, currentOutputPath, projectID)
        elif file_type == 'javascript':
            self.IngestJavascript(inputPath, inputLocation, inputName, currentOutputPath, projectID)
        elif file_type == 'image':
            self.IngestImg(inputPath, inputLocation, inputName, currentOutputPath, projectID)
        # else
        #     todo: add logging here.
        
        # Add more file type handlers as necessary

class NNeo4JImporter(NBaseImporter):
    def __init__(self, neo4j_url="bolt://localhost:7687"):
        self.neo4j_url = neo4j_url
        self.driver = GraphDatabase.driver(self.neo4j_url)
        self.index = clang.cindex.Index.create()
        self.summarizer = pipeline("summarization")
        self.image_model = models.densenet121(pretrained=True)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.code_model = AutoModel.from_pretrained("microsoft/codebert-base")
        logger.info(f"Neo4J importer initialized with URL {neo4j_url}")

    def IngestFile(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        file_type = self.ascertain_file_type(inputName)
        if file_type == 'cpp':
            self.IngestCpp(inputPath, inputLocation, inputName, currentOutputPath, projectID)
        elif file_type == 'python':
            self.IngestPython(inputPath, inputLocation, inputName, currentOutputPath, projectID)
        elif file_type == 'rust':
            self.IngestRust(inputPath, inputLocation, inputName, currentOutputPath, projectID)
        elif file_type == 'javascript':
            self.IngestJavascript(inputPath, inputLocation, inputName, currentOutputPath, projectID)
        elif file_type == 'image':
            self.IngestImg(inputPath, inputLocation, inputName, currentOutputPath, projectID)
        elif file_type == 'text':
            self.IngestTxt(inputPath, inputLocation, inputName, currentOutputPath, projectID)

    def IngestTxt(self, input_path, input_location, input_name, current_output_path, project_id):
        with open(input_path, 'r') as file:
            file_content = file.read()
            medium_chunks = [file_content[i:i+10000] for i in range(0, len(file_content), 10000)]
            with self.driver.session() as session:
                parent_doc_node = session.run(
                    "CREATE (n:Document {name: $name, type: 'text'}) RETURN id(n)",
                    name=input_name
                ).single().value()

                for medium_chunk in medium_chunks:
                    medium_chunk_node = session.run(
                        "CREATE (n:MediumChunk {content: $content, type: 'medium_chunk'}) RETURN id(n)",
                        content=medium_chunk
                    ).single().value()
                    session.run(
                        "MATCH (d:Document), (c:MediumChunk) WHERE id(d) = $doc_id AND id(c) = $chunk_id "
                        "CREATE (d)-[:HAS_CHUNK]->(c), (c)-[:PART_OF]->(d)",
                        doc_id=parent_doc_node, chunk_id=medium_chunk_node
                    )
                    small_chunks = [medium_chunk[i:i+1000] for i in range(0, len(medium_chunk), 1000)]
                    for small_chunk in small_chunks:
                        small_chunk_node = session.run(
                            "CREATE (n:SmallChunk {content: $content, type: 'small_chunk'}) RETURN id(n)",
                            content=small_chunk
                        ).single().value()
                        session.run(
                            "MATCH (mc:MediumChunk), (sc:SmallChunk) WHERE id(mc) = $medium_chunk_id AND id(sc) = $small_chunk_id "
                            "CREATE (mc)-[:HAS_CHUNK]->(sc), (sc)-[:PART_OF]->(mc)",
                            medium_chunk_id=medium_chunk_node, small_chunk_id=small_chunk_node
                        )
                        embedding = self.generate_embedding(small_chunk)
                        embedding_node = session.run(
                            "CREATE (n:Embedding {embedding: $embedding, type: 'embedding'}) RETURN id(n)",
                            embedding=embedding
                        ).single().value()
                        session.run(
                            "MATCH (sc:SmallChunk), (e:Embedding) WHERE id(sc) = $small_chunk_id AND id(e) = $embedding_id "
                            "CREATE (sc)-[:HAS_EMBEDDING]->(e)",
                            small_chunk_id=small_chunk_node, embedding_id=embedding_node
                        )

    def generate_embedding(self, text_chunk):
        model = ollama.Model("nomic-embed-text")
        embedding = model.embed_text(text_chunk)
        return embedding

    def extract_image_features(self, image):
        try:
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor = preprocess(image).unsqueeze(0)
            features = self.image_model(tensor)
            return features.tolist()
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return []

    def IngestImg(self, input_path, input_location, input_name, current_output_path, project_id):
        try:
            image = PIL.Image.open(input_path)
            features = self.extract_image_features(image)

            with self.driver.session() as session:
                image_node = session.run(
                    "CREATE (n:Image {name: $name, type: 'image'}) RETURN id(n)",
                    name=input_name
                ).single().value()
                features_node = session.run(
                    "CREATE (n:ImageFeatures {features: $features, type: 'image_features'}) RETURN id(n)",
                    features=features
                ).single().value()
                session.run(
                    "MATCH (i:Image), (f:ImageFeatures) WHERE id(i) = $image_id AND id(f) = $features_id "
                    "CREATE (i)-[:HAS_FEATURES]->(f)",
                    image_id=image_node, features_id=features_node
                )
        except Exception as e:
            logger.error(f"Error ingesting image: {e}")

   def summarize_cpp_class(self, cls):
        # Generate a summary of the class using LLM
        description = f"Class {cls.spelling} with public methods: "
        public_methods = [c.spelling for c in cls.get_children() if c.access_specifier == clang.cindex.AccessSpecifier.PUBLIC]
        description += ", ".join(public_methods)
        summary = self.summarizer(description, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        return summary

    def summarize_cpp_function(self, func):
        # Generate a summary of the function using LLM
        description = f"Function {func.spelling} in namespace {func.semantic_parent.spelling}. It performs the following tasks: "
        # Additional parsing to extract function details
        summary = self.summarizer(description, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        return summary

    def IngestCpp(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        # Parse the C++ file
        tu = self.index.parse(inputPath)
        functions = []
        classes = []

        for node in tu.cursor.get_children():
            if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                functions.append(node)
            elif node.kind in [clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL]:
                classes.append(node)

        with self.driver.session() as session:
            for cls in classes:
                class_summary = self.summarize_cpp_class(cls)
                embedding = self.generate_embedding(class_summary)
                class_node = session.run(
                    "CREATE (n:Class {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                    name=cls.spelling, summary=class_summary, embedding=embedding
                ).single().value()
                
                for func in cls.get_children():
                    if func.kind == clang.cindex.CursorKind.CXX_METHOD:
                        function_summary = self.summarize_cpp_function(func)
                        embedding = self.generate_embedding(function_summary)
                        function_node = session.run(
                            "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                            name=func.spelling, summary=function_summary, embedding=embedding
                        ).single().value()
                        session.run(
                            "MATCH (c:Class), (f:Function) WHERE id(c) = $class_id AND id(f) = $function_id "
                            "CREATE (c)-[:HAS_METHOD]->(f)",
                            class_id=class_node, function_id=function_node
                        )

            for func in functions:
                function_summary = self.summarize_cpp_function(func)
                embedding = self.generate_embedding(function_summary)
                function_node = session.run(
                    "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                    name=func.spelling, summary=function_summary, embedding=embedding
                ).single().value()

    def tokenize_code(self, code):
        try:
            tokens = self.tokenizer.tokenize(code)
            return tokens
        except Exception as e:
            logger.error(f"Error tokenizing code: {e}")
            return []

    def extract_code_features(self, code):
        try:
            tokens = self.tokenize_code(code)
            inputs = self.tokenizer(tokens, return_tensors="pt")
            features = self.code_model(**inputs)
            return features.last_hidden_state.tolist()
        except Exception as e:
            logger.error(f"Error extracting code features: {e}")
            return []

    def IngestCode(self, input_path, input_location, input_name, current_output_path, project_id):
        try:
            code = open(input_path, 'r').read()
            features = self.extract_code_features(code)

            with self.driver.session() as session:
                code_node = session.run(
                    "CREATE (n:Code {name: $name, type: 'code'}) RETURN id(n)",
                    name=input_name
                ).single().value()
                features_node = session.run(
                    "CREATE (n:CodeFeatures {features: $features, type: 'code_features'}) RETURN id(n)",
                    features=features
                ).single().value()
                session.run(
                    "MATCH (c:Code), (f:CodeFeatures) WHERE id(c) = $code_id AND id(f) = $features_id "
                    "CREATE (c)-[:HAS_FEATURES]->(f)",
                    code_id=code_node, features_id=features_node
                )
        except Exception as e:
            logger.error(f"Error ingesting code: {e}")

    def IngestPython(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        # Parse the Python file
        with open(inputPath, 'r') as file:
            tree = ast.parse(file.read(), filename=inputPath)

        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        with self.driver.session() as session:
            for cls in classes:
                class_summary = self.summarize_python_class(cls)
                embedding = self.generate_embedding(class_summary)
                class_node = session.run(
                    "CREATE (n:Class {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                    name=cls.name, summary=class_summary, embedding=embedding
                ).single().value()
                
                for func in cls.body:
                    if isinstance(func, ast.FunctionDef):
                        function_summary = self.summarize_python_function(func)
                        embedding = self.generate_embedding(function_summary)
                        function_node = session.run(
                            "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                            name=func.name, summary=function_summary, embedding=embedding
                        ).single().value()
                        session.run(
                            "MATCH (c:Class), (f:Function) WHERE id(c) = $class_id AND id(f) = $function_id "
                            "CREATE (c)-[:HAS_METHOD]->(f)",
                            class_id=class_node, function_id=function_node
                        )

            for func in functions:
                if isinstance(func, ast.FunctionDef):
                    function_summary = self.summarize_python_function(func)
                    embedding = self.generate_embedding(function_summary)
                    function_node = session.run(
                        "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                        name=func.name, summary=function_summary, embedding=embedding
                    ).single().value()

    def summarize_python_class(self, cls):
        # Generate a summary of the class using LLM
        description = f"Class {cls.name} with methods: "
        methods = [func.name for func in cls.body if isinstance(func, ast.FunctionDef)]
        description += ", ".join(methods)
        summary = self.summarizer(description, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        return summary

    def summarize_python_function(self, func):
        # Generate a summary of the function using LLM
        description = f"Function {func.name} with arguments: {', '.join(arg.arg for arg in func.args.args)}. It performs the following tasks: "
        # Additional parsing to extract function details
        summary = self.summarizer(description, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        return summary

    def IngestRust(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        # Parse the Rust file
        tree = syn.parse_file(inputPath)
        functions = [item for item in tree.items if isinstance(item, syn.ItemFn)]
        impls = [item for item in tree.items if isinstance(item, syn.ItemImpl)]

        with self.driver.session() as session:
            for impl in impls:
                impl_summary = self.summarize_rust_impl(impl)
                embedding = self.generate_embedding(impl_summary)
                impl_node = session.run(
                    "CREATE (n:Impl {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                    name=impl.trait_.path.segments[0].ident, summary=impl_summary, embedding=embedding
                ).single().value()
                
                for item in impl.items:
                    if isinstance(item, syn.ImplItemMethod):
                        function_summary = self.summarize_rust_function(item)
                        embedding = self.generate_embedding(function_summary)
                        function_node = session.run(
                            "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                            name=item.sig.ident, summary=function_summary, embedding=embedding
                        ).single().value()
                        session.run(
                            "MATCH (i:Impl), (f:Function) WHERE id(i) = $impl_id AND id(f) = $function_id "
                            "CREATE (i)-[:HAS_METHOD]->(f)",
                            impl_id=impl_node, function_id=function_node
                        )

            for func in functions:
                function_summary = self.summarize_rust_function(func)
                embedding = self.generate_embedding(function_summary)
                function_node = session.run(
                    "CREATE (n:Function {name: $name, summary: $summary, embedding: $embedding}) RETURN id(n)",
                    name=func.sig.ident, summary=function_summary, embedding=embedding
                ).single().value()

    def summarize_rust_impl(self, impl):
        # Generate a summary of the implementation block using LLM
        description = f"Implementation of trait {impl.trait_.path.segments[0].ident} with methods: "
        methods = [item.sig.ident for item in impl.items if isinstance(item, syn.ImplItemMethod)]
        description += ", ".join([str(m) for m in methods])
        summary = self.summarizer(description, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        return summary

    def summarize_rust_function(self, func):
        # Generate a summary of the function using LLM
        description = f"Function {func.sig.ident} with arguments: {', '.join(arg.pat.ident for arg in func.sig.inputs)}. It performs the following tasks: "
        # Additional parsing to extract function details
        summary = self.summarizer(description, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        return summary

class NIngest:
    def __init__(self, projectID=None, importer=NNeo4JImporter()):
        self.projectID = projectID or str(uuid.uuid4())
        self.currentOutputPath = os.path.expanduser(f"~/.ngest/projects/{self.projectID}")
        self.importer_ = importer

        if not projectID:
            os.makedirs(self.currentOutputPath, exist_ok=True)
            open(os.path.join(self.currentOutputPath, '.ngest_index'), 'a').close()
            logger.info(f"Created new project directory at {self.currentOutputPath}")

    def Ingest(self, inputPath, currentOutputPath):
        if not os.path.exists(inputPath):
            logger.error(f"Invalid path: {inputPath}")
            return -1

        inputType = 'd' if os.path.isdir(inputPath) else 'f'
        with open(os.path.join(currentOutputPath, '.ngest_index'), 'a') as index_file:
            index_file.write(f"{inputType},{inputPath}\n")
        logger.info(f"INGESTING: {inputPath}")

        inputLocation, inputName = os.path.split(inputPath)
        if inputType == 'd':
            self.IngestDirectory(inputPath, inputLocation, inputName, currentOutputPath, self.projectID)
        else:
            self.IngestFile(inputPath, inputLocation, inputName, currentOutputPath, self.projectID)

    def IngestDirectory(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        newOutputPath = os.path.join(currentOutputPath, inputName)
        os.makedirs(newOutputPath, exist_ok=True)

        for item in os.listdir(inputPath):
            itemPath = os.path.join(inputPath, item)
            self.Ingest(itemPath, newOutputPath)

    def IngestFile(self, inputPath, inputLocation, inputName, currentOutputPath, projectID):
        self.importer_.IngestFile(inputPath, inputLocation, inputName, currentOutputPath, projectID)


# Example Usage:
# ingest_instance = NIngest(importer=NNeo4JImporter())
# ingest_instance.Ingest('/path/to/directory_or_file', ingest_instance.currentOutputPath)



