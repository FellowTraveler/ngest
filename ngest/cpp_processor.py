# Copyright 2024 Chris Odom
# MIT License

import logging
import traceback
import asyncio
from collections import deque, defaultdict
from typing import Dict, Any
import copy
import pprint
import aiofiles

import ngest
from ngest.custom_errors import FileProcessingError, DatabaseError

import clang.cindex

logging.basicConfig(level=logging.info, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from enum import Enum


class FileType(Enum):
    CppCode = 1
    CppHeader = 2
    PythonCode = 3
    RustCode = 4
    JavascriptCode = 5
    
class CppProcessor:
    def __init__(self, do_summarize_text):
        self.failed_nodes = deque()
        self.max_retries = 3
        self.retry_semaphore = asyncio.Semaphore(2)  # Limit concurrent retries

        self.do_summarize_text = do_summarize_text
        
        self.namespaces = defaultdict(dict)
        self.classes = defaultdict(dict)
        self.methods = defaultdict(dict)
        self.functions = defaultdict(dict)
#        self.header_files = {}
        
        self.lock_namespaces = asyncio.Lock()
        self.lock_classes = asyncio.Lock()
        self.lock_methods = asyncio.Lock()
        self.lock_functions = asyncio.Lock()
#        self.lock_header_files = asyncio.Lock()

        self.all_declarations = defaultdict(list)
        self.declaration_types = defaultdict(lambda: defaultdict(str))
        self.declarations_lock = asyncio.Lock()
        
        self.class_declarations = {}
        self.class_implementations = {}
        self.method_declarations = {}
        self.method_implementations = {}
        self.function_declarations = {}
        self.function_implementations = {}

    def get_relationship_type(self, node, file_path):
        if node.is_definition():
            if node.kind in (clang.cindex.CursorKind.CLASS_DECL,
                             clang.cindex.CursorKind.STRUCT_DECL,
                             clang.cindex.CursorKind.CLASS_TEMPLATE,
                             clang.cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                return 'DEFINED_IN_FILE'
            else:
                return 'IMPLEMENTED_IN_FILE'
        elif file_path.endswith(('.hpp', '.h', '.hxx')):
            return 'DECLARED_IN_FILE'
        else:
            return 'INCLUDED_IN_FILE'
            
    def get_usr(self, node):
        """
        Get the USR for a node and return it in normalized form.
        Always use this method to get USRs, as it handles normalization internally.
        """
        if node:
            usr = clang.cindex.conf.lib.clang_getCursorUSR(node)
            normalized_usr = self.normalize_usr(usr.decode('utf-8') if isinstance(usr, bytes) else usr)
            logger.info(f"Generated USR for node {node.spelling}: {normalized_usr}")
            return normalized_usr
        return None
        
    def normalize_usr(self, usr):
        if usr is None:
            return ''
        return usr.replace('@', '_').replace('#', '_').replace(':', '_').replace('>', '_').replace('<', '_').replace('&', '_').replace('$', '_')
    
    async def update_namespace(self, full_name, details):
        async with self.lock_namespaces:
            existing_info = self.namespaces.get(full_name, {})
            merged_details = existing_info.copy()
            for key, value in details.items():
                if key == 'files':
                    existing_files = merged_details.get('files', [])
                    new_file = value[0]
                    file_exists = False
                    for existing_file in existing_files:
                        if existing_file['file_id'] == new_file['file_id']:
                            file_exists = True
                            if new_file['file_type'] == FileType.CppCode or len(new_file['raw_code']) > len(existing_file['raw_code']):
                                existing_file.update(new_file)
                            break
                    if not file_exists:
                        existing_files.append(new_file)
                    merged_details['files'] = existing_files
                elif value or isinstance(value, (int, float)):
                    merged_details[key] = value
            self.namespaces[full_name] = merged_details
        
        
    async def update_class(self, full_name, details):
        async with self.lock_classes:
            try:
                existing_info = self.classes.get(full_name, {})
                merged_details = existing_info.copy()
                for key, value in details.items():
                    if key == 'files':
                        existing_files = merged_details.get('files', [])
                        new_file = value[0]
                        file_exists = False
                        for existing_file in existing_files:
                            if existing_file['file_id'] == new_file['file_id']:
                                file_exists = True
                                if new_file['relationship_type'] == 'DEFINED_IN_FILE' or len(new_file['raw_code']) > len(existing_file['raw_code']):
                                    existing_file.update(new_file)
                                break
                        if not file_exists:
                            existing_files.append(new_file)
                        merged_details['files'] = existing_files
                    elif value or isinstance(value, (int, float)):
                        merged_details[key] = value
                self.classes[full_name] = merged_details
            except Exception as e:
                logger.error(f"Exception thrown in update_class: {e}")
                
    async def update_method(self, full_name, details):
        async with self.lock_methods:
            try:
                existing_info = self.methods.get(full_name, {})
                merged_details = existing_info.copy()
                for key, value in details.items():
                    if key == 'files':
                        existing_files = merged_details.get('files', [])
                        new_file = value[0]
                        file_exists = False
                        for existing_file in existing_files:
                            if existing_file['file_id'] == new_file['file_id']:
                                file_exists = True
                                if new_file['relationship_type'] == 'IMPLEMENTED_IN_FILE' or len(new_file['raw_code']) > len(existing_file['raw_code']):
                                    existing_file.update(new_file)
                                break
                        if not file_exists:
                            existing_files.append(new_file)
                        merged_details['files'] = existing_files
                    elif value or isinstance(value, (int, float)):
                        merged_details[key] = value
                self.methods[full_name] = merged_details
            except Exception as e:
                logger.error(f"Exception thrown in update_method: {e}")

        
    async def update_function(self, full_name, details):
        async with self.lock_functions:
            try:
                existing_info = self.functions.get(full_name, {})
                merged_details = existing_info.copy()
                for key, value in details.items():
                    if key == 'files':
                        existing_files = merged_details.get('files', [])
                        new_file = value[0]
                        file_exists = False
                        for existing_file in existing_files:
                            if existing_file['file_id'] == new_file['file_id']:
                                file_exists = True
                                if new_file['relationship_type'] in ['IMPLEMENTED_IN_FILE'] or len(new_file['raw_code']) > len(existing_file['raw_code']):
                                    existing_file.update(new_file)
                                break
                        if not file_exists:
                            existing_files.append(new_file)
                        merged_details['files'] = existing_files
                    elif value or isinstance(value, (int, float)):
                        merged_details[key] = value
                self.functions[full_name] = merged_details
            except Exception as e:
                logger.error(f"Exception thrown in update_function: {e}")

    async def parse_cpp_file(self, file_id: str, inputPath: str, inputLocation: str, localPath: str, project_id: str):
        try:
#            logger.info(f"Creating Clang Index for parsing C++ file: {inputPath}")
            index = clang.cindex.Index.create()
            translation_unit = index.parse(inputPath)
            
            if translation_unit is None:
                logger.error(f"Failed to parse {inputPath}. Translation unit is None.")
                return

#            logger.info(f"Parsed file, and now reading contents whole: {inputPath}")
            try:
                with open(inputPath, 'r') as file:
                    file_contents = file.read()
#                logger.info(f"Successfully read {len(file_contents)} characters from {inputPath}")
            except IOError as io_err:
                logger.error(f"IOError while reading {inputPath}: {io_err}")
                raise

            # Save raw code of header files
#            if inputPath.endswith('.hpp') or inputPath.endswith('.h'):
##                logger.info(f"Saving contents of header file: {inputPath}")
#                async with self.lock_header_files:
#                    self.header_files[inputPath] = file_contents
#                logger.info(f"Saved contents of header file: {inputPath}")

#            logger.info(f"Starting to process nodes for {inputPath}")
            await self.process_nodes(file_id=file_id, node=translation_unit.cursor, inputLocation=inputLocation, project_path=localPath, project_id=project_id, is_cpp_file=inputPath.endswith('.cpp'))
#            logger.info(f"Finished parsing nodes for {inputPath}")

        except clang.cindex.TranslationUnitLoadError as tu_error:
            logger.error(f"TranslationUnitLoadError for {inputPath}: {tu_error}")
            logger.error(f"Clang diagnostics: {[diag.spelling for diag in translation_unit.diagnostics]}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing C++ file {inputPath}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def get_class_info(self, full_name):
        async with self.lock_classes:
            return self.classes.get(full_name, {})
        
    async def get_method_info(self, full_name):
        async with self.lock_methods:
            return self.methods.get(full_name, {})
        
    async def get_function_info(self, full_name):
        async with self.lock_functions:
            return self.functions.get(full_name, {})
        
    async def prepare_summarization_tasks(self):
        tasks = []
        async with self.lock_classes:
            logger.info(f"Preparing summarization tasks. Total classes: {len(self.classes)}")
            for full_name, class_info in self.classes.items():
                namespace = class_info.get('namespace', '')
#                logger.info(f"Checking class: {full_name}, namespace: {namespace}")
                
                type = class_info.get('type', 'Class')
                
                # Create a simplified version of class_info for logging
#                simplified_info = {
#                    'type': class_info.get('type'),
#                    'name': class_info.get('name'),
#                    'scope': class_info.get('scope'),
#                    'short_name': class_info.get('short_name'),
#                    'has_description': 'description' in class_info,
#                    'has_interface_description': 'interface_description' in class_info,
#                    'has_implementation_description': 'implementation_description' in class_info,
#                    'has_raw_comment': 'raw_comment' in class_info,
#                    'file_path': class_info.get('file_path'),
#                    'is_cpp_file': class_info.get('is_cpp_file'),
#                    'has_raw_code': 'raw_code' in class_info,
#                    'has_interface_embedding': 'interface_embedding' in class_info,
#                    'has_implementation_embedding': 'implementation_embedding' in class_info
#                }
                
#                logger.info(f"Simplified class info: {simplified_info}")
                
                if 'interface_description' in class_info or 'implementation_description' in class_info:
                    tasks.append((type, full_name, copy.deepcopy(class_info)))
#                    logger.info(f"Added summarization task for class: {full_name}")
                else:
                    logger.warning(f"Skipping summarization for class {full_name} in namespace {namespace}. "
                                   f"Missing one of: interface_description: {'interface_description' in class_info}, "
                                   f"implementation_description: {'implementation_description' in class_info}")
                                   
        async with self.lock_methods:
            logger.info(f"Preparing summarization tasks for methods. Total methods: {len(self.methods)}")
            for full_name, method_info in self.methods.items():
                namespace = method_info.get('namespace', '')
#                simplified_info = {
#                    'type': method_info.get('type'),
#                    'name': method_info.get('name'),
#                    'scope': method_info.get('scope'),
#                    'short_name': method_info.get('short_name'),
#                    'has_description': 'description' in method_info,
#                    'has_raw_comment': 'raw_comment' in method_info,
#                    'file_path': method_info.get('file_path'),
#                    'is_cpp_file': method_info.get('is_cpp_file'),
#                    'has_raw_code': 'raw_code' in method_info,
#                    'has_embedding': 'embedding' in method_info
#                }
#                logger.info(f"Checking method: {full_name}, namespace: {namespace}")
#                logger.info(f"Simplified method info: {simplified_info}")
                
                if 'description' in method_info:
                    tasks.append(('Method', full_name, copy.deepcopy(method_info)))
#                    logger.info(f"Added summarization task for method: {full_name}")
                else:
                    logger.warning(f"Skipping summarization for method {full_name} in namespace {namespace} due to missing description")

        async with self.lock_functions:
            logger.info(f"Preparing summarization tasks for functions. Total functions: {len(self.functions)}")
            for full_name, function_info in self.functions.items():
                namespace = function_info.get('namespace', '')
#                simplified_info = {
#                    'type': function_info.get('type'),
#                    'name': function_info.get('name'),
#                    'scope': function_info.get('scope'),
#                    'short_name': function_info.get('short_name'),
#                    'has_description': 'description' in function_info,
#                    'has_raw_comment': 'raw_comment' in function_info,
#                    'file_path': function_info.get('file_path'),
#                    'is_cpp_file': function_info.get('is_cpp_file'),
#                    'has_raw_code': 'raw_code' in function_info,
#                    'has_embedding': 'embedding' in function_info
#                }
#                logger.info(f"Checking function: {full_name}, namespace: {namespace}")
#                logger.info(f"Simplified function info: {simplified_info}")
                
                if 'description' in function_info:
                    tasks.append(('Function', full_name, copy.deepcopy(function_info)))
#                    logger.info(f"Added summarization task for function: {full_name}")
                else:
                    logger.warning(f"Skipping summarization for function {full_name} in namespace {namespace} due to missing description")

        logger.info(f"Total summarization tasks prepared: {len(tasks)}")
        return tasks
    
    
    
    async def prepare_storage_tasks(self):
        tasks = []
        
        async with self.lock_classes:
            for full_name, class_info in self.classes.items():
                if 'interface_summary' in class_info or 'implementation_summary' in class_info:
                    usr = class_info.get('usr', '')
                    namespace_usr = class_info.get('namespace_usr', '')
                    task_type = class_info.get('type', 'Class')
                    if usr:
                        logger.info(f"Preparing storage task for {task_type}: {full_name}, USR: {usr}")
                        files = class_info.get('files', [])
                        converted_files = [{**f, 'file_type': f['file_type'].name} for f in files]
#                        for file in files:
#                            file['file_type'] = file['file_type'].name if isinstance(file['file_type'], FileType) else str(file['file_type'])
                        tasks.append((task_type, full_name, {
                            'usr': usr,
                            'type': task_type,
                            'full_name': full_name,
                            'namespace': class_info.get('namespace', ''),
                            'namespace_usr': namespace_usr,
                            'short_name': class_info.get('short_name', ''),
                            'scope': class_info.get('scope', ''),
                            'interface_summary': class_info.get('interface_summary', ''),
                            'implementation_summary': class_info.get('implementation_summary', ''),
                            'interface_embedding': class_info.get('interface_embedding', []),
                            'implementation_embedding': class_info.get('implementation_embedding', []),
                            'base_classes': [{'usr': base.get('usr', ''), 'name': base.get('name', ''), 'access_specifier': base.get('access_specifier')} for base in class_info.get('base_classes', [])],
                            'files': converted_files
                        }))
                    else:
                        logger.warning(f"Skipping class {full_name} due to missing USR")
                else:
                    logger.warning(f"Skipping class {full_name} due to missing summary(ies)")

        async with self.lock_methods:
            for full_name, method_info in self.methods.items():
                if 'summary' in method_info:
                    usr = method_info.get('usr', '')
                    namespace_usr = method_info.get('namespace_usr', '')
                    task_type = method_info.get('type', 'Method')
                    if usr:
                        logger.info(f"Preparing storage task for {task_type}: {full_name}, USR: {usr}")
                        files = method_info.get('files', [])
                        converted_files = [{**f, 'file_type': f['file_type'].name} for f in files]
#                        for file in files:
#                            file['file_type'] = file['file_type'].name if isinstance(file['file_type'], FileType) else str(file['file_type'])
                        tasks.append(('Method', full_name, {
                            'usr': usr,
                            'type': task_type,
                            'full_name': full_name,
                            'namespace': method_info.get('namespace', ''),
                            'namespace_usr': namespace_usr,
                            'short_name': method_info.get('short_name', ''),
                            'scope': method_info.get('scope', ''),
                            'summary': method_info.get('summary', ''),
                            'embedding': method_info.get('embedding', []),
                            'class_usr': method_info.get('class_usr', ''),
                            'files': converted_files
                        }))
                    else:
                        logger.warning(f"Skipping method {full_name} due to missing USR")
                else:
                    logger.warning(f"Skipping method {full_name} due to missing summary")

        async with self.lock_functions:
            for full_name, function_info in self.functions.items():
                if 'summary' in function_info:
                    usr = function_info.get('usr', '')
                    namespace_usr = function_info.get('namespace_usr', '')
                    task_type = function_info.get('type', 'Function')
                    if usr:
                        logger.info(f"Preparing storage task for {task_type}: {full_name}, USR: {usr}")
                        files = function_info.get('files', [])
                        converted_files = [{**f, 'file_type': f['file_type'].name} for f in files]
#                        for file in files:
#                            file['file_type'] = file['file_type'].name if isinstance(file['file_type'], FileType) else str(file['file_type'])
                        tasks.append(('Function', full_name, {
                            'usr': usr,
                            'type': task_type,
                            'full_name': full_name,
                            'namespace': function_info.get('namespace', ''),
                            'namespace_usr': namespace_usr,
                            'short_name': function_info.get('short_name', ''),
                            'scope': function_info.get('scope', ''),
                            'summary': function_info.get('summary', ''),
                            'embedding': function_info.get('embedding', []),
                            'files': converted_files
                        }))
                    else:
                        logger.warning(f"Skipping function {full_name} due to missing USR")
                else:
                    logger.warning(f"Skipping function {full_name} due to missing summary")

        return tasks
    
#    async def retry_failed_nodes(self, project_path, project_id):
#        retry_tasks = []
#        while self.failed_nodes:
#            node, _, _, is_cpp_file, retry_count = self.failed_nodes.popleft()
#            if retry_count >= self.max_retries:
#                logger.warning(f"Node {node.spelling} has reached max retries. Skipping.")
#                continue
#            retry_tasks.append(self.retry_single_node(node, project_path, project_id, is_cpp_file, retry_count))
#
#        await asyncio.gather(*retry_tasks)
#
#        if self.failed_nodes:
#            logger.warning(f"{len(self.failed_nodes)} nodes still failed after retries")


    async def retry_single_node(self, node, inputLocation, project_path, project_id, is_cpp_file, retry_count):
        async with self.retry_semaphore:
            try:
                logger.info(f"Retrying node {node.spelling} (attempt {retry_count + 1})")
                await self.process_single_node(node, inputLocation, project_path, project_id, is_cpp_file)
            except Exception as e:
                logger.error(f"Error retrying node {node.spelling}: {e}")
                self.failed_nodes.append((node, project_path, project_id, is_cpp_file, retry_count + 1))
                
                
    async def process_nodes(self, file_id: str, node, inputLocation, project_path, project_id, is_cpp_file):
    
#        logger.info(f"process_nodes: project_path: {project_path}")
        
        try:
            if node.kind == clang.cindex.CursorKind.TRANSLATION_UNIT:
                # For the translation unit, process all children without checking the file path
                for child in node.get_children():
                    await self.process_nodes(file_id, child, inputLocation, project_path, project_id, is_cpp_file)
            else:
                # Process the current node
                await self.process_single_node(file_id=file_id, node=node, inputLocation=inputLocation, project_path=project_path, project_id=project_id, is_cpp_file=is_cpp_file)

                # Process child nodes
                if node.kind not in (clang.cindex.CursorKind.NAMESPACE, clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL, clang.cindex.CursorKind.CLASS_TEMPLATE, clang.cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                    await self.process_child_nodes(file_id=file_id, node=node, inputLocation=inputLocation, project_path=project_path, project_id=project_id, is_cpp_file=is_cpp_file)
        except Exception as e:
            logger.error(f"Error processing node {node.spelling}: {e}")
            self.failed_nodes.append((node, project_path, project_id, is_cpp_file, 0))

    async def process_child_nodes(self, file_id: str, node, inputLocation, project_path, project_id, is_cpp_file):
        for child in node.get_children():
            if child.location.file and project_path in child.location.file.name:
#                logger.info(f"Processing child node {child.spelling}, kind: {child.kind}")
                await self.process_nodes(file_id=file_id, node=child, inputLocation=inputLocation, project_path=project_path, project_id=project_id, is_cpp_file=is_cpp_file)

    async def is_exported(self, node):
        for token in node.get_tokens():
            if token.spelling in ["__declspec(dllexport)", "__attribute__((visibility(\"default\")))"]:
                return True
        return False

#    def get_node_type(self, usr):
#        if usr.startswith('c__F_') or usr.endswith('_F_') or '_F_' in usr:  # Functions
#            return 'Function'
#        elif usr.startswith('c__M_') or usr.startswith('c__CM_'):  # Methods
#            return 'Method'
#        elif usr.startswith('c__S_') or usr.startswith('c__C_'):  # Structs or Classes
#            return 'Class'
#        elif usr.startswith('c__N_'):  # Namespaces
#            return 'Namespace'
#        elif '_F_' in usr:  # Functions in header files
#            return 'Function'
#        elif usr.startswith('c_') and '.h_F_' in usr:  # Functions in specific header files
#            return 'Function'
#        else:
#            # Log unknown USR types for further investigation
#            logger.warning(f"Unknown USR type: {usr}")
#            return 'Unknown'


    def get_node_type_from_kind(self, kind):
        if kind in (clang.cindex.CursorKind.NAMESPACE,):
            return 'Namespace'
        elif kind == clang.cindex.CursorKind.STRUCT_DECL:
            return 'Struct'
        elif kind in (clang.cindex.CursorKind.CLASS_DECL,
                      clang.cindex.CursorKind.CLASS_TEMPLATE, clang.cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
            return 'Class'
        elif kind in (clang.cindex.CursorKind.CXX_METHOD, clang.cindex.CursorKind.CONSTRUCTOR, clang.cindex.CursorKind.DESTRUCTOR):
            return 'Method'
        elif kind in (clang.cindex.CursorKind.FUNCTION_DECL,):
            return 'Function'
        else:
            return 'Unknown'

    async def finalize_relationships(self, project_id, session):
        async with self.declarations_lock:
            for usr, declarations in self.all_declarations.items():
                node_type = self.get_node_type_from_kind(declarations[0][1])
                
                if node_type in ['Class', 'Struct']:
                    # Find the declaration that is a definition (if any)
                    definition = next((decl for decl in declarations if self.declaration_types[usr][decl[0]] == 'DEFINED_IN_FILE'), None)
                    
                    if definition:
                        file_full_path = definition[0]
                        
                        # If we found a definition, create a DEFINED_IN_FILE relationship
                        await self.create_relationship(session, usr, file_full_path, 'DEFINED_IN_FILE', definition[2], definition[3], project_id, node_type)
                        await self.create_reverse_relationship(session, file_full_path, usr, 'DEFINES_TYPE', definition[2], definition[3], project_id, 'File', node_type)
            
                elif node_type in ['Method', 'Function']:
                    declaration_file = next((decl[0] for decl in declarations if self.declaration_types[usr][decl[0]] == 'DECLARED_IN_FILE'), None)
                    implementation_file = next((decl[0] for decl in declarations if self.declaration_types[usr][decl[0]] == 'IMPLEMENTED_IN_FILE'), None)
                    
                    if declaration_file and implementation_file:
                        await self.create_relationship(session, usr, declaration_file, 'DECLARED_IN_FILE', declarations[0][2], declarations[0][3], project_id, node_type)
                        await self.create_relationship(session, usr, implementation_file, 'IMPLEMENTED_IN_FILE', declarations[0][2], declarations[0][3], project_id, node_type)
                    elif implementation_file:
                        await self.create_relationship(session, usr, implementation_file, 'IMPLEMENTED_IN_FILE', declarations[0][2], declarations[0][3], project_id, node_type)
                    elif declaration_file:
                        await self.create_relationship(session, usr, declaration_file, 'DECLARED_IN_FILE', declarations[0][2], declarations[0][3], project_id, node_type)
                
                else:
                    logger.warning(f"Unknown node type {node_type} for USR {usr}")

            for usr in self.all_declarations:
                if usr not in self.class_declarations and usr not in self.method_declarations and usr not in self.function_declarations:
                    logger.warning(f"USR {usr} not found in class, method, or function declarations")
                    
                
    async def create_relationship(self, session, usr, local_path, relationship_type, line, column, project_id, node_type):
        query = f"""
        MATCH (n:{node_type} {{usr: $usr, project_id: $project_id}})
        MATCH (f:File {{full_path: $file_path, project_id: $project_id}})
        MERGE (n)-[r:{relationship_type}]->(f)
        SET r.line = $line, r.column = $column
        RETURN n, f, r
        """
        result = await session.run(query, usr=usr, file_path=local_path, line=line, column=column, project_id=project_id)
        summary = await result.consume()
        
        if summary.counters.relationships_created > 0:
            logger.info(f"Created {relationship_type} relationship for {node_type} {usr} to File {local_path}")
        else:
            logger.warning(f"Failed to create {relationship_type} relationship for {node_type} {usr} to File {local_path}")
            # Add debug information
            debug_query = f"""
            MATCH (n:{node_type} {{usr: $usr, project_id: $project_id}})
            RETURN n
            """
            debug_result = await session.run(debug_query, usr=usr, project_id=project_id)
            debug_record = await debug_result.single()
            if debug_record and 'n' in debug_record:
                debug_node = debug_record['n']
                if hasattr(debug_node, 'properties'):
                    logger.info(f"Debug: Found node {node_type} with properties: {debug_node.properties}")
                else:
                    logger.info(f"Debug: Found node {node_type}, but it doesn't have properties attribute")
            else:
                logger.warning(f"Debug: {node_type} node  with USR {usr} not found")


    async def create_reverse_relationship(self, session, file_path, usr, relationship_type, line, column, project_id, from_node_type, to_node_type):
        query = f"""
        MATCH (f:{from_node_type} {{full_path: $file_path, project_id: $project_id}})
        MATCH (n:{to_node_type} {{usr: $usr, project_id: $project_id}})
        MERGE (f)-[r:{relationship_type}]->(n)
        SET r.line = $line, r.column = $column
        RETURN f, n, r
        """
        result = await session.run(query, file_path=file_path, usr=usr, line=line, column=column, project_id=project_id)
        summary = await result.consume()
        
        if summary.counters.relationships_created > 0:
            logger.info(f"Created {relationship_type} relationship from File {file_path} to {to_node_type} {usr}")
        else:
            logger.warning(f"Failed to create {relationship_type} relationship from File {file_path} to {to_node_type} {usr}")
            # Add debug information
            debug_query = f"""
            MATCH (f:{from_node_type} {{full_path: $file_path, project_id: $project_id}})
            RETURN f
            """
            debug_result = await session.run(debug_query, file_path=file_path, project_id=project_id)
            debug_record = await debug_result.single()
            if debug_record and 'f' in debug_record:
                debug_node = debug_record['f']
                if hasattr(debug_node, 'properties'):
                    logger.info(f"Debug: Found node {from_node_type} with properties: {debug_node.properties}")
                else:
                    logger.info(f"Debug: Found node {from_node_type}, but it doesn't have properties attribute")
            else:
                logger.warning(f"Debug: {from_node_type} node with full_path {file_path} not found")
                
                
                        
    async def process_single_node(self, file_id: str, node, inputLocation, project_path, project_id, is_cpp_file):
        try:
            if node.location.file and project_path in node.location.file.name:
                file_name = node.location.file.name if node.location.file else ''
                full_scope = self.get_full_scope(node)
                full_name = self.get_full_scope(node, include_self=True)
                namespace = self.get_namespace(node)

                if node.kind in (clang.cindex.CursorKind.NAMESPACE, clang.cindex.CursorKind.CLASS_DECL,
                                 clang.cindex.CursorKind.STRUCT_DECL, clang.cindex.CursorKind.CLASS_TEMPLATE,
                                 clang.cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
                                 clang.cindex.CursorKind.CXX_METHOD, clang.cindex.CursorKind.CONSTRUCTOR,
                                 clang.cindex.CursorKind.DESTRUCTOR, clang.cindex.CursorKind.FUNCTION_DECL):

                    usr = self.get_usr(node)
                    async with self.declarations_lock:
                        self.all_declarations[usr].append((project_path, node.kind, node.location.line, node.location.column))
                        relationship_type = self.get_relationship_type(node, project_path)
                        self.declaration_types[usr][project_path] = relationship_type

                    # Process specific node types
                    if node.kind == clang.cindex.CursorKind.NAMESPACE:
                        await self.process_namespace_node(file_id=file_id, node=node, inputLocation=inputLocation, project_path=project_path, project_id=project_id, is_cpp_file=is_cpp_file)
                    elif node.kind in (clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL,
                                      clang.cindex.CursorKind.CLASS_TEMPLATE, clang.cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                        if node.is_definition():
                            await self.process_class_node(file_id=file_id, node=node, inputLocation=inputLocation, project_path=project_path, project_id=project_id, file_name=file_name, full_scope=full_scope, full_name=full_name, namespace=namespace, is_cpp_file=is_cpp_file)
                    elif node.kind in (clang.cindex.CursorKind.CXX_METHOD, clang.cindex.CursorKind.CONSTRUCTOR, clang.cindex.CursorKind.DESTRUCTOR):
                        await self.process_method_node(file_id, node, inputLocation, project_path, full_scope, namespace, is_cpp_file)
                    elif node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                        await self.process_function_node(file_id, node, inputLocation, project_path, full_scope, namespace, is_cpp_file)

        except Exception as e:
            logger.error(f"Error processing node {node.spelling} of kind {node.kind}: {e}")
            logger.error(f"Node details: kind={node.kind}, location={node.location}, type={node.type.spelling if node.type else 'None'}")
            
    
    def get_namespace_node(self, node):
        current = node.semantic_parent
        while current and current.kind != clang.cindex.CursorKind.TRANSLATION_UNIT:
            if current.kind == clang.cindex.CursorKind.NAMESPACE:
                return current
            current = current.semantic_parent
        return None

    async def process_namespace_node(self, file_id: str, node, inputLocation, project_path, project_id, is_cpp_file):
        file_name = node.location.file.name if node.location.file else ''
        if project_path not in file_name:
            logger.info(f"project_path not in file_name for namespace node: {node.spelling}, kind: {node.kind}, project_path: {project_path}, file_name: {file_name}")
        else:
            full_scope = self.get_full_scope(node)
            full_name = self.get_full_scope(node, include_self=True)
            namespace = self.get_namespace(node)

            description = f"Namespace {node.spelling} in scope {full_scope} defined in {project_path}"
            raw_comment = node.raw_comment if node.raw_comment else ''
            if raw_comment:
                description += f" with documentation: {raw_comment.strip()}"

            raw_code, start_line, end_line = await self.get_raw_code(node)

            usr = self.get_usr(node)
            if not usr:
                logger.warning(f"Skipping namespace {node.spelling} due to missing USR")
                return

            file_info = {
                'file_id': file_id,
                'file_path': project_path,
                'file_type': FileType.CppCode if is_cpp_file else FileType.CppHeader,
                'start_line': start_line,
                'end_line': end_line,
                'raw_code': raw_code,
                'raw_comment': raw_comment
            }

            details = {
                'type': 'Namespace',
                'name': full_name,
                'scope': full_scope,
                'short_name': node.spelling,
                'description': description,
                'parent_namespace': self.get_parent_namespace(node),
                'usr': usr,
                'files': [file_info]
            }
            await self.update_namespace(full_name, details)

        await self.process_child_nodes(file_id=file_id, node=node, inputLocation=inputLocation, project_path=project_path, project_id=project_id, is_cpp_file=is_cpp_file)
    

    def get_parent_namespace(self, node):
        parent = node.semantic_parent
        while parent and parent.kind != clang.cindex.CursorKind.NAMESPACE:
            parent = parent.semantic_parent
        return parent.spelling if parent else None


    async def process_class_node(self, file_id: str, node, inputLocation, project_path, project_id, file_name, full_scope, full_name, namespace, is_cpp_file):
        type_name = "Class" if node.kind == clang.cindex.CursorKind.CLASS_DECL else "Struct" if node.kind == clang.cindex.CursorKind.STRUCT_DECL else "ClassTemplate"
        class_name = node.spelling

        interface_description = ""
        implementation_description = ""

        description = f"{type_name} {class_name} in scope {full_scope} defined in {project_path}"
        raw_comment = node.raw_comment if node.raw_comment else ''
        if raw_comment:
            description += f" with documentation: {raw_comment.strip()}"

        # Base classes
        bases = [base for base in node.get_children() if base.kind == clang.cindex.CursorKind.CXX_BASE_SPECIFIER]
        if bases:
            base_names = [f"{base.spelling} in scope {self.get_full_scope(base.type.get_declaration())}" for base in bases]
            description += f". Inherits from: {', '.join(base_names)}"

        is_node_exported = await self.is_exported(node)
        if is_node_exported:
            description += ". (EXPORTED)"

        # Public members of the class
        members = []

        def get_public_members(node, inherited=False):
            for member in node.get_children():
                if member.access_specifier == clang.cindex.AccessSpecifier.PUBLIC:
                    origin = " (inherited)" if inherited else ""
                    export_status = "exported" if is_node_exported else "not exported"
                    if member.kind in (clang.cindex.CursorKind.CXX_METHOD, clang.cindex.CursorKind.CONSTRUCTOR, clang.cindex.CursorKind.DESTRUCTOR):
                        members.append(f"public method {member.spelling} in scope {self.get_full_scope(member)} ({export_status}){origin}")
                    elif member.kind == clang.cindex.CursorKind.FIELD_DECL:
                        members.append(f"public attribute {member.spelling} of type {member.type.spelling} in scope {self.get_full_scope(member)} ({export_status}){origin}")

        # Get public members of the class itself
        get_public_members(node, inherited=False)

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
                    if member.kind in (clang.cindex.CursorKind.CXX_METHOD, clang.cindex.CursorKind.CONSTRUCTOR, clang.cindex.CursorKind.DESTRUCTOR):
                        members.append(f"private method {member.spelling} in scope {self.get_full_scope(member)}{origin}")
                    elif member.kind == clang.cindex.CursorKind.FIELD_DECL:
                        members.append(f"private attribute {member.spelling} of type {member.type.spelling} in scope {self.get_full_scope(member)}{origin}")
                    elif member.kind == clang.cindex.CursorKind.VAR_DECL:
                        members.append(f"private static {member.spelling} of type {member.type.spelling} in scope {self.get_full_scope(member)}{origin}")
                elif member.access_specifier == clang.cindex.AccessSpecifier.PROTECTED and inherited:
                    if member.kind in (clang.cindex.CursorKind.CXX_METHOD, clang.cindex.CursorKind.CONSTRUCTOR, clang.cindex.CursorKind.DESTRUCTOR):
                        members.append(f"protected method {member.spelling} in scope {self.get_full_scope(member)}{origin}")
                    elif member.kind == clang.cindex.CursorKind.FIELD_DECL:
                        members.append(f"protected attribute {member.spelling} of type {member.type.spelling} in scope {self.get_full_scope(member)}{origin}")

        # Get implementation members of the class itself
        get_implementation_members(node, inherited=False)

        # Get protected members of base classes
        for base in bases:
            if base.type and base.type.get_declaration().kind == clang.cindex.CursorKind.CLASS_DECL:
                base_class = base.type.get_declaration()
                get_implementation_members(base_class, inherited=True)

        if members:
            implementation_description = "Implementation details: " + ", ".join(members)

        raw_code, start_line, end_line = await self.get_raw_code(node)
                
        # Update base classes information
        base_classes = []
        for base in bases:
            base_type = base.get_definition()
            if base_type:
                base_usr = self.get_usr(base)
                base_classes.append({
                    'usr': base_usr,
                    'name': self.get_full_scope(base_type, include_self=True),
                    'access_specifier': base.access_specifier.name  # Convert to string
                })

        usr = self.get_usr(node)
        if not usr:
            logger.warning(f"Skipping class {class_name} due to missing USR")
            return

        relationship_type = None
            
        async with self.declarations_lock:
            self.all_declarations[usr].append((project_path, node.kind, node.location.line, node.location.column))
            
            relationship_type = self.get_relationship_type(node, project_path)
            current_type = self.declaration_types[usr][project_path]
            
            if relationship_type == 'IMPLEMENTED_IN_FILE' or (relationship_type == 'DEFINED_IN_FILE' and current_type != 'IMPLEMENTED_IN_FILE'):
                self.declaration_types[usr][project_path] = relationship_type
            
            if node.is_definition():
                self.class_implementations[usr] = project_path
            elif project_path.endswith(('.hpp', '.h', '.hxx')):
                self.class_declarations[usr] = project_path
            
            relationship_type = self.declaration_types[usr][project_path]

        file_info = {
            'file_id': file_id,
            'file_path': project_path,
            'file_type': FileType.CppCode if is_cpp_file else FileType.CppHeader,
            'start_line': start_line,
            'end_line': end_line,
            'raw_code': raw_code,
            'raw_comment': node.raw_comment if node.raw_comment else '',
            'relationship_type': relationship_type
        }

        details = {
            'type': type_name,
            'name': full_name,
            'scope': full_scope,
            'short_name': class_name,
            'description': description,
            'interface_description': interface_description,
            'implementation_description': implementation_description,
            'namespace': namespace,
            'base_classes': base_classes,
            'files': [file_info],
            'usr': usr
        }
        await self.update_class(full_name, details)
        
        await self.process_child_nodes(file_id, node, inputLocation, project_path, project_id, is_cpp_file)


    async def process_method_node(self, file_id: str, node, inputLocation, project_path, full_scope, namespace, is_cpp_file):
        type_name = "Method"
        class_name = full_scope
        method_name = node.spelling
        fully_qualified_method_name = f"{full_scope}::{method_name}" if full_scope else method_name

        description = f"Method {method_name} in class {full_scope} defined in {project_path}"
        raw_comment = node.raw_comment if node.raw_comment else ''
        if raw_comment:
            description += f" with documentation: {raw_comment.strip()}"

        is_node_exported = await self.is_exported(node)
        if is_node_exported:
            description += ". (EXPORTED)"

        # Parameters and return type
        params = [f"{param.type.spelling} {param.spelling}" for param in node.get_arguments()]
        return_type = node.result_type.spelling
        description += f". Returns {return_type} and takes parameters: {', '.join(params)}"

        description += "."

        usr = self.get_usr(node)
        if not usr:
            logger.warning(f"Skipping method {method_name} due to missing USR")
            return

        relationship_type = None
        
        async with self.declarations_lock:
            self.all_declarations[usr].append((project_path, node.kind, node.location.line, node.location.column))
            
            relationship_type = self.get_relationship_type(node, project_path)
            current_type = self.declaration_types[usr][project_path]
            
            if relationship_type == 'IMPLEMENTED_IN_FILE' or (relationship_type == 'DECLARED_IN_FILE' and current_type != 'IMPLEMENTED_IN_FILE'):
                self.declaration_types[usr][project_path] = relationship_type
            
            if node.is_definition():
                self.method_implementations[usr] = project_path
            elif project_path.endswith(('.hpp', '.h', '.hxx')):
                self.method_declarations[usr] = project_path
                
            relationship_type = self.declaration_types[usr][project_path]

        raw_code, start_line, end_line = await self.get_raw_code(node)
        file_info = {
            'file_id': file_id,
            'file_path': project_path,
            'file_type': FileType.CppCode if is_cpp_file else FileType.CppHeader,
            'start_line': node.extent.start.line,
            'end_line': node.extent.end.line,
            'raw_code': raw_code,
            'raw_comment': raw_comment,
            'relationship_type': relationship_type
        }

        # Get the class USR
        class_node = node.semantic_parent
        class_usr = self.get_usr(class_node) if class_node.kind in [clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL] else None

        details = {
            'type': type_name,
            'name': fully_qualified_method_name,
            'scope': full_scope,
            'short_name': method_name,
            'return_type': return_type,
            'description': description,
            'namespace': namespace,
            'files': [file_info],
            'usr': usr,
            'class_usr': class_usr if class_usr is not None else ''
        }
        await self.update_method(fully_qualified_method_name, details)
    
    
        
    async def process_function_node(self, file_id: str, node, inputLocation, project_path, full_scope, namespace, is_cpp_file):
        type_name = "Function"
        function_name = node.spelling
        fully_qualified_function_name = f"{full_scope}::{function_name}" if full_scope else function_name

        description = f"Function {function_name} in scope {full_scope} defined in {project_path}"
        raw_comment = node.raw_comment if node.raw_comment else ''
        if raw_comment:
            description += f" with documentation: {raw_comment.strip()}"

        is_node_exported = await self.is_exported(node)
        if is_node_exported:
            description += ". (EXPORTED)"

        # Parameters and return type
        params = [f"{param.type.spelling} {param.spelling}" for param in node.get_arguments()]
        return_type = node.result_type.spelling
        description += f". Returns {return_type} and takes parameters: {', '.join(params)}"

        description += "."

        async with self.lock_functions:
            is_new_function = True if fully_qualified_function_name not in self.functions else False

        raw_code, start_line, end_line = await self.get_raw_code(node)

        usr = self.get_usr(node)
        if not usr:
            logger.warning(f"Skipping function {function_name} due to missing USR")
            return

        relationship_type = None
        
        async with self.declarations_lock:
            self.all_declarations[usr].append((project_path, node.kind, node.location.line, node.location.column))
            
            relationship_type = self.get_relationship_type(node, project_path)
            current_type = self.declaration_types[usr][project_path]
            
            if relationship_type == 'IMPLEMENTED_IN_FILE' or (relationship_type == 'DECLARED_IN_FILE' and current_type != 'IMPLEMENTED_IN_FILE'):
                self.declaration_types[usr][project_path] = relationship_type
            
            if node.is_definition():
                self.function_implementations[usr] = project_path
            elif project_path.endswith(('.hpp', '.h', '.hxx')):
                self.function_declarations[usr] = project_path
                
            relationship_type = self.declaration_types[usr][project_path]

        file_info = {
            'file_id': file_id,
            'file_path': project_path,
            'file_type': FileType.CppCode if is_cpp_file else FileType.CppHeader,
            'start_line': node.extent.start.line,
            'end_line': node.extent.end.line,
            'raw_code': raw_code,
            'raw_comment': raw_comment,
            'relationship_type': relationship_type
        }

        details = {
            'type': type_name,
            'name': fully_qualified_function_name,
            'scope': full_scope,
            'short_name': function_name,
            'description': description,
            'return_type': return_type,
            'namespace': namespace,
            'files': [file_info],
            'usr': usr
        }
        # Cache function information
        await self.update_function(fully_qualified_function_name, details)
        
        
    

    def get_full_scope(self, node, include_self=False):
        scopes = []
        current = node if include_self else node.semantic_parent
        while current and current.kind != clang.cindex.CursorKind.TRANSLATION_UNIT:
            if current.kind == clang.cindex.CursorKind.NAMESPACE or current.spelling:
                scopes.append(current.spelling)
            current = current.semantic_parent
        return "::".join(reversed(scopes))

    def get_namespace(self, node):
        scopes = []
        current = node.semantic_parent
        while current and current.kind != clang.cindex.CursorKind.TRANSLATION_UNIT:
            if current.kind == clang.cindex.CursorKind.NAMESPACE:
                scopes.append(current.spelling)
            current = current.semantic_parent
        return "::".join(reversed(scopes))


    async def get_raw_code(self, node):
        if node.extent.start.file and node.extent.end.file:
            file_name = node.extent.start.file.name
            start_line = node.extent.start.line
            end_line = node.extent.end.line
            start_offset = node.extent.start.offset
            end_offset = node.extent.end.offset

            try:
                async with aiofiles.open(file_name, 'r') as file:
                    await file.seek(start_offset)
                    raw_code = await file.read(end_offset - start_offset)
                return raw_code, start_line, end_line
            except Exception as e:
                logger.error(f"get_raw_code: Error reading file {file_name}: {e}")
                return '', start_line, end_line
        return '', 0, 0
    


    async def summarize_cpp_class(self, class_info) -> (str, str, str, str):
        try:
            class_name = class_info.get('name', 'Unknown')
            namespace = class_info.get('namespace', '')

            header_file_info = next((f for f in class_info.get('files', []) if f['file_type'] == FileType.CppHeader), None)
            impl_file_info = next((f for f in class_info.get('files', []) if f['file_type'] == FileType.CppCode), None)

            interface_info = class_info.copy()
            interface_info['raw_code'] = header_file_info['raw_code'] if header_file_info else 'Interface raw code unavailable. Summary should warn of this.'
            interface_info['raw_comment'] = header_file_info['raw_comment'] if header_file_info else 'Interface raw comment unavailable'

            implementation_info = class_info.copy()
            implementation_info['raw_code'] = impl_file_info['raw_code'] if impl_file_info else 'Implementation raw code unavailable. Summary should mention this.'
            implementation_info['raw_comment'] = impl_file_info['raw_comment'] if impl_file_info else 'Implementation raw comment unavailable'

            class_name, full_scope, interface_summary, interface_description = await self.summarize_cpp_class_public_interface(interface_info)
            implementation_summary, implementation_description = await self.summarize_cpp_class_implementation(implementation_info)

            return (
                class_name,
                full_scope,
                f"{interface_summary}\n\n{interface_description}",
                f"{implementation_summary}\n\n{implementation_description}"
            )
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class for {class_name}: {e}")
            logger.error(f"Class info: {class_info}")
            raise
        

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
            raw_code = class_info.get('raw_code', 'Interface raw code unavailable. Summary should warn of this.')
            file_path = class_info.get('file_path', '')
            raw_comment = class_info.get('raw_comment', 'Interface raw comment unavailable.')
            description = class_info.get('description', '')
            interface_description = class_info.get('interface_description', raw_code)

            background = (
                "Speaking as a senior developer and software architect, describe the purpose and usage of this class in your own words. "
                "Meditate on the provided description and public interface first, before writing your final summary. "
                "Then enclose those meditations in opening and closing <thoughts> tags, and then write your final summary."
            )
            
            # Combine background and description with clear separation
            full_prompt = f"{background}\n\nClass Details:\n{description}. {interface_description}"
#            logger.info(f"Public interface prompt for {class_info.get('name', 'Unknown')}: {full_prompt[:100]}...")  # Log first 100 chars
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_public_interface accessing dict: {e}")
            raise DatabaseError(f"Error in summarize_cpp_class_public_interface accessing dict: {e}")

        try:
            summary = await self.do_summarize_text(full_prompt, 200, 25)
#            logger.info(f"Public interface summary for {class_info.get('name', 'Unknown')}: {summary[:100]}...")  # Log first 100 chars
            return full_name, full_scope, summary, description + ". " + interface_description
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_public_interface for {class_info.get('name', 'Unknown')}: {e}")
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
            raw_code = class_info.get('raw_code', 'Implementation raw code unavailable.')
            file_path = class_info.get('file_path', '')
            raw_comment = class_info.get('raw_comment', 'Implementation raw comment unavailable.')
            description = class_info.get('description', '')
            implementation_description = class_info.get('implementation_description', raw_code)

            background = (
                "Speaking as a senior developer and software architect, describe the implementation and inner workings of this class in your own words. "
                "Meditate on the provided description below, before writing your final summary. "
                "Then enclose those meditations in opening and closing <thoughts> tags, and then write your final summary."
            )
            
            # Combine background and description with clear separation
            full_prompt = f"{background}\n\nClass Details: {description}. {implementation_description}"
#            logger.info(f"Implementation prompt for {class_info.get('name', 'Unknown')}: {full_prompt[:100]}...")  # Log first 100 chars
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_implementation accessing dict: {e}")
            raise DatabaseError(f"Error in summarize_cpp_class_implementation accessing dict: {e}")

        try:
            summary = await self.do_summarize_text(full_prompt, 200, 25)
        except Exception as e:
            logger.error(f"Error 1 in summarize_cpp_class_implementation for {class_info.get('name', 'Unknown')}: {e}")
            raise DatabaseError(f"Error in summarize_cpp_class_implementation: {e}")

        try:
#            logger.info(f"Implementation summary for {class_info.get('name', 'Unknown')}: {summary[:100]}...")  # Log first 100 chars
            return summary, description + ". " + implementation_description

        except Exception as e:
            logger.error(f"Error 2 in summarize_cpp_class_implementation for {class_info.get('name', 'Unknown')}: {e}")
            
            logger.info(f"raw_code:\n{pprint.pformat(raw_code)}")
            logger.info(f"Implementation Description:\n{pprint.pformat(implementation_description)}")

            raise DatabaseError(f"Error in summarize_cpp_class_implementation: {e}")



    
    async def summarize_cpp_function(self, node_info) -> (str, str, str):
        """
        Summarize a C++ function.

        Args:
            node_info: The function or method node's info.

        Returns:
            str: The summary of the function or method.
        """
        longest_file_info = max(node_info.get('files', []), key=lambda x: len(x['raw_code']))
        
        type_name = node_info.get('type', 'Function')
        full_name = node_info.get('name', 'anonymous')
        full_scope = node_info.get('scope', '')
        short_name = node_info.get('short_name', '')
        raw_code = longest_file_info.get('raw_code', 'Function code unavailable.')
        file_path = longest_file_info['file_path']
        raw_comment = longest_file_info.get('raw_comment', 'Function code comment unavailable.')
        description = node_info.get('description', '')

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
#            logger.info(f"Finished summarization for function: {node_info.get('name', 'Unknown')}. "
#                         f"Summary length: {len(summary)}")
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function appending strings: {e}\nnode_info contents:\n{pprint.pformat(node_info, indent=4)}")
            raise DatabaseError(f"Error in summarize_cpp_function appending strings: {e}\nnode_info contents:\n{pprint.pformat(node_info, indent=4)}")

        return retval


    async def get_cpp_count(self):
        async with self.lock_namespaces:
            len_namespaces = len(self.namespaces)
        async with self.lock_classes:
            len_classes = len(self.classes)
        async with self.lock_methods:
            len_methods = len(self.methods)
        async with self.lock_functions:
            len_functions = len(self.functions)
        return len_classes + len_methods + len_functions + len_namespaces














