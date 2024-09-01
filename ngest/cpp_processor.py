# Copyright 2024 Chris Odom
# MIT License

import logging
import traceback
import asyncio
from collections import deque
from typing import Dict, Any
import copy
import pprint
from collections import defaultdict
import aiofiles

import ngest
from ngest.custom_errors import FileProcessingError, DatabaseError

import clang.cindex

logging.basicConfig(level=logging.info, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

    async def update_namespace(self, full_name, details):
        async with self.lock_namespaces:
            existing_info = self.namespaces.get(full_name, {})
            merged_details = existing_info.copy()
            for key, value in details.items():
                if value or isinstance(value, (int, float)):
                    merged_details[key] = value
            self.namespaces[full_name] = merged_details

    async def update_class(self, full_name, details):
        async with self.lock_classes:
            existing_info = self.classes.get(full_name, {})
            merged_details = existing_info.copy()
            for key, value in details.items():
                if key == 'raw_code':
                    if (details.get('is_cpp_file', False) or
                        not merged_details.get('raw_code') or
                        len(value) > len(merged_details['raw_code'])):
                        merged_details['raw_code'] = value
                elif value or isinstance(value, (int, float)):
                    merged_details[key] = value
            self.classes[full_name] = merged_details

    async def update_method(self, full_name, details):
        async with self.lock_methods:
            existing_info = self.methods.get(full_name, {})
            merged_details = existing_info.copy()
            for key, value in details.items():
                if key == 'raw_code':
                    if (details.get('is_cpp_file', False) or
                        not merged_details.get('raw_code') or
                        len(value) > len(merged_details['raw_code'])):
                        merged_details['raw_code'] = value
                elif value or isinstance(value, (int, float)):
                    merged_details[key] = value
            self.methods[full_name] = merged_details

    async def update_function(self, full_name, details):
        async with self.lock_functions:
            existing_info = self.functions.get(full_name, {})
            merged_details = existing_info.copy()
            for key, value in details.items():
                if key == 'raw_code':
                    if (details.get('is_cpp_file', False) or
                        not merged_details.get('raw_code') or
                        len(value) > len(merged_details['raw_code'])):
                        merged_details['raw_code'] = value
                elif value or isinstance(value, (int, float)):
                    merged_details[key] = value
            self.functions[full_name] = merged_details
        
    async def parse_cpp_file(self, inputPath: str, inputLocation: str, project_id: str):
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
            await self.process_nodes(translation_unit.cursor, inputLocation, project_id, inputPath.endswith('.cpp'))
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
                
                # Create a simplified version of class_info for logging
                simplified_info = {
                    'type': class_info.get('type'),
                    'name': class_info.get('name'),
                    'scope': class_info.get('scope'),
                    'short_name': class_info.get('short_name'),
                    'has_description': 'description' in class_info,
                    'has_interface_description': 'interface_description' in class_info,
                    'has_implementation_description': 'implementation_description' in class_info,
                    'has_raw_comment': 'raw_comment' in class_info,
                    'file_path': class_info.get('file_path'),
                    'is_cpp_file': class_info.get('is_cpp_file'),
                    'has_raw_code': 'raw_code' in class_info,
                    'has_interface_embedding': 'interface_embedding' in class_info,
                    'has_implementation_embedding': 'implementation_embedding' in class_info
                }
                
#                logger.info(f"Simplified class info: {simplified_info}")
                
                if 'interface_description' in class_info or 'implementation_description' in class_info:
                    tasks.append(('Class', full_name, copy.deepcopy(class_info)))
#                    logger.info(f"Added summarization task for class: {full_name}")
                else:
                    logger.warning(f"Skipping summarization for class {full_name} in namespace {namespace}. "
                                   f"Missing one of: interface_description: {'interface_description' in class_info}, "
                                   f"implementation_description: {'implementation_description' in class_info}")
                                   
        async with self.lock_methods:
            logger.info(f"Preparing summarization tasks for methods. Total methods: {len(self.methods)}")
            for full_name, method_info in self.methods.items():
                namespace = method_info.get('namespace', '')
                simplified_info = {
                    'type': method_info.get('type'),
                    'name': method_info.get('name'),
                    'scope': method_info.get('scope'),
                    'short_name': method_info.get('short_name'),
                    'has_description': 'description' in method_info,
                    'has_raw_comment': 'raw_comment' in method_info,
                    'file_path': method_info.get('file_path'),
                    'is_cpp_file': method_info.get('is_cpp_file'),
                    'has_raw_code': 'raw_code' in method_info,
                    'has_embedding': 'embedding' in method_info
                }
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
                simplified_info = {
                    'type': function_info.get('type'),
                    'name': function_info.get('name'),
                    'scope': function_info.get('scope'),
                    'short_name': function_info.get('short_name'),
                    'has_description': 'description' in function_info,
                    'has_raw_comment': 'raw_comment' in function_info,
                    'file_path': function_info.get('file_path'),
                    'is_cpp_file': function_info.get('is_cpp_file'),
                    'has_raw_code': 'raw_code' in function_info,
                    'has_embedding': 'embedding' in function_info
                }
#                logger.info(f"Checking function: {full_name}, namespace: {namespace}")
#                logger.info(f"Simplified function info: {simplified_info}")
                
                if 'description' in function_info:
                    tasks.append(('Function', full_name, copy.deepcopy(function_info)))
#                    logger.info(f"Added summarization task for function: {full_name}")
                else:
                    logger.warning(f"Skipping summarization for function {full_name} in namespace {namespace} due to missing description")

        logger.info(f"Total summarization tasks prepared: {len(tasks)}")
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


    async def retry_single_node(self, node, project_path, project_id, is_cpp_file, retry_count):
        async with self.retry_semaphore:
            try:
                logger.info(f"Retrying node {node.spelling} (attempt {retry_count + 1})")
                await self.process_single_node(node, project_path, project_id, is_cpp_file)
            except Exception as e:
                logger.error(f"Error retrying node {node.spelling}: {e}")
                self.failed_nodes.append((node, project_path, project_id, is_cpp_file, retry_count + 1))
                
                
    async def process_nodes(self, node, project_path, project_id, is_cpp_file):
        try:
            if node.kind == clang.cindex.CursorKind.TRANSLATION_UNIT:
                # For the translation unit, process all children without checking the file path
                for child in node.get_children():
                    await self.process_nodes(child, project_path, project_id, is_cpp_file)
            else:
                # Process the current node
                await self.process_single_node(node=node, project_path=project_path, project_id=project_id, is_cpp_file=is_cpp_file)

                # Process child nodes
                if node.kind not in (clang.cindex.CursorKind.NAMESPACE, clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL, clang.cindex.CursorKind.CLASS_TEMPLATE, clang.cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                    await self.process_child_nodes(node=node, project_path=project_path, project_id=project_id, is_cpp_file=is_cpp_file)
        except Exception as e:
            logger.error(f"Error processing node {node.spelling}: {e}")
            self.failed_nodes.append((node, project_path, project_id, is_cpp_file, 0))

    async def process_child_nodes(self, node, project_path, project_id, is_cpp_file):
        for child in node.get_children():
            if child.location.file and project_path in child.location.file.name:
#                logger.info(f"Processing child node {child.spelling}, kind: {child.kind}")
                await self.process_nodes(node=child, project_path=project_path, project_id=project_id, is_cpp_file=is_cpp_file)

    async def is_exported(self, node):
        for token in node.get_tokens():
            if token.spelling in ["__declspec(dllexport)", "__attribute__((visibility(\"default\")))"]:
                return True
        return False

    async def process_single_node(self, node, project_path, project_id, is_cpp_file):
#        logger.info(f"Processing node: {node.spelling}, kind: {node.kind}")
        try:
            file_name = node.location.file.name if node.location.file else ''
            if node.location.file and project_path in node.location.file.name:
                full_scope = self.get_full_scope(node)
                full_name = self.get_full_scope(node, include_self=True)
                namespace = self.get_namespace(node)

                if node.kind == clang.cindex.CursorKind.NAMESPACE:
#                    logger.info(f"Processing namespace: {node.spelling}")
                    await self.process_namespace_node(node=node, project_path=project_path, project_id=project_id, is_cpp_file=is_cpp_file)
#                    logger.info(f"Finished processing namespace: {node.spelling}")
                elif node.kind in (clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL,
                                  clang.cindex.CursorKind.CLASS_TEMPLATE, clang.cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                    await self.process_class_node(node=node, project_path=project_path, project_id=project_id, file_name=file_name, full_scope=full_scope, full_name=full_name, namespace=namespace, is_cpp_file=is_cpp_file)
                elif node.kind in (clang.cindex.CursorKind.CXX_METHOD, clang.cindex.CursorKind.CONSTRUCTOR, clang.cindex.CursorKind.DESTRUCTOR):
                    await self.process_method_node(node, file_name, full_scope, namespace, is_cpp_file)
                elif node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                    await self.process_function_node(node, file_name, full_scope, namespace, is_cpp_file)
#                else:
#                    logger.info(f"Unhandled node type: {node.kind} for node: {node.spelling}")
#                    await self.process_child_nodes(node, project_path, project_id, is_cpp_file)
        except Exception as e:
            logger.error(f"Error processing node {node.spelling}: {e}")
            logger.error(f"Node details: kind={node.kind}, location={node.location}, type={node.type.spelling if node.type else 'None'}")
            self.failed_nodes.append((node, project_path, project_id, is_cpp_file, 0))


    async def process_namespace_node(self, node, project_path, project_id, is_cpp_file):
#        logger.info(f"Processing namespace: {node.spelling}")
        file_name = node.location.file.name if node.location.file else ''
        if project_path not in file_name:
            logger.info(f"project_path not in file_name for namespace node: {node.spelling}, kind: {node.kind}, project_path: {project_path}, file_name: {file_name}")
        else:
            full_scope = self.get_full_scope(node)
            full_name = self.get_full_scope(node, include_self=True)
            namespace = self.get_namespace(node)

            description = f"Namespace {node.spelling} in scope {full_scope} defined in {file_name}"
            raw_comment = node.raw_comment if node.raw_comment else ''
            if raw_comment:
                description += f" with documentation: {raw_comment.strip()}"

            async with self.lock_namespaces:
                is_new_namespace = True if full_name not in self.namespaces else False

            if is_new_namespace:
                details = {
                    'type': 'Namespace',
                    'name': full_name,
                    'scope': full_scope,
                    'short_name': node.spelling,
                    'description': description,
                    'raw_comment': raw_comment,
                    'file_path': file_name,
                    'namespace': namespace
                }
                await self.update_namespace(full_name, details)

            # Process the namespace's children
            await self.process_child_nodes(node=node, project_path=project_path, project_id=project_id, is_cpp_file=is_cpp_file)
            
#        logger.info(f"Finished processing namespace: {node.spelling}")
    
    async def process_class_node(self, node, project_path, project_id, file_name, full_scope, full_name, namespace, is_cpp_file):
        type_name = "Class" if node.kind == clang.cindex.CursorKind.CLASS_DECL else "Struct" if node.kind == clang.cindex.CursorKind.STRUCT_DECL else "ClassTemplate"
        class_name = node.spelling

#        logger.info(f"Processing {type_name}: {full_name} in file {file_name}")

        interface_description = ""
        implementation_description = ""

        description = f"{type_name} {class_name} in scope {full_scope} defined in {file_name}"
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

#        async with self.lock_header_files:
#            header_code = self.header_files.get(file_name, '')
        raw_code = await self.get_raw_code(node) if is_cpp_file else ''
        details = {
            'type': type_name,
            'name': full_name,
            'scope': full_scope,
            'short_name': class_name,
            'description': description,
            'interface_description': interface_description,
            'implementation_description': implementation_description,
            'raw_comment': raw_comment,
            'file_path': file_name,
            'is_cpp_file': is_cpp_file,
            'raw_code': raw_code,
            'namespace': namespace
        }
        await self.update_class(full_name, details)
                    
        await self.process_child_nodes(node, project_path, project_id, is_cpp_file)
#        logger.info(f"Finished processing {type_name}: {full_name} in file {file_name}")

    async def process_method_node(self, node, file_name, full_scope, namespace, is_cpp_file):
        type_name = "Method"
        class_name = full_scope
        method_name = node.spelling
        fully_qualified_method_name = f"{class_name}::{method_name}" if class_name else method_name

#        logger.info(f"Processing {type_name}: {fully_qualified_method_name} in file {file_name}")

        description = f"Method {method_name} in class {full_scope} defined in {file_name}"
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

        async with self.lock_methods:
            is_new_method = True if fully_qualified_method_name not in self.methods else False

        # Cache method information, prioritize CPP file
        if is_cpp_file or is_new_method:
            raw_code = await self.get_raw_code(node) if is_cpp_file else ''
            # Cache method information
            details = {
                'type': type_name,
                'name': fully_qualified_method_name,
                'scope': full_scope,
                'short_name': method_name,
                'return_type': return_type,
                'description': description,
                'raw_comment': raw_comment,
                'file_path': file_name,
                'is_cpp_file': is_cpp_file,
                'raw_code': raw_code,
                'namespace': namespace
            }
            await self.update_method(fully_qualified_method_name, details)
#        logger.info(f"Finished processing method: {node.spelling} in file {file_name} full_name: {fully_qualified_method_name}")

    async def process_function_node(self, node, file_name, full_scope, namespace, is_cpp_file):
        type_name = "Function"
        function_name = node.spelling
        fully_qualified_function_name = f"{full_scope}::{function_name}" if full_scope else function_name

#        logger.info(f"Processing {type_name}: {fully_qualified_function_name} in file {file_name}")

        description = f"Function {function_name} in scope {full_scope} defined in {file_name}"
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

        # Store function information, prioritize CPP file
        if is_cpp_file or is_new_function:
            raw_code = await self.get_raw_code(node) if is_cpp_file else ''
            details = {
                'type': type_name,
                'name': fully_qualified_function_name,
                'scope': full_scope,
                'short_name': function_name,
                'description': description,
                'return_type': return_type,
                'raw_comment': raw_comment,
                'file_path': file_name,
                'is_cpp_file': is_cpp_file,
                'raw_code': raw_code,
                'namespace': namespace
            }
            # Cache function information
            await self.update_function(fully_qualified_function_name, details)
#        logger.info(f"Finished processing function: {node.spelling} in file {file_name} full_name: {fully_qualified_function_name}")


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
            start_offset = node.extent.start.offset
            end_offset = node.extent.end.offset

            try:
                async with aiofiles.open(file_name, 'r') as file:
                    await file.seek(start_offset)
                    raw_code = await file.read(end_offset - start_offset)
                return raw_code
            except Exception as e:
                logger.error(f"get_raw_code: Error reading file {file_name}: {e}")
                return ''
        return ''

    async def summarize_cpp_class(self, class_info) -> (str, str, str, str):
        try:
            class_name = class_info.get('name', 'Unknown')
            namespace = class_info.get('namespace', '')
#            logger.info(f"Summarizing class: {class_info['name']}, interface_description: {class_info.get('interface_description', 'Missing')}, implementation_description: {class_info.get('implementation_description', 'Missing')}")

            class_name, full_scope, interface_summary, interface_description = await self.summarize_cpp_class_public_interface(class_info)
#            logger.info(f"Completed public interface summarization for {class_name}")
            
            implementation_summary, implementation_description = await self.summarize_cpp_class_implementation(class_info)
#            logger.info(f"Completed implementation summarization for {class_name}")
            
#            logger.info(f"Finished summarization for class: {class_name}, interface_summary length: {len(interface_summary)}, implementation_summary length: {len(implementation_summary)}")
            
            return class_name, full_scope, interface_summary + "\n\n" + interface_description, implementation_summary + "\n\n" + implementation_description
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
#            logger.info(f"Implementation prompt for {class_info.get('name', 'Unknown')}: {full_prompt[:100]}...")  # Log first 100 chars
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_implementation accessing dict: {e}")
            raise DatabaseError(f"Error in summarize_cpp_class_implementation accessing dict: {e}")

        try:
            summary = await self.do_summarize_text(full_prompt, 200, 25)
#            logger.info(f"Implementation summary for {class_info.get('name', 'Unknown')}: {summary[:100]}...")  # Log first 100 chars
            return summary, description + ". " + implementation_description

        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_implementation for {class_info.get('name', 'Unknown')}: {e}")
            raise DatabaseError(f"Error in summarize_cpp_class_implementation: {e}")


    async def summarize_cpp_function(self, node_info) -> (str, str, str):
        """
        Summarize a C++ function.

        Args:
            node_info: The function or method node's info.

        Returns:
            str: The summary of the function or method.
        """
#        logger.info(f"Starting summarization for function: {node_info.get('name', 'Unknown')}")

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
