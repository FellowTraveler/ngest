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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CppProcessor:
    def __init__(self, do_summarize_text):
        self.failed_nodes = deque()
        self.max_retries = 3
        self.retry_semaphore = asyncio.Semaphore(5)  # Limit concurrent retries

        self.do_summarize_text = do_summarize_text
        
        self.classes = defaultdict(dict)
        self.methods = defaultdict(dict)
        self.functions = defaultdict(dict)
        self.header_files = {}
        
        self.lock_classes = asyncio.Lock()
        self.lock_methods = asyncio.Lock()
        self.lock_functions = asyncio.Lock()
        self.lock_header_files = asyncio.Lock()

    async def update_class(self, full_name, details):
        async with self.lock_classes:
            existing_info = self.classes.get(full_name, {})
            if not existing_info or (details.get('is_cpp_file', False) and not existing_info.get('is_cpp_file', False)):
                self.classes[full_name] = details
            else:
                merged_details = {**existing_info, **details}
                if details.get('is_cpp_file', False):
                    merged_details['raw_code'] = details.get('raw_code', existing_info.get('raw_code', ''))
                self.classes[full_name] = merged_details

    async def update_method(self, full_name, details):
        async with self.lock_methods:
            existing_info = self.methods.get(full_name, {})
            if not existing_info or (details.get('is_cpp_file', False) and not existing_info.get('is_cpp_file', False)):
                self.methods[full_name] = details
            else:
                merged_details = {**existing_info, **details}
                if details.get('is_cpp_file', False):
                    merged_details['raw_code'] = details.get('raw_code', existing_info.get('raw_code', ''))
                self.methods[full_name] = merged_details

    async def update_function(self, full_name, details):
        async with self.lock_functions:
            existing_info = self.functions.get(full_name, {})
            if not existing_info or (details.get('is_cpp_file', False) and not existing_info.get('is_cpp_file', False)):
                self.functions[full_name] = details
            else:
                merged_details = {**existing_info, **details}
                if details.get('is_cpp_file', False):
                    merged_details['raw_code'] = details.get('raw_code', existing_info.get('raw_code', ''))
                self.functions[full_name] = merged_details

    async def process_cpp_file(self, inputPath: str, inputLocation: str, project_id: str):
        logger.info(f"Starting to process C++ file: {inputPath}")
        try:
            logger.debug("Creating Clang Index")
            index = clang.cindex.Index.create()
            
            logger.debug(f"Attempting to parse file: {inputPath}")
            translation_unit = index.parse(inputPath)
            
            if translation_unit is None:
                logger.error(f"Failed to parse {inputPath}. Translation unit is None.")
                return

            logger.info(f"Successfully parsed {inputPath}")

            logger.debug(f"Reading contents of file: {inputPath}")
            try:
                with open(inputPath, 'r') as file:
                    file_contents = file.read()
                logger.debug(f"Successfully read {len(file_contents)} characters from {inputPath}")
            except IOError as io_err:
                logger.error(f"IOError while reading {inputPath}: {io_err}")
                raise

            # Save raw code of header files
            if inputPath.endswith('.hpp') or inputPath.endswith('.h'):
                logger.debug(f"Saving contents of header file: {inputPath}")
                async with self.lock_header_files:
                    self.header_files[inputPath] = file_contents
                logger.debug(f"Saved contents of header file: {inputPath}")

            logger.debug(f"Starting to process nodes for {inputPath}")
            await self.process_nodes(translation_unit.cursor, inputLocation, project_id, inputPath.endswith('.cpp'))
            logger.info(f"Finished processing nodes for {inputPath}")

            # After processing all nodes, retry any failed nodes
            await self.retry_failed_nodes(inputLocation, project_id)

        except clang.cindex.TranslationUnitLoadError as tu_error:
            logger.error(f"TranslationUnitLoadError for {inputPath}: {tu_error}")
            logger.error(f"Clang diagnostics: {[diag.spelling for diag in translation_unit.diagnostics]}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing C++ file {inputPath}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
    async def prepare_summarization_tasks(self):
        tasks = []
        async with self.lock_classes:
            for class_name, class_info in self.classes.items():
                name = class_info.get('name', 'anonymous')
                class_info['name'] = name
                info_copy = copy.deepcopy(class_info)
                tasks.append(('Class', name, info_copy))

        async with self.lock_methods:
            for method_name, method_info in self.methods.items():
                name = method_info.get('name', 'anonymous')
                method_info['name'] = name
                info_copy = copy.deepcopy(method_info)
                tasks.append(('Method', name, info_copy))

        async with self.lock_functions:
            for function_name, function_info in self.functions.items():
                name = function_info.get('name', 'anonymous')
                function_info['name'] = name
                info_copy = copy.deepcopy(function_info)
                tasks.append(('Function', name, info_copy))

        return tasks
    
    
    async def retry_failed_nodes(self, project_path, project_id):
        retry_tasks = []
        while self.failed_nodes:
            node, _, _, is_cpp_file, retry_count = self.failed_nodes.popleft()
            if retry_count >= self.max_retries:
                logger.warning(f"Node {node.spelling} has reached max retries. Skipping.")
                continue
            retry_tasks.append(self.retry_single_node(node, project_path, project_id, is_cpp_file, retry_count))

        await asyncio.gather(*retry_tasks)

        if self.failed_nodes:
            logger.warning(f"{len(self.failed_nodes)} nodes still failed after retries")


    async def retry_single_node(self, node, project_path, project_id, is_cpp_file, retry_count):
        async with self.retry_semaphore:
            try:
                logger.info(f"Retrying node {node.spelling} (attempt {retry_count + 1})")
                await self.process_nodes(node, project_path, project_id, is_cpp_file)
            except Exception as e:
                logger.error(f"Error retrying node {node.spelling}: {e}")
                self.failed_nodes.append((node, project_path, project_id, is_cpp_file, retry_count + 1))
                
    async def process_nodes(self, node, project_path, project_id, is_cpp_file):
        try:
            # Process the current node
            await self.process_single_node(node, project_path, project_id, is_cpp_file)
            
            # Recursively process child nodes
            for child in node.get_children():
                if project_path in child.location.file.name:
                    await self.process_nodes(child, project_path, project_id, is_cpp_file)
        except Exception as e:
            logger.error(f"Error processing node {node.spelling}: {e}")
            self.failed_nodes.append((node, project_path, project_id, is_cpp_file, 0))

    async def process_single_node(self, node, project_path, project_id, is_cpp_file):
        def is_exported(node):
            for token in node.get_tokens():
                if token.spelling in ["__declspec(dllexport)", "__attribute__((visibility(\"default\")))"]:
                    return True
            return False
        try:
            for child in node.get_children():
                file_name = child.location.file.name if child.location.file else ''
                if project_path in file_name:
                    if child.kind in (clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL,
                                      clang.cindex.CursorKind.CLASS_TEMPLATE, clang.cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                        type_name = "Class" if child.kind == clang.cindex.CursorKind.CLASS_DECL else "Struct" if child.kind == clang.cindex.CursorKind.STRUCT_DECL else "ClassTemplate"
                        class_name = child.spelling
                        full_scope = self.get_full_scope(child)
                        fully_qualified_name = f"{full_scope}::{class_name}" if full_scope else class_name

                        logger.debug(f"Processing {type_name}: {fully_qualified_name} in file {file_name}")

                        interface_description = ""
                        implementation_description = ""

                        description = f"{type_name} {class_name} in scope {full_scope} defined in {file_name}"
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
                        raw_code = await self.get_raw_code(child) if is_cpp_file else ''
                        details = {
                            'type': type_name,
                            'name': fully_qualified_name,
                            'scope': full_scope,
                            'short_name': class_name,
                            'description': description,
                            'interface_description': interface_description,
                            'implementation_description': implementation_description,
                            'raw_comment': raw_comment,
                            'file_path': file_name,
                            'is_cpp_file': is_cpp_file,
                            'raw_code': raw_code
                        }
                        await self.update_class(fully_qualified_name, details)
                        logger.debug(f"Finished processing {type_name}: {fully_qualified_name}")

                    elif child.kind == clang.cindex.CursorKind.CXX_METHOD:
                        logger.debug(f"Processing method: {child.spelling} in file {file_name}")
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
                            raw_code = await self.get_raw_code(child) if is_cpp_file else ''
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
                                'raw_code': raw_code
                            }
                            await self.update_method(fully_qualified_method_name, details)
                        logger.debug(f"Finished processing method: {fully_qualified_method_name}")

                    elif child.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                        logger.debug(f"Processing function: {child.spelling} in file {file_name}")
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
                            raw_code = await self.get_raw_code(child) if is_cpp_file else ''
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
                                'raw_code': raw_code
                            }
                            # Cache function information
                            await self.update_function(fully_qualified_function_name, details)
                        logger.debug(f"Finished processing function: {fully_qualified_function_name}")
                    else:
                        await self.process_nodes(child, project_path, project_id, is_cpp_file)
        except Exception as e:
            logger.error(f"Error processing node {node.spelling}: {e}")
            logger.error(f"Node details: kind={node.kind}, location={node.location}, type={node.type.spelling if node.type else 'None'}")
            self.failed_nodes.append((node, project_path, project_id, is_cpp_file, 0))
            
        
    async def update_classes(self, full_name, details):
        async with self.lock_classes:
            self.classes[full_name].update(details)

    async def update_methods(self, full_name, details):
        async with self.lock_methods:
            self.methods[full_name].update(details)

    async def update_functions(self, full_name, details):
        async with self.lock_functions:
            self.functions[full_name].update(details)
            
    def get_full_scope(self, node):
        scopes = []
        current = node.semantic_parent
        while current and current.kind != clang.cindex.CursorKind.TRANSLATION_UNIT:
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
        """
        Summarize a C++ class.

        Args:
            class_info: The class node's info.

        Returns:
            str: The summary of the class.
        """
        logger.debug(f"Starting summarization for class: {class_info.get('name', 'Unknown')}")
        try:
            class_name, full_scope, interface_summary, interface_description = await self.summarize_cpp_class_public_interface(class_info)
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_public_interface for {class_name}: {e}")
            raise DatabaseError(f"Error in summarize_cpp_class_public_interface for {class_name}: {e}")
        try:
            implementation_summary, implementation_description = await self.summarize_cpp_class_implementation(class_info)
            logger.debug(f"Finished summarization for class: {class_info.get('name', 'Unknown')}. "
                         f"Interface summary length: {len(interface_summary)}, "
                         f"Implementation summary length: {len(implementation_summary)}")
        except Exception as e:
            logger.error(f"Error in summarize_cpp_class_implementationfor {class_name}: {e}")
            raise DatabaseError(f"Error in summarize_cpp_class_implementation for {class_name}: {e}")

        return class_name, full_scope, interface_summary + "\n\n" + interface_description, implementation_summary + "\n\n" + implementation_description

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


    async def summarize_cpp_function(self, node_info) -> (str, str, str):
        """
        Summarize a C++ function.

        Args:
            node_info: The function or method node's info.

        Returns:
            str: The summary of the function or method.
        """
        logger.debug(f"Starting summarization for function: {node_info.get('name', 'Unknown')}")

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
            logger.debug(f"Finished summarization for function: {node_info.get('name', 'Unknown')}. "
                         f"Summary length: {len(summary)}")
        except Exception as e:
            logger.error(f"Error in summarize_cpp_function appending strings: {e}\nnode_info contents:\n{pprint.pformat(node_info, indent=4)}")
            raise DatabaseError(f"Error in summarize_cpp_function appending strings: {e}\nnode_info contents:\n{pprint.pformat(node_info, indent=4)}")

        return retval


    async def get_cpp_count(self):
        async with self.lock_classes:
            len_classes = len(self.classes)
        async with self.lock_methods:
            len_methods = len(self.methods)
        async with self.lock_functions:
            len_functions = len(self.functions)
        return len_classes + len_methods + len_functions
