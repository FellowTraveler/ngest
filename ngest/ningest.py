# Copyright 2024 Chris Odom
# MIT License

import os
import asyncio
import logging
import shutil
import fnmatch
from typing import List, Dict, Any
from neo4j import AsyncGraphDatabase
from tqdm.asyncio import tqdm

from ngest.base_importer import NBaseImporter
from ngest.neo4j_importer import NNeo4JImporter
from ngest.project import Project

logger = logging.getLogger(__name__)

class NIngest:
    """
    Manages the ingestion process, counting files, handling directories, and updating or deleting projects.
    """
    def __init__(self, project_id: str = None, importer: NBaseImporter = NNeo4JImporter()):
        self.project_id = project_id
        
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
        self.start_ingest_semaphore = asyncio.Semaphore(1)

        self.finalize_semaphore = asyncio.Semaphore(1)
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

        if project_id is not None:
            self.currentOutputPath = os.path.expanduser(f"~/.ngest/projects/{self.project_id}")
            if not os.path.exists(self.currentOutputPath):
                os.makedirs(self.currentOutputPath, exist_ok=True)
                open(os.path.join(self.currentOutputPath, '.ngest_index'), 'a').close()
                logger.info(f"Created new project directory at {self.currentOutputPath}")
        else:
            self.currentOutputPath = None
            
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
                    self.progress_bar_scan = tqdm(total=total, desc="Parsing / Ingesting", unit="files")
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

            try:
                async with self.importer_.get_session() as session:
                    project = await Project.create_in_database(session, self.project_id, project_root, description=f"Project created from {inputPath}")
                    if not project:
                        logger.error("Failed to create project in database")
                        return -1

                self.gitignore_patterns = self.load_gitignore_patterns(inputPath)

                if not os.path.exists(self.currentOutputPath):
                    os.makedirs(self.currentOutputPath, exist_ok=True)
                    open(os.path.join(self.currentOutputPath, '.ngest_index'), 'a').close()
                    logger.info(f"Created new project ingestion directory at {self.currentOutputPath}")

                self.total_files = await self.count_files(inputPath)
                result = 0

                await self.start_progress_scan(self.total_files)
                result = await self.RecursiveParse(inputPath, project_input_location, topLevelOutputPath, self.currentOutputPath)
                
                if result == 0:
                    async with self.summarize_semaphore:
                        await self.start_progress_summarize(self.total_files)
                        await self.importer_.summarize_all_cpp(self.project_id)
                    
                    async with self.store_semaphore:
                        await self.start_progress_store(self.total_files)
                        await self.importer_.store_all_cpp(self.project_id)

                    async with self.finalize_semaphore:
                        async with self.importer_.get_session() as session:
                            await self.importer_.finalize_relationships(self.project_id, session)

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
    
    async def cleanup_partial_ingestion(self, project_id: str):
        """
        Clean up partial ingestion by removing all nodes and relationships related to the project.

        Args:
            project_id (str): The project ID.
        """
        async with self.importer_.get_session() as session:
            try:
                await (await session.run(
                    "MATCH (n) WHERE n.project_id = $project_id "
                    "DETACH DELETE n",
                    project_id=self.project_id
                )).consume()
                logger.info(f"Cleaned up partial ingestion for project {self.project_id}")
            except Exception as e:
                logger.error(f"Error during cleanup for project {self.project_id}: {e}")

    # This is where the root input directory is passed. Everyhing scanned / ingested happens from here on in.
    # Recursively starting here.
    async def RecursiveParse(self, inputPath: str, topLevelInputPath: str, topLevelOutputPath: str, currentOutputPath: str) -> int:
        """
        Ingest files from the input path into the output path.

        Args:
            inputPath (str): The path to the input file or directory.
            topLevelInputPath (str): The top-level input path.
            topLevelOutputPath (str): The top-level output path.
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
        try:
            # Phase 1: Scanning / Parsing
            if not os.path.exists(inputPath):
                logger.error(f"Invalid path: {inputPath}")
                return -1

            inputType = 'd' if os.path.isdir(inputPath) else 'f'
            index_file_path = os.path.join(topLevelOutputPath, '.ngest_index')

            localPath = strip_top_level(inputPath, topLevelInputPath)

            with open(index_file_path, 'a') as index_file:
                index_file.write(f"{inputType},{localPath}\n")

            inputLocation, inputName = os.path.split(inputPath)
            tasks = []

            if inputType == 'd':
                tasks.append(self.ParseDirectory(inputPath=inputPath, inputLocation=inputLocation, inputName=inputName, topLevelInputPath=topLevelInputPath, topLevelOutputPath=topLevelOutputPath, currentOutputPath=currentOutputPath, project_id=self.project_id))
            else:
                if not self.should_ignore_file(inputPath):
                    tasks.append(self.ParseFile(inputPath=inputPath, inputLocation=inputLocation, inputName=inputName, topLevelInputPath=topLevelInputPath, topLevelOutputPath=topLevelOutputPath, currentOutputPath=currentOutputPath, project_id=self.project_id))

            # Wait for all parsing tasks to complete
            await asyncio.gather(*tasks)
            return 0
        except Exception as e:
            logger.error(f"Error during parsing in Ingest: {e}")
            await self.cleanup_partial_ingestion(self.project_id)
            return -1
    
    
    async def ParseDirectory(self, inputPath: str, inputLocation: str, inputName: str, topLevelInputPath: str, topLevelOutputPath: str, currentOutputPath: str, project_id: str) -> None:
        """
        Ingest a directory by recursively processing its contents.

        Args:
            inputPath (str): The path to the input directory.
            inputLocation (str): The location of the input directory.
            inputName (str): The name of the input directory.
            topLevelInputPath (str): The top-level input path.
            topLevelOutputPath (str): The top-level output path.
            currentOutputPath (str): The current output path.
            project_id (str): The project ID.
        """
        newOutputPath = os.path.join(currentOutputPath, inputName)
        os.makedirs(newOutputPath, exist_ok=True)

        tasks = []
        for item in os.listdir(inputPath):
            itemPath = os.path.join(inputPath, item)
            if not self.should_ignore_file(itemPath):
                tasks.append(self.RecursiveParse(itemPath, topLevelInputPath, topLevelOutputPath, newOutputPath))
        await asyncio.gather(*tasks)

    async def ParseFile(self, inputPath: str, inputLocation: str, inputName: str, topLevelInputPath: str, topLevelOutputPath: str, currentOutputPath: str, project_id: str) -> None:
        """
        Ingest a single file.

        Args:
            inputPath (str): The path to the input file.
            inputLocation (str): The location of the input file.
            inputName (str): The name of the input file.
            topLevelInputPath (str): The top-level input path.
            topLevelOutputPath (str): The top-level output path.
            currentOutputPath (str): The current output path.
            project_id (str): The project ID.
        """
        await self.importer_.ParseFile(inputPath, inputLocation, inputName, topLevelInputPath, topLevelOutputPath, currentOutputPath, project_id)

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

    async def update_project(self, project_id: str, inputPath: str) -> int:
        """
        Update a project with new files.

        Args:
            project_id (str): The project ID.
            inputPath (str): The path to the new files.

        Returns:
            int: The result of the update process.
        """
        # Implement project update logic here
        pass

    async def delete_project(self, project_id: str) -> int:
        """
        Delete a project by removing all related data and files.

        Args:
            project_id (str): The project ID.

        Returns:
            int: The result of the deletion process.
        """
        try:
            # Delete from database
            await self.cleanup_partial_ingestion(project_id)
            
            # Delete project directory
            project_path = os.path.expanduser(f"~/.ngest/projects/{project_id}")
            if os.path.exists(project_path):
                shutil.rmtree(project_path)
            
            logger.info(f"Project {project_id} deleted successfully")
            return 0
        except Exception as e:
            logger.error(f"Error deleting project {project_id}: {e}")
            return -1

    async def export_project(self, project_id: str, outputPath: str) -> int:
        """
        Export a project to a JSON file.

        Args:
            project_id (str): The project ID.
            outputPath (str): The path to the output file.

        Returns:
            int: The result of the export process.
        """
        try:
            async with self.importer_.get_session() as session:
                result = await (await session.run(
                    "MATCH (n) WHERE n.project_id = $project_id "
                    "RETURN n",
                    project_id=project_id
                )).consume()
                
                nodes = await result.data()
                
                with open(outputPath, 'w') as f:
                    json.dump(nodes, f)
                
            logger.info(f"Project {project_id} exported successfully to {outputPath}")
            return 0
        except Exception as e:
            logger.error(f"Error exporting project {project_id}: {e}")
            return -1
