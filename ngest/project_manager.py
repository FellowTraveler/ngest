# Copyright 2024 Chris Odom
# MIT License

import asyncio
import logging
import uuid
from typing import List, Dict
from ngest.ningest import NIngest

logger = logging.getLogger(__name__)

class ProjectManager:
    def __init__(self):
        self.ingest_semaphore = asyncio.Semaphore(1)

    async def list_projects(self) -> List[Dict[str, str]]:
        """
        List all projects in the database.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing project information.
        """
        try:
            async with self.ingest_semaphore:
                ingest_instance = NIngest(project_id=None)  # We don't need a specific project ID for listing
                async with ingest_instance.importer_.get_session() as session:
                    result = await session.run(
                        "MATCH (p:Project) RETURN p.project_id AS project_id, p.folder_name AS folder_name, p.created_date AS created_date"
                    )
                    projects = await result.data()
                return projects
        except Exception as e:
            logger.error(f"Error listing projects: {e}")
            raise
            
    async def create_project(self, input_path: str) -> str:
        async with self.ingest_semaphore:
            project_id = str(uuid.uuid4())
            ingest_instance = NIngest(project_id=project_id)
            
            try:
                result = await ingest_instance.start_ingestion(input_path)
                if result == 0:
                    logger.info(f"Project created and ingested. Project ID: {project_id}")
                    return project_id
                else:
                    raise Exception("Failed to start project ingestion")
            except Exception as e:
                logger.error(f"Error creating project or starting ingestion: {e}")
                # Clean up any partial ingestion
                await ingest_instance.cleanup_partial_ingestion(project_id)
                raise

    async def update_project(self, project_id: str, input_path: str) -> int:
        """
        Update an existing project with new/deleted/updated files.

        Args:
            project_id (str): The project ID.
            input_path (str): The path to the new files.

        Returns:
            int: The result of the update process.
        """
        async with self.ingest_semaphore:
            ingest_instance = NIngest(project_id=project_id)
            return await ingest_instance.update_project(project_id, input_path)

    async def delete_project(self, project_id: str) -> int:
        """
        Delete a project.

        Args:
            project_id (str): The project ID.

        Returns:
            int: The result of the deletion process.
        """
        async with self.ingest_semaphore:
            ingest_instance = NIngest(project_id=project_id)
            return await ingest_instance.delete_project(project_id)

    async def export_project(self, project_id: str, export_path: str) -> int:
        """
        Export a project to a JSON file.

        Args:
            project_id (str): The project ID.
            export_path (str): The path to the output file.

        Returns:
            int: The result of the export process.
        """
        async with self.ingest_semaphore:
            ingest_instance = NIngest(project_id=project_id)
            return await ingest_instance.export_project(project_id, export_path)
