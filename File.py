# Copyright 2024 Chris Odom
# MIT License

from neo4j import GraphDatabase
from datetime import datetime

class File:
    def __init__(self, filename, full_path, size_in_bytes, project_id, created_date=None, modified_date=None, extension=None):
        self.filename = filename
        self.full_path = full_path
        self.size_in_bytes = size_in_bytes
        self.project_id = project_id
        self.created_date = created_date or datetime.now().isoformat()
        self.modified_date = modified_date or datetime.now().isoformat()
        self.extension = extension or self.get_extension(filename)

    @staticmethod
    def get_extension(filename):
        return filename.split('.')[-1] if '.' in filename else ''

    async def generate_cypher_query(self, session):
        element_id = await self.get_element_id_by_project_and_path(session, self.project_id, self.full_path)
        if element_id:
            query = f"""
            MATCH (n:File) WHERE elementId(n) = '{element_id}'
            SET n.filename = '{self.filename}',
                n.full_path = '{self.full_path}',
                n.size_in_bytes = {self.size_in_bytes},
                n.project_id = '{self.project_id}',
                n.created_date = '{self.created_date}',
                n.modified_date = '{self.modified_date}',
                n.extension = '{self.extension}'
            RETURN elementId(n)
            """
        else:
            query = f"""
            CREATE (n:File {{
                filename: '{self.filename}',
                full_path: '{self.full_path}',
                size_in_bytes: {self.size_in_bytes},
                project_id: '{self.project_id}',
                created_date: '{self.created_date}',
                modified_date: '{self.modified_date}',
                extension: '{self.extension}'
            }})
            RETURN elementId(n)
            """
        return query, element_id

    @staticmethod
    async def retrieve_from_database(session, project_id, full_path):
        element_id = await File.get_element_id_by_project_and_path(session, project_id, full_path)
        if not element_id:
            return None
        query = """
        MATCH (n:File) WHERE elementId(n) = $element_id
        RETURN n.filename AS filename, n.full_path AS full_path, n.size_in_bytes AS size_in_bytes,
               n.project_id AS project_id, n.created_date AS created_date, n.modified_date AS modified_date, n.extension AS extension
        """
        result = await session.run(query, element_id=element_id)
        record = await result.single()
        if record:
            return File(record["filename"], record["full_path"], record["size_in_bytes"],
                        record["project_id"], record["created_date"], record["modified_date"], record["extension"])
        return None
    @staticmethod
    async def get_element_id_by_project_and_path(session, project_id, full_path):
        query = """
        MATCH (n:File) WHERE n.project_id = $project_id AND n.full_path = $full_path
        RETURN elementId(n) AS element_id
        """
        result = await session.run(query, project_id=project_id, full_path=full_path)
        record = await result.single()
        if record:
            return record["element_id"]
        return None
