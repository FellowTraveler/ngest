# Copyright 2024 Chris Odom
# MIT License

from file import File
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

class Document(File):
    def __init__(self, filename, full_path, size_in_bytes, project_id, created_date=None, modified_date=None, extension=None, content_type=None, id=None):
        super().__init__(filename, full_path, size_in_bytes, project_id, created_date, modified_date, extension)
        self.content_type = content_type
        self.id = id or f"{project_id}/{full_path}"  # Use provided id or construct it

    @staticmethod
    async def create_in_database(session, project_id, full_path, content_type):
        file_id = f"{project_id}/{full_path}"
        query = """
        MATCH (f:File {id: $file_id})
        SET f:Document
        SET f.content_type = $content_type
        RETURN f
        """
        try:
            result = await session.run(query,
                file_id=file_id,
                content_type=content_type
            )
            record = await result.single()
            if record:
                node = record["f"]
                return Document(
                    filename=node["filename"],
                    full_path=node["full_path"],
                    size_in_bytes=node["size_in_bytes"],
                    project_id=node["project_id"],
                    created_date=node["created_date"],
                    modified_date=node["modified_date"],
                    extension=node["extension"],
                    content_type=content_type,
                    id=file_id  # Pass the id to the constructor
                )
            return None
        except Neo4jError as e:
            print(f"Error creating document in database: {e}")
            return None

    @staticmethod
    async def retrieve_from_database(session, project_id, full_path):
        file = await File.retrieve_from_database(session, project_id, full_path)
        if not file:
            return None

        query = """
        MATCH (d:Document {id: $id})
        RETURN d.content_type AS content_type
        """
        result = await session.run(query, id=file.id)
        record = await result.single()
        if record:
            return Document(
                file.filename, file.full_path, file.size_in_bytes,
                file.project_id, file.created_date, file.modified_date, file.extension,
                record["content_type"]
            )
        return None
