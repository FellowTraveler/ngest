# Copyright 2024 Chris Odom
# MIT License

from neo4j import GraphDatabase
from datetime import datetime
from neo4j.exceptions import Neo4jError

class File:
    def __init__(self, filename, full_path, size_in_bytes, project_id, created_date=None, modified_date=None, extension=None):
        self.filename = filename
        self.full_path = full_path
        self.size_in_bytes = size_in_bytes
        self.project_id = project_id
        self.created_date = created_date or datetime.now().isoformat()
        self.modified_date = modified_date or datetime.now().isoformat()
        self.extension = extension or self.get_extension(filename)
        self.id = f"{project_id}/{full_path}"

    @staticmethod
    def get_extension(filename):
        return filename.split('.')[-1] if '.' in filename else ''

    @staticmethod
    async def create_in_database(session, filename, full_path, size_in_bytes, project_id, created_date=None, modified_date=None, extension=None):
        file = File(filename, full_path, size_in_bytes, project_id, created_date, modified_date, extension)
        query = query = """
            MERGE (f:File {id: $id})
            SET f.filename = $filename,
                f.full_path = $full_path,
                f.size_in_bytes = $size_in_bytes,
                f.project_id = $project_id,
                f.created_date = $created_date,
                f.modified_date = $modified_date,
                f.extension = $extension
            WITH f
            MATCH (p:Project {project_id: $project_id})
            MERGE (p)-[:HAS_FILE]->(f)
            MERGE (f)-[:BELONGS_TO_PROJECT]->(p)
            RETURN f
            """
        try:
            result = await session.run(query,
                id=file.id,
                filename=file.filename,
                full_path=file.full_path,
                size_in_bytes=file.size_in_bytes,
                project_id=file.project_id,
                created_date=file.created_date,
                modified_date=file.modified_date,
                extension=file.extension
            )
            record = await result.single()
            if record:
                return file
            return None
        except Neo4jError as e:
            print(f"Error creating file in database: {e}")
            return None
            
    @staticmethod
    async def retrieve_from_database(session, project_id, full_path):
        file_id = f"{project_id}/{full_path}"
        query = """
        MATCH (f:File {id: $id})
        RETURN f
        """
        result = await session.run(query, id=file_id)
        record = await result.single()
        if record:
            node = record["f"]
            return File(
                node["filename"], node["full_path"], node["size_in_bytes"],
                node["project_id"], node["created_date"], node["modified_date"], node["extension"]
            )
        return None
