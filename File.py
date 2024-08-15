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
        self.uri = "bolt://localhost:7689"
        self.user = "neo4j"
        self.password = "mynewpassword"

    @staticmethod
    def get_extension(filename):
        return filename.split('.')[-1] if '.' in filename else ''

    async def generate_cypher_query(self, session):
        element_id = await self.get_element_id_by_project_and_path(session, self.project_id, self.full_path)
        if element_id:
            query = f"""
            MATCH (f:File) WHERE elementId(f) = $element_id
            SET f.filename = $filename, f.full_path = $full_path, f.size_in_bytes = $size_in_bytes,
                f.project_id = $project_id, f.created_date = $created_date, f.modified_date = $modified_date, f.extension = $extension
            RETURN f
            """
        else:
            query = """
            CREATE (f:File {filename: $filename, full_path: $full_path, size_in_bytes: $size_in_bytes,
                            project_id: $project_id, created_date: $created_date, modified_date: $modified_date, extension: $extension})
            RETURN f
            """
        return query, element_id


    @staticmethod
    async def retrieve_from_database(session, element_id):
        query = """
        MATCH (f:File) WHERE elementId(f) = $element_id
        RETURN f.filename AS filename, f.full_path AS full_path, f.size_in_bytes AS size_in_bytes,
               f.project_id AS project_id, f.created_date AS created_date, f.modified_date AS modified_date, f.extension AS extension
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
        MATCH (f:File) WHERE f.project_id = $project_id AND f.full_path = $full_path
        RETURN elementId(f) AS element_id
        """
        result = await session.run(query, project_id=project_id, full_path=full_path)
        record = await result.single()
        if record:
            return record["element_id"]
        return None
