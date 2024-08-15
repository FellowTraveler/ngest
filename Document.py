from File import File
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

class Document(File):
    uri = "bolt://localhost:7689"
    user = "neo4j"
    password = "mynewpassword"

    def __init__(self, filename, full_path, size_in_bytes, project_id, created_date=None, modified_date=None, extension=None, content_type=None):
        super().__init__(filename, full_path, size_in_bytes, project_id, created_date, modified_date, extension)
        self.content_type = content_type

    async def generate_cypher_query(self, session):
        element_id = await File.get_element_id_by_project_and_path(session, self.project_id, self.full_path)
        properties = {
            "filename": self.filename,
            "full_path": self.full_path,
            "size_in_bytes": self.size_in_bytes,
            "project_id": self.project_id,
            "created_date": self.created_date,
            "modified_date": self.modified_date,
            "extension": self.extension,
            "content_type": self.content_type
        }
        if element_id:
            query = f"""
            MATCH (d:File) WHERE elementId(d) = $element_id
            SET d:Document
            SET d += $properties
            RETURN d
            """
        else:
            query = f"""
            CREATE (d:File:Document)
            SET d = $properties
            RETURN d
            """
        return query, element_id


    @staticmethod
    async def retrieve_from_database(session, element_id):
        query = """
        MATCH (d:Document) WHERE elementId(d) = $element_id
        RETURN d
        """
        result = await session.run(query, element_id=element_id)
        record = await result.single()
        if record:
            node = record["d"]
            return Document(
                node["filename"], node["full_path"], node["size_in_bytes"],
                node["project_id"], node["created_date"], node["modified_date"], node["extension"],
                node["content_type"]
            )
        return None


