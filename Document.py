from File import File
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

class Document(File):

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
            MATCH (n:File) WHERE elementId(n) = '{element_id}'
            SET n:Document
            SET n.filename = '{properties["filename"]}',
                n.full_path = '{properties["full_path"]}',
                n.size_in_bytes = {properties["size_in_bytes"]},
                n.project_id = '{properties["project_id"]}',
                n.created_date = '{properties["created_date"]}',
                n.modified_date = '{properties["modified_date"]}',
                n.extension = '{properties["extension"]}',
                n.content_type = '{properties["content_type"]}'
            RETURN elementId(n)
            """
        else:
            query = f"""
            CREATE (n:File:Document {{
                filename: '{properties["filename"]}',
                full_path: '{properties["full_path"]}',
                size_in_bytes: {properties["size_in_bytes"]},
                project_id: '{properties["project_id"]}',
                created_date: '{properties["created_date"]}',
                modified_date: '{properties["modified_date"]}',
                extension: '{properties["extension"]}',
                content_type: '{properties["content_type"]}'
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
        MATCH (n:Document) WHERE elementId(n) = $element_id
        RETURN n
        """
        result = await session.run(query, element_id=element_id)
        record = await result.single()
        if record:
            node = record["n"]
            return Document(
                node["filename"], node["full_path"], node["size_in_bytes"],
                node["project_id"], node["created_date"], node["modified_date"], node["extension"],
                node["content_type"]
            )
        return None


