from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from datetime import datetime

class Project:
    def __init__(self, project_id, folder_name, description=None, status=None, created_date=None, modified_date=None):
        self.project_id = project_id
        self.folder_name = folder_name
        self.created_date = created_date or datetime.now().isoformat()
        self.modified_date = modified_date or datetime.now().isoformat()
        self.description = description
        self.status = status

    async def generate_cypher_query(self, session, element_id=None):
        if element_id:
            query = f"""
            MATCH (p:Project) WHERE elementId(p) = '{element_id}'
            SET p.folder_name = '{self.folder_name}', p.description = '{self.description}', p.status = '{self.status}', p.created_date = '{self.created_date}', p.modified_date = '{self.modified_date}'
            RETURN elementId(p)
            """
        else:
            query = f"""
            CREATE (p:Project {{
                project_id: '{self.project_id}',
                folder_name: '{self.folder_name}',
                description: '{self.description}',
                status: '{self.status}',
                created_date: '{self.created_date}',
                modified_date: '{self.modified_date}'
            }})
            RETURN elementId(p)
            """
        return query, element_id

    @staticmethod
    async def retrieve_from_database(session, element_id):
        query = """
        MATCH (p:Project) WHERE elementId(p) = $element_id
        RETURN p
        """
        result = await session.run(query, element_id=element_id)
        record = await result.single()
        if record:
            node = record["p"]
            return Project(
                node["project_id"], node["folder_name"], node["description"],
                node["status"], node["created_date"], node["modified_date"]
            )
        return None

    @staticmethod
    async def get_project_id_by_folder_name(session, folder_name):
        query = """
        MATCH (p:Project) WHERE p.folder_name = $folder_name
        RETURN p.project_id AS project_id
        """
        result = await session.run(query, folder_name=folder_name)
        record = await result.single()
        if record:
            return record["project_id"]
        return None
