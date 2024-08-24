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

    def generate_cypher_query(self, element_id=None):
        if element_id:
            query = f"""
            MATCH (p:Project) WHERE elementId(p) = $element_id
            SET p.folder_name = $folder_name, p.description = $description, p.status = $status, p.created_date = $created_date, p.modified_date = $modified_date
            RETURN p
            """
        else:
            query = """
            CREATE (p:Project {project_id: $project_id, folder_name: $folder_name, description: $description, status: $status, created_date: $created_date, modified_date: $modified_date})
            RETURN p
            """
        return query


    @staticmethod
    async def retrieve_from_database(session, element_id):
    
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
        uri = "bolt://localhost:7689"
        user = "neo4j"
        password = "mynewpassword"
        
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                query = """
                MATCH (p:Project) WHERE elementId(p) = $element_id
                RETURN p.project_id AS project_id, p.folder_name AS folder_name, p.description AS description, p.status AS status, p.created_date AS created_date, p.modified_date AS modified_date
                """
                result = session.run(query, element_id=element_id)
                record = result.single()
                if record:
                    return Project(record["project_id"], record["folder_name"], record["description"], record["status"], record["created_date"], record["modified_date"])
                return None
