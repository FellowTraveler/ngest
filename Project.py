from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from datetime import datetime

class Project:
    def __init__(self, project_id, folder_name, description=None, status="Created", created_date=None, modified_date=None):
        self.project_id = project_id
        self.folder_name = folder_name
        self.created_date = created_date or datetime.now().isoformat()
        self.modified_date = modified_date or datetime.now().isoformat()
        self.description = description
        self.status = status

    @staticmethod
    async def create_in_database(session, project_id, folder_name, description=None):
        created_date = datetime.now().isoformat()
        query = """
        CREATE (p:Project {
            project_id: $project_id,
            folder_name: $folder_name,
            description: $description,
            status: 'Created',
            created_date: $created_date,
            modified_date: $created_date
        })
        RETURN p
        """
        try:
            result = await session.run(query,
                                       project_id=project_id,
                                       folder_name=folder_name,
                                       description=description,
                                       created_date=created_date)
            record = await result.single()
            if record:
                node = record["p"]
                return Project(
                    node["project_id"], node["folder_name"], node["description"],
                    node["status"], node["created_date"], node["modified_date"]
                )
            return None
        except Neo4jError as e:
            print(f"Error creating project in database: {e}")
            return None

    # ... rest of the class remains the same ...

    @staticmethod
    async def update_in_database(session, project_id):
        modified_date = datetime.now().isoformat()
        query = """
        MATCH (p:Project {project_id: $project_id})
        SET p.modified_date = $modified_date
        RETURN p
        """
        try:
            result = await session.run(query, project_id=project_id, modified_date=modified_date)
            record = await result.single()
            if record:
                node = record["p"]
                return Project(
                    node["project_id"], node["folder_name"], node["description"],
                    node["status"], node["created_date"], node["modified_date"]
                )
            return None
        except Neo4jError as e:
            print(f"Error updating project in database: {e}")
            return None

    @staticmethod
    async def retrieve_from_database(session, project_id):
        query = """
        MATCH (p:Project {project_id: $project_id})
        RETURN p
        """
        try:
            result = await session.run(query, project_id=project_id)
            record = await result.single()
            if record:
                node = record["p"]
                return Project(
                    node["project_id"], node["folder_name"], node["description"],
                    node["status"], node["created_date"], node["modified_date"]
                )
            return None
        except Neo4jError as e:
            print(f"Error retrieving project from database: {e}")
            return None
