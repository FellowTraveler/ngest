# retrieve.py
import faiss
import numpy as np
from neo4j import GraphDatabase
import ollama
from transformers import pipeline
import logging
import os
import re
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Neo4J driver
neo4j_url = "bolt://localhost:7687"
neo4j_driver = GraphDatabase.driver(neo4j_url)

class NBaseRetriever(ABC):
    def __init__(self, projectID=None):
        self.projectID = projectID

    @abstractmethod
    def retrieve(self, query):
        pass

class NFilesystemRetriever(NBaseRetriever):
    def __init__(self, projectID=None, base_dir="~/.ngest/projects/"):
        super().__init__(projectID)
        self.base_dir = os.path.expanduser(base_dir)
        self.project_dir = os.path.join(self.base_dir, self.projectID) if self.projectID else self.base_dir

    def retrieve(self, query):
        matches = []
        for root, _, files in os.walk(self.project_dir):
            for file in files:
                if file.endswith(".txt") or file.endswith(".cpp") or file.endswith(".py") or file.endswith(".rs"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if re.search(query, content, re.IGNORECASE):
                            matches.append((file_path, content))
        return matches

class NNeo4JRetriever(NBaseRetriever):
    def __init__(self, projectID=None, neo4j_driver=None, embeddings=None, entity_ids=None):
        super().__init__(projectID)
        self.neo4j_driver = neo4j_driver
        self.embeddings = embeddings
        self.entity_ids = entity_ids
        self.model = ollama.Model("nomic-embed-text")  # For embedding
        self.summarizer = pipeline("summarization")

    def vector_search(self, query_embedding, k=10):
        index = faiss.IndexFlatL2(self.embeddings.shape[1])
        index.add(self.embeddings)
        D, I = index.search(query_embedding, k)
        initial_candidates = [self.entity_ids[i] for i in I[0]]
        return initial_candidates

    def graph_traversal(self, initial_candidates):
        query = """
        MATCH (e:Entity)-[:RELATED_TO*1..2]-(related)
        WHERE e.id IN $initial_candidates
        RETURN DISTINCT related
        ORDER BY related.relevance DESC
        LIMIT 20
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, initial_candidates=initial_candidates)
            return [record["related"] for record in result]

    def re_rank_candidates(self, candidates, query):
        ranked_candidates = sorted(candidates, key=lambda x: x['relevance'], reverse=True)
        return ranked_candidates

    def generate_hypothetical_document(self, ranked_candidates):
        relevant_snippets = [self.get_text(candidate['id']) for candidate in ranked_candidates]
        combined_text = " ".join(relevant_snippets)
        hypothetical_doc = self.summarizer(combined_text, max_length=500, min_length=100, do_sample=False)[0]['summary_text']
        return hypothetical_doc

    def get_text(self, entity_id):
        query = "MATCH (e:Entity {id: $entity_id}) RETURN e.content AS content"
        with self.neo4j_driver.session() as session:
            result = session.run(query, entity_id=entity_id)
            return result.single()["content"]

    def generate_response(self, query, ranked_candidates):
        response = f"Here are the details about '{query}':\n\n"
        for candidate in ranked_candidates:
            response += f"- {candidate['summary']}\n\n"
        return response

    def retrieve(self, query):
        query_embedding = self.model.embed_text(query)
        initial_candidates = self.vector_search(query_embedding)
        contextual_entities = self.graph_traversal(initial_candidates)
        ranked_candidates = self.re_rank_candidates(contextual_entities, query)

        if not ranked_candidates:
            response = self.generate_hypothetical_document(ranked_candidates)
        else:
            response = self.generate_response(query, ranked_candidates)
        return response

# Example Usage
embeddings = np.load("embeddings.npy")
entity_ids = np.load("entity_ids.npy")

retriever_instance = NNeo4JRetriever(projectID="example_project", neo4j_driver=neo4j_driver, embeddings=embeddings, entity_ids=entity_ids)

query = "How to use the public API in the opentxs project?"
response = retriever_instance.retrieve(query)
print(response)
