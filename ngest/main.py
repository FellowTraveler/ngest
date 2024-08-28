# Copyright 2024 Chris Odom
# MIT License

import os
import datetime
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import logging
import asyncio
from neo4j import AsyncGraphDatabase
import argparse
import configparser
import dotenv
from typing import List, Dict

import ngest  # This import will trigger the Clang setup in __init__.py
from ngest.ningest import NIngest
from ngest.project_manager import ProjectManager

# Load environment variables
dotenv.load_dotenv()

# Constants (now loaded from environment variables with fallbacks to config file)
config = configparser.ConfigParser()
config_dir = os.path.expanduser("~/.ngest")
config_file = os.path.join(config_dir, 'config.ini')

# Configure logging with different levels
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the logging level based on a condition (e.g., a command-line argument or an environment variable)
hide_info_logs = False  # Set this to True to hide info logs

if hide_info_logs:
    logging.getLogger().setLevel(logging.WARNING)
else:
    logging.getLogger().setLevel(logging.INFO)

async def main():
    parser = argparse.ArgumentParser(description='Manage projects in Neo4j database')
    parser.add_argument('action', choices=['create', 'update', 'delete', 'export', 'list'], help='Action to perform')
    parser.add_argument('--input_path', type=str, help='Path to file or directory to ingest')
    parser.add_argument('--project_id', type=str, help='Project ID for update, delete, or export actions')
    parser.add_argument('--export_path', type=str, help='Output path for export action')
    args = parser.parse_args()

    project_manager = ProjectManager()

    try:
        if args.action == 'create':
            if not args.input_path:
                raise ValueError("input_path is required for create action")
            project_id = await project_manager.create_project(args.input_path)
            print(f"Project created and ingestion started successfully. Project ID: {project_id}")
        elif args.action == 'update':
            if not args.project_id or not args.input_path:
                raise ValueError("Both project_id and input_path are required for update action")
            result = await project_manager.update_project(args.project_id, args.input_path)
            print(f"Project update {'succeeded' if result == 0 else 'failed'}")
        elif args.action == 'delete':
            if not args.project_id:
                raise ValueError("project_id is required for delete action")
            result = await project_manager.delete_project(args.project_id)
            print(f"Project deletion {'succeeded' if result == 0 else 'failed'}")
        elif args.action == 'export':
            if not args.project_id or not args.export_path:
                raise ValueError("Both project_id and export_path are required for export action")
            result = await project_manager.export_project(args.project_id, args.export_path)
            print(f"Project export {'succeeded' if result == 0 else 'failed'}")
        elif args.action == 'list':
            projects = await project_manager.list_projects()
            if projects:
                print("Projects:")
                for project in projects:
                    print(f"ID: {project['project_id']}, Name: {project['folder_name']}, Created: {project['created_date']}")
            else:
                print("No projects found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def run_main():
    asyncio.run(main())
    
if __name__ == "__main__":
    run_main()
