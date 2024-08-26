# Copyright 2024 Chris Odom
# MIT License

# ngest/ngest/__init__.py

from .utils.clang_setup import setup_clang

# Automatically set up Clang when the package is imported
setup_clang()

from .ningest import NIngest
from .project_manager import ProjectManager
from .file import File
from .document import Document
from .project import Project
from .pdf import PDF

__all__ = ['NIngest', 'ProjectManager', 'File', 'Document', 'Project', 'PDF']

