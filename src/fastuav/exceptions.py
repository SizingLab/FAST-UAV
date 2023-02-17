"""
Exception for cmd package
"""

from fastoad.exceptions import FastError


class FastLcaProjectDoesNotExist(FastError):
    """Raised when the brightway project has not been created yet."""

    def __init__(self, project_name=None):
        msg = f'LCA project "{project_name}" does not exist. Create it using `create_lca_project` cmd.'
        self.project_name = project_name
        super().__init__(msg)


class FastLcaDatabaseIsNotImported(FastError):
    """Raised when the background LCA database (e.g., EcoInvent) has not been imported yet."""

    def __init__(self, project_name=None, db_name=None):
        msg = f'LCA database "{db_name}" has not been imported in project "{project_name}".'
        self.project_name = project_name
        self.db_name = db_name
        super().__init__(msg)


class FastLcaMethodDoesNotExist(FastError):
    """Raised when the lca method does not exist in brightway."""

    def __init__(self, method_name=None):
        msg = f'LCA method "{method_name}" does not exist in brightway2.'
        self.method_name = method_name
        super().__init__(msg)


class FastLcaParameterNotDeclared(FastError):
    """Raised when the lca parameter has not been declared."""

    def __init__(self, param_name=None):
        msg = f'LCA parameter "{param_name}" has not been declared in lca_algebraic.'
        self.param_name = param_name
        super().__init__(msg)