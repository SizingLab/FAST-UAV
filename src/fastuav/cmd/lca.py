"""
API
"""
import brightway2 as bw
import lca_algebraic as lcalg
from fastuav.constants import LCA_USER_DB, LCA_MODEL_KEY


def create_lca_project(project_name, db_path, db_name):
    """
    Set up the project if not already done, and import eco invent database.
    """
    # switch to the project passed as argument, and create it first if it doesn't exist
    bw.projects.set_current(project_name)

    # Set up project
    bw.bw2setup()

    if db_name in bw.databases:
        print(f'"{db_name}" database already present!!! No setup is needed')
    else:
        ei = bw.SingleOutputEcospold2Importer(db_path, db_name)
        ei.apply_strategies()
        ei.statistics()
        ei.drop_unlinked(i_am_reckless=True)
        ei.write_database()
        print(f'Created project: "{project_name}" in directory: "{bw.projects.dir}"')

    print(f'Available projects on your computer: \n {list(bw.projects)}')
    print(f'Available databases in project "{project_name}": \n {list(bw.databases)}')


def get_lca_activities(db_name: str = LCA_USER_DB):
    """Get all activities declared in a database."""
    return [act for act in bw.Database(db_name)]


def get_lca_main_activity(db_name: str = LCA_USER_DB):
    """Get the main activity (i.e., the top-level activity, or the model) declare in LCA module of FAST."""
    return lcalg.getActByCode(db_name, LCA_MODEL_KEY)


def list_lca_parameters():
    """List all parameters declared in the LCA module of FAST."""
    return lcalg.list_parameters()

