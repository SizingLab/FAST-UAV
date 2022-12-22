# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['fastuav',
 'fastuav.configurations',
 'fastuav.data',
 'fastuav.models',
 'fastuav.models.add_ons',
 'fastuav.models.aerodynamics',
 'fastuav.models.geometry',
 'fastuav.models.mtow',
 'fastuav.models.performance',
 'fastuav.models.performance.mission',
 'fastuav.models.performance.mission.mission_definition',
 'fastuav.models.performance.mission.mission_definition.resources',
 'fastuav.models.performance.mission.mission_definition.tests',
 'fastuav.models.propulsion',
 'fastuav.models.propulsion.energy',
 'fastuav.models.propulsion.energy.battery',
 'fastuav.models.propulsion.esc',
 'fastuav.models.propulsion.gearbox',
 'fastuav.models.propulsion.motor',
 'fastuav.models.propulsion.propeller',
 'fastuav.models.propulsion.propeller.aerodynamics',
 'fastuav.models.scenarios',
 'fastuav.models.scenarios.thrust',
 'fastuav.models.scenarios.wing_loading',
 'fastuav.models.stability',
 'fastuav.models.stability.static_longitudinal',
 'fastuav.models.stability.static_longitudinal.center_of_gravity',
 'fastuav.models.stability.static_longitudinal.center_of_gravity.components',
 'fastuav.models.structures',
 'fastuav.models.structures.wing',
 'fastuav.models.wires',
 'fastuav.notebooks',
 'fastuav.utils',
 'fastuav.utils.catalogues',
 'fastuav.utils.drivers',
 'fastuav.utils.postprocessing',
 'fastuav.utils.postprocessing.old',
 'fastuav.utils.postprocessing.sensitivity_analysis']

package_data = \
{'': ['*'],
 'fastuav': ['missions/*'],
 'fastuav.data': ['catalogues/Batteries/*',
                  'catalogues/ESC/*',
                  'catalogues/Motors/*',
                  'catalogues/Propeller/*',
                  'catalogues/Propeller/performances/*'],
 'fastuav.models': ['performance/mission/mission_definition/tests/data/*']}

install_requires = \
['SALib==1.4.5',
 'cma>=3.1.0,<4.0.0',
 'fast-oad-core>=1.4.1,<2.0.0',
 'kaleido==0.2.1',
 'matplotlib>=3.6.2,<4.0.0',
 'numpy==1.23.5',
 'openmdao<3.18.0',
 'psutil',
 'scikit-learn>=1.0.2,<2.0.0',
 'stdatm==0.2.0']

entry_points = \
{'fastoad.plugins': ['uav = fastuav']}

setup_kwargs = {
    'name': 'fastuav',
    'version': '0.0.0',
    'description': 'FAST-UAV is a framework for performing rapid Overall Aircraft Design for Unmanned Aerial Vehicles',
    'long_description': 'FAST-UAV\n========\n\nFAST-UAV is a solution to perform optimal drone design with a multi-disciplinary approach.\n\nBased on the [FAST-OAD](https://github.com/fast-aircraft-design/FAST-OAD) framework, it allows to easily switch between models to address different types of configurations. \n\nCurrently, FAST-UAV is bundled with analytical models for multirotor and fixed wing drones.\n\nInstall\n-------\nIt is recommended to install FAST-UAV in a virtual environment, using poetry.\n\n> ``` {.bash}\n> $ conda create -n fastuav python=3.8\n> $ conda activate fastuav\n> $ conda install poetry\n> $ poetry install\n> ```\n\nRun\n-------\n> ``` {.bash}\n> $ jupyter lab\n> ```\n\nPublications\n------------\n> M. Budinger, A. Reysset, A. Ochotorena, and S. Delbecq, ‘Scaling laws and similarity models for the preliminary design of multirotor drones’, Aerospace Science and Technology, vol. 98, p. 105658, Mar. 2020, doi: 10.1016/j.ast.2019.105658.\n\n> S. Delbecq, M. Budinger, A. Ochotorena, A. Reysset, and F. Defay, ‘Efficient sizing and optimization of multirotor drones based on scaling laws and similarity models’, Aerospace Science and Technology, vol. 102, p. 105873, Jul. 2020, doi: 10.1016/j.ast.2020.105873.\n\n> F. Pollet, S. Delbecq, M. Budinger, and J.-M. Moschetta, ‘Design optimization of multirotor drones in cruise’, Sep. 2021.\n\n> S. Delbecq, M. Budinger, C. Coic, and N. Bartoli, ‘Trajectory and design optimization of multirotor drones with system simulation’, VIRTUAL EVENT, United States, Jan. 2021. doi: 10.2514/6.2021-0211.\n\n> [DroneApp](https://github.com/SizingLab/droneapp-legacy) sizing tool\n\n',
    'author': 'Félix POLLET',
    'author_email': 'felix.pollet@isae-supaero.fr',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://github.com/SizingLab/FAST-UAV',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.8,<3.10',
}


setup(**setup_kwargs)
