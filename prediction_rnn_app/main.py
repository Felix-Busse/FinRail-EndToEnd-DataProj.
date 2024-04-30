# Add finrail_db module to access data base
modules_path = os.path.join(os.getcwd(), 'data_collect_app')
if modules_path not in sys.path:
    sys.path.append(modules_path)

import finrail_db