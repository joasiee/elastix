from thesispy.elastix_wrapper.parameters import Parameters
from thesispy.elastix_wrapper.runner import run
from thesispy.definitions import ROOT_DIR

json_file = ROOT_DIR / "json_queue" / "params.json"

with open(json_file) as f:
    params = Parameters.from_json(f.read())

run_dir = ROOT_DIR / "output" / str(params)
run([params], run_dir, validate=False)
