import logging
from pathlib import Path
import time
from typing import Any, Dict, List
from thesispy.definitions import Collection

from thesispy.elastix_wrapper.parameters import Parameters
from thesispy.elastix_wrapper.watchdog import SaveStrategy, Watchdog
from thesispy.elastix_wrapper.wrapper import execute_elastix, execute_visualize, get_run_result
from thesispy.experiments.validation import calc_validation

logger = logging.getLogger("Runner")


def run(
    params_list: List[Parameters],
    run_dir: Path,
    save_strategy: SaveStrategy = None,
    suppress_stdout: bool = True,
    visualize: bool = False,
    validate: bool = True,
) -> Dict[str, Any]:
    """Run elastix with the given list of parameters.
    
    Args:
        params_list: List of parameters to run elastix in consecutive fashion with.
        run_dir: Directory to run elastix in.
        save_strategy: Strategy for saving the results of the run.
        suppress_stdout: Whether to suppress the output of elastix.
        visualize: Whether to visualize the results of the run after it is done using any of the supported visualizers.
        validate: Whether to calculate the validation metrics and plot visualizations.
    """
    time_start = time.perf_counter()

    if type(params_list) is not list:
        params_list = [params_list]

    run_dir.mkdir(parents=True)
    param_files = []
    main_params = params_list[-1]
    for i, params in enumerate(params_list):
        param_files.append(params.write(run_dir, i + 1))

    out_dir = run_dir.joinpath(Path("out"))
    if save_strategy:
        wd = Watchdog(out_dir, main_params["NumberOfResolutions"])
        wd.set_strategy(save_strategy)
        wd.start()

    finished = False
    run_result = None
    logger.info(f"Running elastix in: {str(run_dir)}")
    try:
        execute_elastix(param_files, out_dir, main_params, suppress_stdout)
        finished = True
    except KeyboardInterrupt:
        logger.warning(f"Run ended prematurely by user.")
    except Exception as e:
        logger.error(f"Run ended with exception: {e}")
    finally:
        if finished:
            out_dir = run_dir.joinpath(Path("out"))
            transform_params = out_dir / "TransformParameters.0.txt"
            run_result = get_run_result(
                Collection(main_params["Collection"]), int(main_params["Instance"]), transform_params
            )

            if validate:
                val_metrics = calc_validation(run_result)

        if save_strategy:
            if finished and validate:
                for metric in val_metrics:
                    wd.sv_strategy.save_custom(metric)
            wd.stop()
            wd.join()
            wd.sv_strategy.close(finished)

        time_end = time.perf_counter()
        logger.info(f"Run ended. It took {time_end - time_start:0.4f} seconds")

        if finished and visualize:
            execute_visualize(out_dir)

    return run_result


if __name__ == "__main__":
    params_main = (
        Parameters.from_base(mesh_size=4, seed=1)
        .mogomea(mixing_components=5, pop_size=100, elitist_archive_size_target=200)
        .stopping_criteria(max_time_s=600)
        .instance(Collection.SYNTHETIC, 1)
    )
    run(
        [params_main],
        Path("output/" + str(params_main)),
        suppress_stdout=False,
        visualize=False,
        validate=False,
    )
