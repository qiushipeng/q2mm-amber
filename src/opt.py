import logging
import logging.config
import textwrap
import copy
import subprocess as sp

import constants as co
import data_structs
from data_structs import FF
from calculators import *

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)


class OptError(Exception):
    """
    Raised when an optimizer does something bad.
    """
    pass

def catch_run_errors(func):
    def wrapper(*args, **kwargs):
        papa_bear = args[0]
        try:
            return func(*args, **kwargs)
        except (ZeroDivisionError, OptError, data_structs.ParamError, sp.CalledProcessError) as e:
            logger.warning('opt.catch_run_errors caught an error!')
            logger.warning(e)
            if papa_bear.best_ff is None:
                logger.warning('Exiting {} and returning initial FF.'.format(
                        papa_bear.__class__.__name__.lower()))
                papa_bear.ff.export_ff(papa_bear.ff.path)
                return papa_bear.ff
            else:
                logger.warning('Exiting {} and returning best FF.'.format(
                        papa_bear.__class__.__name__.lower()))
                papa_bear.best_ff.export_ff(papa_bear.best_ff.path)
                return papa_bear.best_ff
    return wrapper

class Optimizer(object):

    def __init__(self,
                 direc,
                 ff_to_opt,
                 args_ff,
                 ref_data):
        logger.log(logging.INFO, '~~ {} SETUP ~~'.format(
                self.__class__.__name__.upper()).rjust(79, '~'))
        self.direc = direc
        self.ff_to_opt:FF = ff_to_opt
        self.args_ff = args_ff # this will need to persist for calculate calls to produce the correct scripts and calcs
        self.args_ref = ref_data # this will be unnecessary and should just be a passed in ReferenceData object or something
        self.commands = parse_commands(args_ff)
        
        self.new_ffs = []

        self.param_coord = self.ff_to_opt.to_coords()
        self.param_bounds = self.ff_to_opt.get_bounds()

        self.calculator = Calculator(self.direc, self.ff_to_opt, commands=args_ff)

        self.initial_score = self.calculate_and_score(self.param_coord)

    @abstractmethod
    def calculate_and_score(self, coord):
        self.ff_to_opt.set_param_values(coord)
        self.calculator.update_ff(self.ff_to_opt)
        self.calculator.calculate(coord, self.commands)
        return
    
    @catch_run_errors
    def run(
        self,
        convergence_precision=0.001
    ):
        """
        Once all attributes are setup as you so desire, run this method to
        optimize the parameters.

        Returns
        -------
        `datatypes.FF` (or subclass)
            Contains the best parameters found.
        """

        # calculate initial FF results
        data = calculate.main(self.args_ff)
        c_dict = compare.data_by_type(data)
        r_dict, c_dict = compare.trim_data(self.r_dict, c_dict)
        self.ff.score = compare.compare_data(r_dict, c_dict)

        logger.log(logging.INFO, "~~ HYBRID OPTIMIZATION ~~".rjust(79, "~"))
        logger.log(logging.INFO, "INIT FF SCORE: {}".format(self.ff.score))
        opt.pretty_ff_results(self.ff, level=20)

        self.best_ff_params, self.best_ff_score = self.hybrid_opt.run(
            precision=convergence_precision, strategy=strategy
        )

        # replace initial ff params with best params
        assert len(self.best_ff_params) == len(self.ff.params)

        self.best_ff = copy.deepcopy(
            self.ff
        )  # schrod_indep_filetypes.FF(path=(self.ff.path[:-4] +'.hybrid.fld'), params=self.best_ff_params, score=self.best_ff_score)
        self.best_ff.path = self.best_ff.path[:-4] + ".hybrid.fld"
        self.best_ff.set_param_values(self.best_ff_params)
        self.best_ff.score = self.best_ff_score

        logger.log(logging.INFO, "BEST:")
        pretty_ff_results(self.best_ff, level=20)
        logger.log(logging.INFO, "~~ END HYBRID CYCLE ~~".rjust(79, "~"))

        if self.best_ff.score < self.ff.score:
            logger.log(logging.INFO, "~~ HYBRID FINISHED WITH IMPROVEMENTS ~~".rjust(79, "~"))
        else:
            logger.log(logging.INFO, "~~ HYBRID FINISHED WITHOUT IMPROVEMENTS ~~".rjust(79, "~"))
            # This restores the inital parameters, so no need to use
            # restore_simp_ff here.
            self.best_ff = self.ff

        pretty_ff_results(self.ff, level=20)
        pretty_ff_results(self.best_ff, level=20)
        logger.log(logging.INFO, "  -- Writing best force field from Hybrid Optimization.")
        self.best_ff.export_ff(self.best_ff.path)
        return self.best_ff

class SerialOptimizer(Optimizer):

    def __init__(self, direc=None, ff_to_opt=None, args_ff=None, ref_data=None):
        super().__init__(direc, ff_to_opt, args_ff, ref_data)

    def calculate_and_score(self, coord):
        return super().calculate_and_score(coord)


class ParallelOptimizer(Optimizer):

    def __init__(self, direc, ff_to_opt:FF, args_ff, ref_data, num_threads:int, num_ff_candidates:int):
        super().__init__(direc, ff_to_opt, args_ff, ref_data)

        self.dir:Path = direc

        assert(num_threads <= num_ff_candidates)
        self.num_threads = num_threads
        self.num_ff_candidates = num_ff_candidates
        self.sub_dirs:List[Path] = [Path(os.path.join(self.dir, 'calc_{}'.format(ff_num))) for ff_num in range(len(force_fields))]
        self.force_fields = []
        self.calculators:List[Calculator] = []

        # set up and initialize the Swarm Optimizer or whatever it is
        # get all of the new particle coordinates back
        # create ff objects for all of those particle coordinates

        self.setup_parallel_calc()

        self.setup_worker_pool()
        # create worker pool, reuse throughout
        # pass pool to Calculator in the calculate_and_score method where it calls the Calculator.calculate() - ???

        self.calculate_and_score()

    def setup_worker_pool(self):

        # then the export of the ff will take care of itself, will just need to pass worker num to calculate and score
        return

    def setup_parallel_calc(self):

        # setup parallel directories
        # ask the respective EngineCalculator to setup calculations given a set of directories to use, num threads, num ffs
        # just type plan for an EngineCalculator since it will inherit from EngineCalculator so all should have correct
        #  implemented and overridden methods (should all be the same calls here, differences all handled in the Calculator classes)
        # for AMBER, I think we should have a reference directory, an input directory, and then the calc_# directories for each
        #  thread and an intermediates directory which contains all the intermediate files (cycle intermediates)
        # Upon successful completion/convergence, the calc_# directories are all deleted (perhaps have a flag for this for debugging)
        # after the final files are copied to a final_output directory where the compare is rerun to ensure correct data
        
        for i in range(self.num_ff_candidates):
            sp.call("mkdir calc_"+str(i))
            # copy in the necessary files
            # create and append the Calculator classes
            calc = Calculator(self.dir.joinpath('calc_'+str(i)), self.force_fields[i], self.commands)
            self.calculators.append(calc)

        return


    
    def calculate_and_score(self, coords):
        # use worker pool
        # if coords is none then don't update the ff
        for calculator in self.calculators:
            calculator.calculate(coords, self.commands)
        
        return #super().calculate_and_score(coord)



class SwarmOptimizer(ParallelOptimizer): 

    # region SWARM OPTIMIZER HYPERPARAMETER CONFIGURATIONS
    TIGHT_SEARCH_CONFIG = {
        "vectorize_func": False,
        "taper_GA": True,
        "taper_mutation": True,
        "skew_social": True,
        "mutation_strategy": "DE/best/1",
        "differential_weight": (0.4, 0.1),
        "recomb_constant": (0.7, 0.7),
        "inertia": (0.7, 0.4),  # LDIW strategy
        "cognitive": (2.0, 0.5),
        "social": (1.0, 2.5),
    }

    GLOBAL_SEARCH_CONFIG = (
        {  # NOTE: user should also increase population size, global has slower convergence
            "vectorize_func": False,
            "taper_GA": True,
            "taper_mutation": True,
            "skew_social": True,
            "mutation_strategy": "DE/best/1",
            "differential_weight": (0.7, 0.1),
            "recomb_constant": (0.7, 0.7),
            "inertia": (0.9, 0.4),  # LDIW strategy
            "cognitive": (2.5, 0.5),
            "social": (0.5, 2.5),
        }
    )
    # widens the search radius but slows convergence

    # endregion

    def __init__(
        self,
        direc=None,
        ff: FF = None,
        ff_lines=None,
        args_ff=None,
        args_ref=None,
        bias_to_current=True,
        loose_bounds=False,
        ref_data=None,
        tight_spread='T',
        num_ho_cores=1,
        max_iter=1000,
        pop_size=24,
        ff_row_expand_bounds:int=None
    ):
        super(SwarmOptimizer, self).__init__(direc, ff, ff_lines, args_ff, args_ref)

        lower_bounds = []
        upper_bounds = []
        deviations = []
        tighter_spread = tight_spread == 'TT'
        tight_spread = 'T' in tight_spread
        for param in self.ff.params:
            lower_bounds.append(param.allowed_range[0])
            upper_bounds.append(param.allowed_range[1])
            if param.ptype == "af":
                #lower_bounds.append(0.1)
                #upper_bounds.append(7.0)
                deviations.append(0.125) if tight_spread else deviations.append(1.0)
            elif param.ptype == "bf":
                #lower_bounds.append(15.0) if ff_row_expand_bounds == param.ff_row else lower_bounds.append(0.1)
                #upper_bounds.append(30.0) if ff_row_expand_bounds == param.ff_row else upper_bounds.append(7.0)
                deviations.append(0.125) if tight_spread else deviations.append(1.0)
            elif param.ptype == "ae":
                #lower_bounds.append(0.0)
                #upper_bounds.append(180.0)
                deviations.append(15.0)
            elif param.ptype == "be":
                #lower_bounds.append(0.0)
                #upper_bounds.append(6.0)
                deviations.append(0.5)  # TODO reassess
            elif param.ptype == "df":
                #lower_bounds.append(-5.0)
                #upper_bounds.append(5.0)
                deviations.append(np.inf)
            elif (
                param.ptype == "q"
            ):  # TODO MF - this may be removed bc charges will now always be parameterized with mjESP or mgESP in a linear fashion following P-ON 2024 work
                lower_bounds.append(-10.0)
                upper_bounds.append(10.0)
                deviations.append(2.0) if param.value != 0 else deviations.append(10.0)
            elif param.ptype == "imp1" | param.ptype == "imp2":
                lower_bounds.append(0.0)
                upper_bounds.append(50.0)
                deviations.append(np.inf)
            elif param.ptype == "sb":
                lower_bounds.append(0.0)
                upper_bounds.append(50.0)
                deviations.append(np.inf)
            else:
                raise ("Parameter type not supported: " + param.ptype)

        logger.log(logging.INFO, upper_bounds)
        ff_params = [param.value for param in self.ff.params]

        param_opt_config = {
            "lb": lower_bounds,
            "ub": upper_bounds,
            "size_pop": pop_size,
            "max_iter": max_iter,
            "initial_guesses": ff_params if bias_to_current else None,
            "guess_deviation": deviations,
            "guess_ratio": 0.7 if tight_spread else 0.3,
        }
        self.opt_config = (
            {**param_opt_config, **self.TIGHT_SEARCH_CONFIG}
            if tight_spread
            else {**param_opt_config, **self.GLOBAL_SEARCH_CONFIG}
        )

        if ref_data is None:
            self.ref_data = opt.return_ref_data(self.args_ref)
        else:
            self.ref_data = ref_data
        self.r_dict = compare.data_by_type(self.ref_data)

        if num_ho_cores >= 1: #TODO this should just be > not >=, testing now
            self.setup_parallel_licenses_directories(num_ho_cores)
        # if num_ho_cores > 1:
        #    assert struct_file_locks is not None, "If processing in parallel, there must be protections (locks) on shared resources such as structure files."
        set_run_mode(self.calculate_and_score, "multiprocessing")

        self.hybrid_opt = PSO_DE(
            self.calculate_and_score,
            len(self.ff.params),
            config=self.opt_config,
            func_args=self.r_dict,
            n_processes=self.num_ff_threads,
            pass_particle_num=True,
            verbose=True,
            bounds_strategy=Bounds_Handler.REFLECTIVE,
        )

        self.setup_ff_pool()

        self.hybrid_opt.Y = self.hybrid_opt.cal_y()
        self.hybrid_opt.update_pbest()
        self.hybrid_opt.update_gbest()
        self.hybrid_opt.recorder()

        def calculate_and_score(self, ref_dict, enumerable_input) -> float:

        ff_num, parameter_set = enumerable_input
        if ff_num is not None:
            ff = self.pool_ff_objects[ff_num]
            logger.log(logging.DEBUG, "FF Num: " + str(ff_num))
        else:
            ff = self.ff

        # send ffs to Calculator to calculate

        logger.log(logging.INFO, "FF " + str(ff_num) + " Score: " + str(score))
        print("FF " + str(ff_num) + " Score: " + str(score))
        return ff.get_score()

class GradOptimizer(SerialOptimizer):

    def __init__(self, direc=None, ff_to_opt=None, args_ff=None, ref_data=None):
        super().__init__(direc, ff_to_opt, args_ff, ref_data)


def dependency_check(ff:FF, ff_args, store_data=False, variance=0.2):

    heuristic_by_param_variation = dict()
    mod_ff = copy.deepcopy(ff)
    heuristic_by_param_variation['baseline'] = cal_ff(ff, ff_args, parent_ff=ff, store_data=store_data) # this will call AmberCalculator

    for param_index in range(len(ff.params)):
        heuristic_by_param_variation[0] = []
        mod_ff.params[param_index].value = ff.params[param_index].value + (variance * ff.params[param_index].value)
        mod_ff.export_ff()
        forward = calculate.main(ff_args)
        heuristic_by_param_variation[0].append(forward)
        mod_ff.params[param_index].value = ff.params[param_index].value - (variance * ff.params[param_index].value)
        mod_ff.export_ff()
        backward = calculate.main(ff_args)
        heuristic_by_param_variation[0].append(backward)
        mod_ff.params[param_index].value = ff.params[param_index].value
    return heuristic_by_param_variation



def pretty_ff_params(ffs, level=20):
    """
    Shows parameters from many force fields.

    Parameters
    ----------
    ffs : list of `datatypes.FF` (or subclass)
    level : int
    """
    if logger.getEffectiveLevel() <= level:
        wrapper = textwrap.TextWrapper(width=79, subsequent_indent=' '*29)
        logger.log(
            level,
            '--' + ' PARAMETER '.ljust(25, '-') +
            '--' + ' VALUES '.ljust(48, '-') +
            '--')
        for i in range(0, len(ffs[0].params)):
            wrapper.initial_indent = ' {:25s} '.format(repr(ffs[0].params[i]))
            all_param_values = [x.params[i].value for x in ffs]
            all_param_values = ['{:8.4f}'.format(x) for x in all_param_values]
            logger.log(level, wrapper.fill(' '.join(all_param_values)))
        logger.log(level, '-' * 79)

def pretty_ff_results(ff, level=20):
    """
    Shows a force field's method, parameters, and score.

    Parameters
    ----------
    ff : `datatypes.FF` (or subclass)
    level : int
    """
    if logger.getEffectiveLevel() <= level:
        wrapper = textwrap.TextWrapper(width=79)
        logger.log(level, ' {} '.format(ff.method).center(79, '='))
        logger.log(level, 'SCORE: {}'.format(ff.score))
        logger.log(level, 'PARAMETERS:')
        logger.log(level, wrapper.fill(' '.join(map(str, ff.params))))
        logger.log(level, '=' * 79)
        logger.log(level, '')









