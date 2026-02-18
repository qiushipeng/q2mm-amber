#!/usr/bin/env python
"""
Extracts data from reference files or calculates FF data.

#TODO: MF - these should all be methods that calculate can call , replaces the collect_data method
 and breaks it down into multiple methods and modules, organized by the type of calculation or by the engine, need
 to decide.

Takes a sequence of keywords corresponding to various
datatypes (ex. mb = MacroModel bond lengths) followed by filenames,
and extracts that particular data type from the file.

Note that the order of filenames IS IMPORTANT!

Used to manage calls to MacroModel but that is now done in the
Mae class inside schrod_indep_filetypes. I'm still debating if that should be
there or here. Will see how this framework translates into
Amber and then decide.
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import logging.config
from pathlib import Path
from typing import List
import numpy as np
import os
import sys
from abc import ABC, abstractmethod

# I don't really want to import all of chain if possible. I only want
# chain.from_iterable.
# chain.from_iterable flattens a list of lists similar to:
#   [child for parent in grandparent for child in parent]
# However, I think chain.from_iterable works on any number of nested lists.
from itertools import chain
from textwrap import TextWrapper

import constants as co
from utilities import Frcmod
from data_structs import FF, AmberFF

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)

class Calculator(ABC):
    """Manages the methods, inputs, and files necessary to run back-end calculations with external engines like AMBER, acts as an interface with these engines
    """    

    def __new__(cls, ff):
        subclass_map = {type(subclass.ff): subclass for subclass in cls.__subclasses__()}
        subclass = subclass_map[type(ff)]
        instance = super(Calculator, subclass).__new__(subclass)
        return instance
        
    def __init__(self, work_dir:Path, ff, commands) -> None:
        self.working_directory:Path = work_dir
        self.ff = ff
        self.commands = commands

        self.calculate(self.ff, self.commands)
        pass

    @abstractmethod
    def update_ff(self, ff:FF):
        return

    @abstractmethod
    def calculate_geometry(self): # TODO I don't see why this can't encapsulate bonds, angles, torsions...
        return

    @abstractmethod
    def calculate_hessian(self):
        return

    @abstractmethod
    def calculate_energy(self): # TODO do we even use/need this? we don't do energy, just Hessian, but both energy and opt energy in main codebase
        return

    @abstractmethod
    def calculate(self, new_param_values, commands=None): # replaces the overly complicated case of different objects for each calculation type/file and it running itself
        return

    @abstractmethod
    def gather_results(self): # replaces collect_data, effectively
        return

class AmberCalculator(Calculator):
    ff_type = type(AmberFF)

    def __init__(self, work_dir: Path, ff: AmberFF, commands, ff_name) -> None:
        super().__init__(work_dir, ff, commands)
        # when changed AmberCalculator will create a file instance and write out the new ff
        # have this be a list of FFs, have it manage the calc_# directories and pool and just reuse the same .in etc, this might be best although I like the idea of a calculator per thread
        # alternatively, this is a file object classed in utilities which contains a FF object classed in data_structs
        self.topology = None # this is a file object classed in utilities, this will just be passed as the same one in parallel cases
        self.leap_input = None # this is a file object classed in utilities
        self.frcmod = Frcmod(self.working_directory.joinpath(ff_name))


        # change constant TODO see note in constants.py on how to make this more consistent, simpler
        co.STEPS["bf"] = 10.00
        co.STEPS["af"] = 10.0
        co.STEPS["df"] = 10.0

    #FF property which will write a new forcefield when it is updated, automatically

    def write_scripts(self):
        #write scripts for analysis
        return

    def update_ff(self, ff:AmberFF):
        self.frcmod.force_field = ff
        self.update_topology()

    def update_topology(self): # TODO frcmod file should already be updated and written, but perhaps we have a stale flag to ensure it is re-written if not
        
        # delete existing prmtop so that it will not use pre-existing one if the leap script fails
        # make and write leap input file if doesn't already exist, confirm that leap input file FF property is correct, matches this one's, run leap input
        # confirm that prmtop and inpcrd exist

        return

    def calculate_geometry(self): # TODO I don't see why this can't encapsulate bonds, angles, torsions...
        return

    def calculate_hessian(self):
        return
    
    def calculate_energy(self): # TODO do we even use/need this? we don't do energy, just Hessian, but both energy and opt energy in main codebase
        return


    def calculate(self, new_param_values, commands=None): # replaces the overly complicated case of different objects for each calculation type/file and it running itself
        self.frcmod.ff.set_param_values(new_param_values)
        self.update_topology()

        return
    
    def gather_results(self): # replaces collect_data, effectively; calls file io to return data structs of results
        return
    


#TODO: MF - calculate kept separate from gather_results bc, when running in parallel, might want to start calculations but not gather them until they have completed
# however, likely that we will do both in one go bc running multiple candidate FFs in the same folder for a single round, if so then just merge into one method

# Have the HO just pass the set of particles and their coordinates to the swarm optimizer
# swarm optimizer then pops these back to a set of FF data objects,
# these then get passed to the Calculator which updates the respective files, rewriting them
# Calculator then initializes or continues a pool of workers which run the files
# Once all calculations/scripts complete, the Calculator then retrieves the results from the files
# Swarm optimizer or optimizer then runs the compare and returns the scores to the HO

    #endregion Parallelization
