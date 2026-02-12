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

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)

class Calculator(ABC):
    """Manages the methods, inputs, and files necessary to run back-end calculations with external engines like AMBER, acts as an interface with these engines
    """    
    def __init__(self, work_dir:Path, ff:FF) -> None:
        self.working_directory:Path = work_dir
        self.ff = ff
        pass

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
    def calculate(self, amber_commands): # replaces the overly complicated case of different objects for each calculation type/file and it running itself
        return

    @abstractmethod
    def gather_results(self): # replaces collect_data, effectively
        return

class ParallelCalculator(ABC):
    """Manages the methods, inputs, and files necessary to run back-end calculations with external engines like AMBER, acts as an interface with these engines
    """    
    def __init__(self, work_dir:Path, ff:FF) -> None:
        self.working_directory:Path = work_dir
        self.ff = ff
        pass

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
    def calculate(self, amber_commands): # replaces the overly complicated case of different objects for each calculation type/file and it running itself
        return

    @abstractmethod
    def gather_results(self): # replaces collect_data, effectively
        return

#TODO: MF - calculate kept separate from gather_results bc, when running in parallel, might want to start calculations but not gather them until they have completed
# however, likely that we will do both in one go bc running multiple candidate FFs in the same folder for a single round, if so then just merge into one method

class AmberCalculator(Calculator):

    def __init__(self):
        super(Calculator, self).__init__()
        self.sub_names = []
        self._atom_types = None
        self._lines = None
        self.ff = None
        # 3 options, this paradigm should be established in the superclass:
        # ^ this is a data object classed in data_structs, when changed AmberCalculator will create a file instance and write out the new ff
        # a 3rd option is to have this be a list of FFs, have it manage the calc_# directories and pool and just reuse the same .in etc, this might be best although I like the idea of a calculator per thread
        # alternatively, this is a file object classed in utilities which contains a FF object classed in data_structs
        self.topology = None # this is a file object classed in utilities
        self.leap_input = None # this is a file object classed in utilities
        # change constant
        co.STEPS["bf"] = 10.00
        co.STEPS["af"] = 10.0
        co.STEPS["df"] = 10.0

    def update_topology(self):
        return

    def calculate_geometry(self): # TODO I don't see why this can't encapsulate bonds, angles, torsions...
        return

    def calculate_hessian(self):
        return
    
    def calculate_energy(self): # TODO do we even use/need this? we don't do energy, just Hessian, but both energy and opt energy in main codebase
        return

    def calculate(self, amber_commands): # replaces the overly complicated case of different objects for each calculation type/file and it running itself

        self.update_topology()

        return
    
    def gather_results(self): # replaces collect_data, effectively
        return
