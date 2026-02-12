#!/usr/bin/env python
"""
Extracts data from reference files or calculates FF data.

#TODO: MF - when refactor, try to get rid of calling main to calculate, these should all be methods
 main() can just call those methods...

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
import numpy as np
import os
import sys

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

#region to Utilities?

# Right now, this only looks good if the logger doesn't append each log
# message with something (module, date/time, etc.).
# It would be great if this output looked good regardless of the settings
# used for the logger.
# That goes for all of these pretty output functions that use TextWrapper.
def pretty_commands_for_files(commands_for_files, log_level=5):
    """
    Logs the .mae commands dictionary, or the all of the commands
    used on a particular file.

    Arguments
    ---------
    commands_for_files : dic
    log_level : int
    """
    if logger.getEffectiveLevel() <= log_level:
        foobar = TextWrapper(
            width=48, subsequent_indent=' '*26)
        logger.log(
            log_level,
            '--' + ' FILENAME '.center(22, '-') +
            '--' + ' COMMANDS '.center(22, '-') +
            '--')
        for filename, commands in commands_for_files.items():
            foobar.initial_indent = '  {:22s}  '.format(filename)
            logger.log(log_level, foobar.fill(' '.join(commands)))
        logger.log(log_level, '-'*50)

def pretty_all_commands(commands, log_level=5):
    """
    Logs the arguments/commands given to calculate that are used
    to request particular datatypes from particular files.

    Arguments
    ---------
    commands : dic
    log_level : int
    """
    if logger.getEffectiveLevel() <= log_level:
        foobar = TextWrapper(width=48, subsequent_indent=' '*24)
        logger.log(log_level, '')
        logger.log(
            log_level,
            '--' + ' COMMAND '.center(9, '-') +
            '--' + ' GROUP # '.center(9, '-') +
            '--' + ' FILENAMES '.center(24, '-') +
            '--')
        for command, groups_filenames in commands.items():
            for i, filenames in enumerate(groups_filenames):
                if i == 0:
                    foobar.initial_indent = \
                        '  {:9s}  {:^9d}  '.format(command, i+1)
                else:
                    foobar.initial_indent = \
                        '  ' + ' '*9 + '  ' + '{:^9d}  '.format(i+1)
                logger.log(log_level, foobar.fill(' '.join(filenames)))
        logger.log(log_level, '-'*50)

def pretty_data(data, log_level=20):
    """
    Logs data as a table.

    Arguments
    ---------
    data : list of Datum
    log_level : int
    """
    # Really, this should check every data point instead of only the 1st.
    if not data[0].wht:
        import_weights(data)
    if log_level:
        string = ('--' + ' LABEL '.center(22, '-') +
                  '--' + ' WEIGHT '.center(22, '-') +
                  '--' + ' VALUE '.center(22, '-') +
                  '--')
        logger.log(log_level, string)
    for d in data:
        if d.wht or d.wht == 0:
            string = ('  ' + '{:22s}'.format(d.lbl) +
                      '  ' + '{:22.4f}'.format(d.wht) +
                      '  ' + '{:22.4f}'.format(d.val))
        else:
            string = ('  ' + '{:22s}'.format(d.lbl) +
                      '  ' + '{:22.4f}'.format(d.val))
        if log_level:
            logger.log(log_level, string)
        else:
            print(string)
    if log_level:
        logger.log(log_level, '-' * 50)

def sort_commands_by_filename(commands):
    '''
    Takes a dictionary of commands like...

     {'me': [['a1.01.mae', 'a2.01.mae', 'a3.01.mae'],
             ['b1.01.mae', 'b2.01.mae']],
      'mb': [['a1.01.mae'], ['b1.01.mae']],
      'jeig': [['a1.01.in,a1.out', 'b1.01.in,b1.out']]
     }

    ... and turn it into a dictionary that looks like...

    {'a1.01.mae': ['me', 'mb'],
     'a1.01.in': ['jeig'],
     'a1.out': ['jeig'],
     'a2.01.mae': ['me'],
     'a3.01.mae': ['me'],
     'b1.01.mae': ['me', 'mb'],
     'b1.01.in': ['jeig'],
     'b1.out': ['jeig'],
     'b2.01.mae': ['me']
    }

    Arguments
    ---------
    commands : dic

    Returns
    -------
    dictionary of the sorted commands
    '''
    sorted_commands = {}
    for command, groups_filenames in commands.items():
        for comma_separated in chain.from_iterable(groups_filenames):
            for filename in comma_separated.split(','):
                if filename in sorted_commands:
                    sorted_commands[filename].append(command)
                else:
                    sorted_commands[filename] = [command]
    return sorted_commands

#endregion to Utilities?

def calculate(opts):
    commands = {key: value for key, value in opts.__dict__.items() if key
                in co.COM_ALL and value}
    # Add in the empty commands. I'd rather not do this, but it makes later
    # coding when collecting data easier.
    for command in co.COM_ALL:
        if command not in commands:
            commands.update({command: []})
    pretty_all_commands(commands)
    # This groups all of the data type commands associated with one file. TODO: MF - this seems entirely unncessary if we just read all of the data from the file that Q2MM could need...
    # commands_for_filenames looks like:
    # {'a1.01.mae': ['me', 'mb'],
    #  'a1.01.in': ['jeig'],
    #  'a1.out': ['jeig'],
    #  'a2.01.mae': ['me'],
    #  'a3.01.mae': ['me'],
    #  'b1.01.mae': ['me', 'mb'],
    #  'b1.01.in': ['jeig'],
    #  'b1.out': ['jeig'],
    #  'b2.01.mae': ['me']
    # }
    commands_for_filenames = sort_commands_by_filename(commands)
    pretty_commands_for_files(commands_for_filenames)
    # This dictionary associates the filename that the user supplied with
    # the command file that has to be used to execute some backend software
    # calculate in order to retrieve the data that the user requested.
    # inps looks like:
    # {'a1.01.mae': <__main__.Mae object at 0x1110e10>,
    #  'a1.01.in': None,
    #  'a1.out': None,
    #  'a2.01.mae': <__main__.Mae object at 0x1733b23>,
    #  'a3.01.mae': <__main__.Mae object at 0x1853e12>,
    #  'b1.01.mae': <__main__.Mae object at 0x2540e10>,
    #  'b1.01.in': None,
    #  'b1.out': None,
    #  'b2.01.mae': <__main__.Mae object at 0x1353e11>,
    # }
    inps = {}
    # This generates any of the necessary command files. It uses
    # commands_for_filenames, which contains all of the data types associated
    # with the given file.
    # Stuff below doesn't need both comma separated filenames simultaneously.
    for filename, commands_for_filename in commands_for_filenames.items():
        logger.log(1, '>>> filename: {}'.format(filename))
        logger.log(1, '>>> commands_for_filename: {}'.format(
            commands_for_filename))
        # These next two if statements will break down what command files
        # have to be written by the backend software package.

        # Gausssian to Amber
        if any(x in ['gaa','gab','gat','gaao','gabo','gato'] for x in commands_for_filename):
            if os.path.splitext(filename)[1] == ".log":
                inps[filename] = schrod_indep_filetypes.AmberLeap_Gaus(
                    os.path.join(opts.directory, filename))
                inps[filename].commands = commands_for_filename
                    
        elif any(x in co.COM_AMBER for x in commands_for_filename):
            if os.path.splitext(filename)[1] == ".in": # leap.in as for now
                inps[filename] = schrod_indep_filetypes.AmberLeap(os.path.join(opts.directory, filename))
                inps[filename].commands = commands_for_filename
            # This doesn't work.
            # We need to know both filenames simultaneously for this Amber crap.
            # Have to add these to `inps` in some other way.
            # pass
        # In this case, no command files have to be written.
        else:
            inps[filename] = None
    # Stuff below needs both comma separated filenames simultaneously.
    # Do the Amber inputs.
    # Leaving the filenames together because Taylor said this would work well.
#    for comma_sep_filenames in flatten(commands['ae']):
#        # Maybe make more specific later.
#        inps[comma_sep_filenames] = schrod_indep_filetypes.AmberInput(
#            'DOES_PATH_EVEN_MATTER')
#        split_it = comma_sep_filenames.split(',')
#        inps[comma_sep_filenames].directory = opts.directory
#        inps[comma_sep_filenames].inpcrd = split_it[0]
#        inps[comma_sep_filenames].prmtop = split_it[1]
    logger.log(1, '>>> commands: {}'.format(commands))
    # Check whether or not to skip calculations.
    if opts.norun or opts.fake:
        logger.log(15, "  -- Skipping backend calculations.")
    else:
        for filename, some_class in inps.items():
            logger.log(1, '>>> filename: {}'.format(filename))
            logger.log(1, '>>> some_class: {}'.format(some_class))
            # Works if some class is None too.
            if hasattr(some_class, 'run'):
                # Ideally this can be the same for each software backend,
                # but that means we're going to have to make some changes
                # so that this token argument is handled properly.
                some_class.run(check_tokens=opts.check)
    # `data` is a list comprised of schrod_indep_filetypes.Datum objects.
    # If we remove/with sorting removed, the Datum class is less
    # useful. We may want to reduce this to a N x 3 matrix or
    # 3 vectors (labels, weights, values).
    sub_names = ['OPT']
    if opts.subnames:
        sub_names = opts.subnames
    if opts.fake:
        data = collect_data_fake(
            commands, inps, direc=opts.directory, invert=opts.invert,
            sub_names=sub_names)
    else:
        data = collect_data(
            commands, inps, direc=opts.directory, invert=opts.invert,
            sub_names=sub_names)
    # Adds weights to the data points in the data list.
    if opts.weight:
        import_weights(data)
    # Optional printing or logging of data.
    if opts.doprint:
        pretty_data(data, log_level=None)
    return data



def main(args):
    """
    Arguments
    ---------
    args : string or list of strings
           Evaluated using parser returned by return_calculate_parser(). If
           it's a string, it will be converted into a list of strings.
    """
    # Should be a list of strings for use by argparse. Ensure that's the case.
    # basestring is deprecated in python3, str is probably safe to use in both
    # but should be tested, for now sys.version_info switch can handle it
    if sys.version_info > (3, 0):
        if isinstance(args, str):
            args = args.split()
    else:
        if isinstance(args, basestring):
            args = args.split()
    parser = return_calculate_parser()
    opts = parser.parse_args(args)
    # This makes a dictionary that only contains the arguments related to
    # extracting data from everything in the argparse dictionary, opts.
    # Given that the user supplies:
    # python calculate.py -me a1.01.mae a2.01.mae a3.01.mae -me b1.01.mae
    #    b2.01.mae -mb a1.01.mae b1.01.mae -jeig a1.01.in,a1.out
    #    b1.01.in,b1.out
    # commands looks like:
    # {'me': [['a1.01.mae', 'a2.01.mae', 'a3.01.mae'],
    #         ['b1.01.mae', 'b2.01.mae']],
    #  'mb': [['a1.01.mae'], ['b1.01.mae']],
    #  'jeig': [['a1.01.in,a1.out', 'b1.01.in,b1.out']]
    # }
    calculate(opts)

def return_calculate_parser(add_help=True, parents=None):
    '''
    Command line argument parser for calculate.

    Arguments
    ---------
    add_help : bool
               Whether or not to add help to the parser. Default
               is True.
    parents : argparse.ArgumentParser
              Parent parser incorporated into this parser. Default
              is None.
    '''
    # Whether or not to add parents parsers. Not sure if/where this may be used
    # anymore.
    if parents is None: parents = []
    # Whether or not to add help. You may not want to add help if these
    # arguments are being used in another, higher level parser.
    if add_help:
        parser = argparse.ArgumentParser(
            description=__doc__, parents=parents)
    else:
        parser = argparse.ArgumentParser(
            add_help=False, parents=parents)
    # region GENERAL OPTIONS
    opts = parser.add_argument_group("calculate options")
    opts.add_argument(
        '--append', '-a', type=str, metavar='sometext',
        help='Append this text to command files generated by Q2MM.')
    opts.add_argument(
        '--directory', '-d', type=str, metavar='somepath', default=os.getcwd(),
        help=('Directory searched for files '
              '(ex. *.mae, *.log, mm3.fld, etc.). '
              'Subshell commands (ex. MacroModel) are executed from here. '
              'Default is the current directory.'))
    opts.add_argument(
        '--doprint', '-p', action='store_true',
        help=("Logs data. Can generate extensive log files."))
    opts.add_argument(
        '--fake', action='store_true',
        help=("Generate fake data sets. Used to expedite testing."))
    opts.add_argument(
        '--ffpath', '-f', type=str, metavar='somepath',
        help=("Path to force field. Only necessary for certain data types "
              "if you don't provide the substructure name."))
    opts.add_argument(
        '--invert', '-i', type=float, metavar='somefloat',
        help=("This option will invert the smallest eigenvalue to be whatever "
              "value is specified by this argument whenever a Hessian is "
              "read."))
    opts.add_argument(
        '--nocheck', '-nc', action='store_false', dest='check', default=True,
        help=("By default, Q2MM checks whether MacroModel tokens are "
              "available before attempting a MacroModel calculation. If this "
              "option is supplied, MacroModel will not check for tokens "
              "first."))
    opts.add_argument(
        '--norun', '-n', action='store_true',
        help="Don't run 3rd party software.")
    opts.add_argument(
        '--subnames',  '-s', type=str, nargs='+',
        metavar='"Substructure Name OPT"',
        help=("Names of the substructures containing parameters to "
              "optimize in a mm3.fld file."))
    opts.add_argument(
        '--weight', '-w', action='store_true',
        help='Add weights to data points.')
    # endregion GENERAL OPTIONS

    # region GAUSSIAN OPTIONS
    gau_args = parser.add_argument_group("gaussian reference data types")
    gau_args.add_argument(
        '-gta', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian angles using Tinker.'))
    gau_args.add_argument(
        '-gtb', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian bonds using Tinker.'))
    gau_args.add_argument(
        '-gtt', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian torsions using Tinker.'))
    gau_args.add_argument(
        '-gaa', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian angles using Amber.'))
    gau_args.add_argument(
        '-gab', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian bonds using Amber.'))
    gau_args.add_argument(
        '-gat', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian torsions using Amber.'))
    gau_args.add_argument(
        '-gaao', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian angles using Amber (POST OPT).'))
    gau_args.add_argument(
        '-gabo', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian bonds using Amber (POST OPT).'))
    gau_args.add_argument(
        '-gato', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian torsions using Amber (POST OPT).'))
    gau_args.add_argument(
        '-ge', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian energies.'))
    gau_args.add_argument(
        '-ge1', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian energy.'))
    gau_args.add_argument(
        '-gea', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian energies. Energies will be relative to the average '
              'energy within this data type.'))
    gau_args.add_argument(
        '-geo', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian energies. Same as -ge, except the files selected '
              'by this command will have their energies compared to those '
              'selected by -meo.'))
    gau_args.add_argument(
        '-ge1o', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian energy. Used for FF a1o commands.'))
    gau_args.add_argument(
        '-geao', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian energies. Same as -ge, except the files selected '
              'by this command will have their energies compared to those '
              'selected by -meo. Energies will be relative to the average '
              'energy within this data type.'))
    gau_args.add_argument(
        '-gh', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help='Gaussian Hessian extracted from a .log archive.')
    gau_args.add_argument(
        '-geigz', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian eigenmatrix. Incluldes all elements, but zeroes '
              'all off-diagonal elements. Uses only the .log for '
              'the eigenvalues and eigenvectors.'))
    # endregion GAUSSIAN OPTIONS

    # ADDITIONAL REFERENCE OPTIONS
    ref_args = parser.add_argument_group("other reference data types")
    ref_args.add_argument(
        '-r', type=str, nargs='+', action='append',
        default=[], metavar='somename.txt',
        help=('Read reference data from file. The reference file should '
              '3 space or tab separated columns. Column 1 is the labels, '
              'column 2 is the weights and column 3 is the values.'))
    
    # region AMBER OPTIONS
    amb_args = parser.add_argument_group("amber data types")
    amb_args.add_argument(
        '-ae', type=str, nargs='+', action='append',
        default=[], metavar='somename.inpcrd,somename.prmtop',
        help='Amber energies.')
    amb_args.add_argument(
        '-abo', type=str, nargs='+', action='append',
        default=[], metavar='somename.in',
        help=('Amber bonds (post-FF optimization).'))
    amb_args.add_argument(
        '-aao', type=str, nargs='+', action='append',
        default=[], metavar='somename.in',
        help=('Amber angles (post-FF optimization).'))
    amb_args.add_argument(
        '-ato', type=str, nargs='+', action='append',
        default=[], metavar='somename.in',
        help=('Amber torsion (post-FF optimization).'))
    amb_args.add_argument(
        '-ae1', type=str, nargs='+', action='append',
        default=[], metavar='somename.in',
        help='Amber energy (pre-FF optimization).')
    amb_args.add_argument(
        '-ae1o', type=str, nargs='+', action='append',
        default=[], metavar='somename.in',
        help='Amber energy (post-FF optimization).')
    amb_args.add_argument(
        '-ah', type=str, nargs='+', action='append',
        default=[], metavar='somename.in',
        help='Amber Hessian (post-FF optimization).')
    amb_args.add_argument(
        '-aha', type=str, nargs='+', action='append',
        default=[], metavar='somename.in',
        help='Amber Hessian (post-FF optimization).')
    # endregion AMBER OPTIONS
    
    return parser

if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])