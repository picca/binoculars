#!/usr/bin/env python3

import sys
import os
import argparse
import numpy

import binoculars.space
import binoculars.util

# INFO
def command_info(args):
    parser = argparse.ArgumentParser(prog='binoculars info')
    parser.add_argument('infile', nargs='+', help='input files, must be .hdf5')
    parser.add_argument("--config", help="display config used to generate the hdf5 file", action='store_true')
    parser.add_argument("--extractconfig", help="save config used to generate the hdf5 file in a new text file", action='store', dest='output')
    args = parser.parse_args(args)

    if args.output:
        if len(args.infile) > 1:
            print('only one space file argument is support with extractconfig -> using the first')
        config = binoculars.util.ConfigFile.fromfile(args.infile[0])
        config.totxtfile(args.output)
    else:
        for f in args.infile:
            try:
                axes = binoculars.space.Axes.fromfile(f)
            except Exception as e:
                print(f'{f}: unable to load Space: {e!r}')
            else:
                print(f'{f} \n{axes!r}')
            if args.config:
                try:
                    config = binoculars.util.ConfigFile.fromfile(f)
                except Exception as e:
                    print(f'{f}: unable to load util.ConfigFile: {e!r}')
                else:
                    print(f'{config!r}')


# CONVERT
def command_convert(args):
    parser = argparse.ArgumentParser(prog='binoculars convert')
    parser.add_argument('--wait', action='store_true', help='wait for input files to appear')
    binoculars.util.argparse_common_arguments(parser, 'project', 'slice', 'pslice', 'rebin', 'transform', 'subtract')
    parser.add_argument('--read-trusted-zpi', action='store_true', help='read legacy .zpi files, ONLY FROM A TRUSTED SOURCE!')
    parser.add_argument('infile', help='input file, must be a .hdf5')
    parser.add_argument('outfile', help='output file, can be .hdf5 or .edf or .txt')

    args = parser.parse_args(args)

    if args.wait:
        binoculars.util.statusnl(f'waiting for {args.infile} to appear')
        binoculars.util.wait_for_file(args.infile)
        binoculars.util.statusnl('processing...')

    if args.infile.endswith('.zpi'):
        if not args.read_trusted_zpi:
            print('error: .zpi files are unsafe, use --read-trusted-zpi to open')
            sys.exit(1)
        space = binoculars.util.zpi_load(args.infile)
    else:
        space = binoculars.space.Space.fromfile(args.infile)
    ext = os.path.splitext(args.outfile)[-1]

    if args.subtract:
        space -= binoculars.space.Space.fromfile(args.subtract)

    space, info = binoculars.util.handle_ordered_operations(space, args)

    if ext == '.edf':
        binoculars.util.space_to_edf(space, args.outfile)
        print(f'saved at {args.outfile}')

    elif ext == '.txt':
        binoculars.util.space_to_txt(space, args.outfile)
        print(f'saved at {args.outfile}')

    elif ext == '.hdf5':
        space.tofile(args.outfile)
        print(f'saved at {args.outfile}')

    else:
        sys.stderr.write(f'unknown extension {ext}, unable to save!\n')
        sys.exit(1)


# PLOT
def command_plot(args):
    import matplotlib.pyplot as pyplot
    import binoculars.fit
    import binoculars.plot

    parser = argparse.ArgumentParser(prog='binoculars plot')
    parser.add_argument('infile', nargs='+')
    binoculars.util.argparse_common_arguments(parser, 'savepdf', 'savefile', 'clip', 'nolog', 'project', 'slice', 'pslice', 'subtract', 'rebin', 'transform')
    parser.add_argument('--multi', default=None, choices=('grid', 'stack'))
    parser.add_argument('--fit', default=None)
    parser.add_argument('--guess', default=None)
    args = parser.parse_args(args)

    if args.subtract:
        subtrspace = binoculars.space.Space.fromfile(args.subtract)
        subtrspace, subtrinfo = binoculars.util.handle_ordered_operations(subtrspace, args, auto3to2=True)
        args.nolog = True

    guess = []
    if args.guess is not None:
        for n in args.guess.split(','):
            guess.append(float(n.replace('m', '-')))

    # PLOTTING AND SIMPLEFITTING
    pyplot.figure(figsize=(12, 9))
    plotcount = len(args.infile)
    plotcolumns = int(numpy.ceil(numpy.sqrt(plotcount)))
    plotrows = int(numpy.ceil(float(plotcount) / plotcolumns))

    for i, filename in enumerate(args.infile):
        space = binoculars.space.Space.fromfile(filename)
        space, info = binoculars.util.handle_ordered_operations(space, args, auto3to2=True)

        fitdata = None
        if args.fit:
            fit = binoculars.fit.get_class_by_name(args.fit)(space, guess)
            print(fit)
            if fit.success:
                fitdata = fit.fitdata

        if plotcount > 1:
            if space.dimension == 1 and args.multi is None:
                args.multi = 'stack'
            if space.dimension == 2 and args.multi != 'grid':
                if args.multi is not None:
                    sys.stderr.write('warning: stack display not supported for multi-file-plotting, falling back to grid\n')
                args.multi = 'grid'
            # elif space.dimension == 3:
                # not reached, project_and_slice() guarantees that
            elif space.dimension > 3:
                sys.stderr.write('error: cannot display 4 or higher dimensional data, use --project or --slice to decrease dimensionality\n')
                sys.exit(1)

        if args.subtract:
            space -= subtrspace

        basename = os.path.splitext(os.path.basename(filename))[0]

        if args.multi == 'grid':
            pyplot.subplot(plotrows, plotcolumns, i+1)
        binoculars.plot.plot(space, pyplot.gcf(), pyplot.gca(), label=basename, log=not args.nolog, clipping=float(args.clip), fit=fitdata)

        if plotcount > 1 and args.multi == 'grid':
            pyplot.gca().set_title(basename)

    if plotcount == 1:
        label = basename
    else:
        label = f'{plotcount} files'

    if args.subtract:
        label = f'{label} (subtracted {os.path.splitext(os.path.basename(args.subtract))[0]})'

    if plotcount > 1 and args.multi == 'stack':
        pyplot.legend()

    pyplot.suptitle('{}, {}'.format(label, ' '.join(info)))

    if args.savepdf or args.savefile:
        if args.savefile:
            pyplot.savefig(args.savefile)
        else:
            filename = f'{os.path.splitext(args.infile[0])[0]}_plot.pdf'
            filename = binoculars.util.find_unused_filename(filename)
            pyplot.savefig(filename)
    else:
        pyplot.show()


# FIT
def command_fit(args):
    import matplotlib.pyplot as pyplot
    import binoculars.fit
    import binoculars.plot

    parser = argparse.ArgumentParser(prog='binoculars fit')
    parser.add_argument('infile')
    parser.add_argument('axis')
    parser.add_argument('resolution')
    parser.add_argument('func')
    parser.add_argument('--follow', action='store_true', help='use the result of the previous fit as guess for the next')
    binoculars.util.argparse_common_arguments(parser, 'savepdf', 'savefile', 'clip', 'nolog')
    args = parser.parse_args(args)

    axes = binoculars.space.Axes.fromfile(args.infile)
    axindex = axes.index(args.axis)
    ax = axes[axindex]
    axlabel = ax.label
    if float(args.resolution) < ax.res:
        raise ValueError(f'interval {args.resolution} to low, minimum interval is {ax.res}')

    mi, ma = ax.min, ax.max
    bins = numpy.linspace(mi, ma, numpy.ceil(1 / numpy.float(args.resolution) * (ma - mi)) + 1)

    parameters = []
    variance = []
    fitlabel = []
    guess = None

    basename = os.path.splitext(os.path.basename(args.infile))[0]

    if args.savepdf or args.savefile:
        if args.savefile:
            filename = binoculars.util.filename_enumerator(args.savefile)
        else:
            filename = binoculars.util.filename_enumerator(f'{basename}_fit.pdf')

    fitclass = binoculars.fit.get_class_by_name(args.func)

    for start, stop in zip(bins[:-1], bins[1:]):
        info = []
        key = [slice(None) for i in axes]
        key[axindex] = slice(start, stop)
        newspace = binoculars.space.Space.fromfile(args.infile, key)
        left, right = newspace.axes[axindex].min, newspace.axes[axindex].max
        if newspace.dimension == axes.dimension:
            newspace = newspace.project(axindex)

        fit = fitclass(newspace, guess)

        paramnames = fit.parameters
        print(fit)
        if fit.success:
            fitlabel.append(numpy.mean([start, stop]))
            parameters.append(fit.result)
            variance.append(fit.variance)
            if args.follow and not fit.variance[0] == float(0):
                guess = fit.result
            else:
                guess = None
            fit = fit.fitdata
        else:
            fit = None
            guess = None

        print(guess)

        if args.savepdf or args.savefile:
            if len(newspace.get_masked().compressed()):
                if newspace.dimension == 1:
                    pyplot.figure(figsize=(12, 9))
                    pyplot.subplot(111)
                    binoculars.plot.plot(newspace, pyplot.gcf(), pyplot.gca(), label=basename, log=not args.nolog, clipping=float(args.clip), fit=fit)
                elif newspace.dimension == 2:
                    pyplot.figure(figsize=(12, 9))
                    pyplot.subplot(121)
                    binoculars.plot.plot(newspace, pyplot.gcf(), pyplot.gca(), label=basename, log=not args.nolog, clipping=float(args.clip), fit=None)
                    pyplot.subplot(122)
                    binoculars.plot.plot(newspace, pyplot.gcf(), pyplot.gca(), label=basename, log=not args.nolog, clipping=float(args.clip), fit=fit)

                info.append(f'sliced in {axlabel} from {left} to {right}')
                pyplot.suptitle('{}'.format(' '.join(info)))

                pyplot.savefig(next(filename))
                pyplot.close()

    parameters = numpy.vstack(n for n in parameters).T
    variance = numpy.vstack(n for n in variance).T

    pyplot.figure(figsize=(9, 4 * parameters.shape[0] + 2))

    for i in range(parameters.shape[0]):
        pyplot.subplot(parameters.shape[0], 1, i)
        pyplot.plot(fitlabel, parameters[i, :])
        if paramnames[i] in ['I']:
            pyplot.semilogy()
        pyplot.xlabel(paramnames[i])

    pyplot.suptitle(f'fit summary of {args.infile}')
    if args.savepdf or args.savefile:
        if args.savefile:
            root, ext = os.path.split(args.savefile)
            pyplot.savefig(f'{root}_summary{ext}')
            print(f'saved at {root}_summary{ext}')
            filename = '{}_summary{}'.format(root, '.txt')
        else:
            pyplot.savefig(f'{os.path.splitext(args.infile)[0]}_summary.pdf')
            print(f'saved at {os.path.splitext(args.infile)[0]}_summary.pdf')
            filename = f'{os.path.splitext(args.infile)[0]}_summary.txt'

        file = open(filename, 'w')
        file.write('L\t')
        file.write('\t'.join(paramnames))
        file.write('\n')
        for n in range(parameters.shape[1]):
            file.write(f'{fitlabel[n]}\t')
            file.write('\t'.join(numpy.array(parameters[:, n], dtype=numpy.str)))
            file.write('\n')
        file.close()


# PROCESS
def command_process(args):
    import binoculars.main

    binoculars.util.register_python_executable(__file__)
    binoculars.main.Main.from_args(args)  # start of main thread


# SUBCOMMAND ARGUMENT HANDLING
def usage(msg=''):
    print("""usage: binoculars COMMAND ...
{1}
available commands:

 convert    mathematical operations & file format conversions
 info       basic information on Space in .hdf5 file
 fit        crystal truncation rod fitting
 plot       1D & 2D plotting (parts of) Space and basic fitting
 process    data crunching / binning

run binoculars COMMAND --help more info on that command
""".format(sys.argv[0], msg))
    sys.exit(2)


def main():
    binoculars.space.silence_numpy_errors()

    subcommands = {'info': command_info, 'convert': command_convert, 'plot': command_plot, 'fit': command_fit, 'process': command_process}

    if len(sys.argv) < 2:
        usage()
    subcommand = sys.argv[1]
    if subcommand in ('-h', '--help'):
        usage()
    if subcommand not in subcommands:
        usage(f"binoculars error: unknown command '{subcommand}'\n")

    subcommands[sys.argv[1]](sys.argv[2:])
