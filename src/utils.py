import argparse

__cline_parser = []
__args = []

def create_cline_parser(description=''):
    assert len(__cline_parser) == 0
    __cline_parser.append(argparse.ArgumentParser(description=description,
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter))

def add_arg(*pargs, **kwargs):
    if not __cline_parser:
        create_cline_parser()
    __cline_parser[0].add_argument(*pargs, **kwargs)

def cl_args():
    assert __cline_parser
    if not __args:
        __args.append(__cline_parser[0].parse_args())
    return __args[0]
