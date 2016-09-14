from __future__ import print_function

import itertools
import subprocess as sp

dim_vals = map(str, range(2,4)) # incompressible in 1D produces error
const_eqns = ['lin_elastic', 'neo_hookean']

product = itertools.product(dim_vals, const_eqns)
command = ['python', 'test_unit_domain.py', '-d', None, '-m', None]

for prod in product:

    print('*'*80 + '\n' + '*'*80)
    print('dim = %s, material = \'%s\'' % prod)

    curr_command = list(command)
    curr_command[-3] = prod[0]
    curr_command[-1] = prod[1]

    print('Simulating...')

    print('*'*80)
    print('...compressible, forward.')
    flag = sp.check_call(curr_command) # compressible, forward

    print('*'*80)
    print('...compressible, inverse.')
    flag = sp.check_call(curr_command + ['-inv']) # compressible, inverse

    curr_command += ['-inc'] # add incompressible flag to command

    print('*'*80)
    print('...incompressible, forward.')
    flag = sp.check_call(curr_command) # incompressible, forward

    print('*'*80)
    print('...incompressible, inverse.')
    flag = sp.check_call(curr_command + ['-inv']) # incompressible, inverse
