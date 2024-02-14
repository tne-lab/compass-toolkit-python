import numpy as np
from scipy.optimize import fmin

def fminsearchbnd(fun, x0, LB=None, UB=None, options=None, *args):
    # Size checks
    xsize = np.shape(x0)
    x0 = np.ravel(x0)
    n = len(x0)

    if LB is None or len(LB) == 0:
        LB = -np.inf * np.ones(n)
    else:
        LB = np.ravel(LB)

    if UB is None or len(UB) == 0:
        UB = np.inf * np.ones(n)
    else:
        UB = np.ravel(UB)

    if n != len(LB) or n != len(UB):
        raise ValueError('x0 is incompatible in size with either LB or UB.')

    # Set default options if necessary
    if options is None:
        options = {'disp': False}

    # Stuff into a dictionary to pass around
    params = {
        'args': args,
        'LB': LB,
        'UB': UB,
        'fun': fun,
        'n': n,
        'xsize': xsize,
        'OutputFcn': None,
        'BoundClass': np.zeros(n),
    }

    # 0 --> unconstrained variable
    # 1 --> lower bound only
    # 2 --> upper bound only
    # 3 --> dual finite bounds
    # 4 --> fixed variable
    for i in range(n):
        k = np.isfinite(LB[i]) + 2 * np.isfinite(UB[i])
        params['BoundClass'][i] = k
        if k == 3 and (LB[i] == UB[i]):
            params['BoundClass'][i] = 4

    # Transform starting values into their unconstrained surrogates
    x0u = np.zeros(n)
    k = 0
    for i in range(n):
        if params['BoundClass'][i] == 1:
            if x0[i] <= LB[i]:
                x0u[k] = 0
            else:
                x0u[k] = np.sqrt(x0[i] - LB[i])
            k += 1
        elif params['BoundClass'][i] == 2:
            if x0[i] >= UB[i]:
                x0u[k] = 0
            else:
                x0u[k] = np.sqrt(UB[i] - x0[i])
            k += 1
        elif params['BoundClass'][i] == 3:
            if x0[i] <= LB[i]:
                x0u[k] = -np.pi / 2
            elif x0[i] >= UB[i]:
                x0u[k] = np.pi / 2
            else:
                x0u[k] = 2 * (x0[i] - LB[i]) / (UB[i] - LB[i]) - 1
                x0u[k] = 2 * np.pi + np.arcsin(np.clip(x0u[k], -1, 1))
            k += 1
        elif params['BoundClass'][i] == 0:
            x0u[k] = x0[i]
            k += 1

    # If any of the unknowns were fixed, then shorten x0u now
    if k <= n:
        x0u[k:n] = []

    # Were all the variables fixed?
    if not x0u.any():
        # All variables were fixed. Quit immediately, setting the
        # appropriate parameters, then return.
        x = xtransform(x0u, params)
        x = np.reshape(x, xsize)
        fval = fun(x, *params['args'])
        exitflag = 0
        output = {
            'iterations': 0,
            'funcCount': 1,
            'algorithm': 'fminsearch',
            'message': 'All variables were held fixed by the applied bounds'
        }
        return x, fval, exitflag, output

    # Check for an outputfcn. If there is any, then substitute my
    # own wrapper function.
    if options and 'outputfcn' in options:
        params['OutputFcn'] = options['outputfcn']
        options['outputfcn'] = outfun_wrapper

    # Now we can call fmin, but with our own intra-objective function.
    results = fmin(intrafun, x0u, args=(params,), disp=options['disp'], full_output=True)
    xu = results[0]
    fval= results[1]
    exitflag= results[2]
    output= results[3]
    # Undo the variable transformations into the original space
    x = xtransform(xu, params)

    # Final reshape to make sure the result has the proper shape
    x = np.reshape(x, xsize)

    return x, fval, exitflag, output


# Nested function as the OutputFcn wrapper
def outfun_wrapper(x, *args):
    # We need to transform x first
    xtrans = xtransform(x, args[0])

    # Then call the user supplied OutputFcn
    return args[0]['OutputFcn'](xtrans, *args[1:])


# Subfunction to transform variables
def xtransform(x, params):
    xtrans = np.zeros(params['xsize'])
    k = 0
    for i in range(params['n']):
        if params['BoundClass'][i] == 1:
            xtrans[i] = params['LB'][i] + x[k] ** 2
            k += 1
        elif params['BoundClass'][i] == 2:
            xtrans[i] = params['UB'][i] - x[k] ** 2
            k += 1
        elif params['BoundClass'][i] == 3:
            xtrans[i] = (np.sin(x[k]) + 1) / 2
            xtrans[i] = xtrans[i] * (params['UB'][i] - params['LB'][i]) + params['LB'][i]
            xtrans[i] = np.clip(xtrans[i], params['LB'][i], params['UB'][i])
            k += 1
        elif params['BoundClass'][i] == 4:
            xtrans[i] = params['LB'][i]
        elif params['BoundClass'][i] == 0:
            xtrans[i] = x[k]
            k += 1

    return xtrans


# Subfunction to transform variables and call the original function
def intrafun(x, params):
    # Transform
    xtrans = xtransform(x, params)

    # Call fun
    fval = params['fun'](np.reshape(xtrans, params['xsize']), *params['args'])
    return fval


# # Example usage:
# def rosen(x):
#     return (1 - x[0]) ** 2 + 105 * (x[1] - x[0] ** 2) ** 2
#
#
# result = fminsearchbnd(rosen, [3, 3], [2, 2], [])
# print(result)
