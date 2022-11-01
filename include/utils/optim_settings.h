//
// Created by lei on 22-11-1.
//

#ifndef NUMERICOPTIM_OPTIM_SETTINGS_H
#define NUMERICOPTIM_OPTIM_SETTINGS_H

#ifdef USE_FLOAT
using VAL_T = float;
#else
using VAL_T = double;
#endif

struct settings {
    size_t max_iters = 2000; // maximum iterations
    VAL_T grad_err = 1e-8; // limit of grad's err
};

#endif //NUMERICOPTIM_OPTIM_SETTINGS_H
