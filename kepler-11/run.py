#!/usr/bin/env python
import allesfitter

fig = allesfitter.show_initial_guess('.')
#allesfitter.prepare_ttv_fit('.', style='tessplot')

# nested sampling
#allesfitter.ns_fit('.')
allesfitter.ns_output('.')

# mcmc (if needed)
#allesfitter.mcmc_fit('.')
#allesfitter.mcmc_output('.')
