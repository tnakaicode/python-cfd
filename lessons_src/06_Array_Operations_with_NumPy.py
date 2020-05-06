#!/usr/bin/env python
# coding: utf-8
Text provided under a Creative Commons Attribution license, CC-BY.  All code is made available under the FSF-approved BSD-3 license.  (c) Lorena A. Barba, Gilbert F. Forsyth 2017. Thanks to NSF for support via CAREER award #1149784.
# [@LorenaABarba](https://twitter.com/LorenaABarba)

# 12 steps to Navier–Stokes
# =====
# ***

# This lesson complements the first interactive module of the online [CFD Python](https://github.com/barbagroup/CFDPython) class, by Prof. Lorena A. Barba, called **12 Steps to Navier–Stokes.** It was written with BU graduate student Gilbert Forsyth.

# Array Operations with NumPy
# ----------------
# 
# For more computationally intensive programs, the use of built-in Numpy functions can provide an  increase in execution speed many-times over.  As a simple example, consider the following equation:
# 
# $$u^{n+1}_i = u^n_i-u^n_{i-1}$$
# 
# Now, given a vector $u^n = [0, 1, 2, 3, 4, 5]\ \ $   we can calculate the values of $u^{n+1}$ by iterating over the values of $u^n$ with a for loop.  

# In[1]:


import numpy


# In[2]:


u = numpy.array((0, 1, 2, 3, 4, 5))

for i in range(1, len(u)):
    print(u[i] - u[i-1])


# This is the expected result and the execution time was nearly instantaneous.  If we perform the same operation as an array operation, then rather than calculate $u^n_i-u^n_{i-1}\ $ 5 separate times, we can slice the $u$ array and calculate each operation with one command:

# In[3]:


u[1:] - u[0:-1]


# What this command says is subtract the 0th, 1st, 2nd, 3rd, 4th and 5th elements of $u$ from the 1st, 2nd, 3rd, 4th, 5th and 6th elements of $u$.  
# 
# ### Speed Increases
# 
# For a 6 element array, the benefits of array operations are pretty slim.  There will be no appreciable difference in execution time because there are so few operations taking place.  But if we revisit 2D linear convection, we can see some substantial speed increases.  
# 

# In[4]:


nx = 81
ny = 81
nt = 100
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx

x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)

u = numpy.ones((ny, nx)) ##create a 1xn vector of 1's
un = numpy.ones((ny, nx)) 

###Assign initial conditions

u[int(.5 / dy): int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2


# With our initial conditions all set up, let's first try running our original nested loop code, making use of the iPython "magic" function `%%timeit`, which will help us evaluate the performance of our code. 
# 
# **Note**: The `%%timeit` magic function will run the code several times and then give an average execution time as a result.  If you have any figures being plotted within a cell where you run `%%timeit`, it will plot those figures repeatedly which can be a bit messy. 
# 
# The execution times below will vary from machine to machine.  Don't expect your times to match these times, but you _should_ expect to see the same general trend in decreasing execution time as we switch to array operations.

# In[5]:


get_ipython().run_cell_magic('timeit', '', 'u = numpy.ones((ny, nx))\nu[int(.5 / dy): int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2\n\nfor n in range(nt + 1): ##loop across number of time steps\n    un = u.copy()\n    row, col = u.shape\n    for j in range(1, row):\n        for i in range(1, col):\n            u[j, i] = (un[j, i] - (c * dt / dx * \n                                  (un[j, i] - un[j, i - 1])) - \n                                  (c * dt / dy * \n                                   (un[j, i] - un[j - 1, i])))\n            u[0, :] = 1\n            u[-1, :] = 1\n            u[:, 0] = 1\n            u[:, -1] = 1')


# With the "raw" Python code above, the mean execution time achieved was 3.07 seconds (on a MacBook Pro Mid 2012).  Keep in mind that with these three nested loops, that the statements inside the **j** loop are being evaluated more than 650,000 times.   Let's compare that with the performance of the same code implemented with array operations:

# In[6]:


get_ipython().run_cell_magic('timeit', '', 'u = numpy.ones((ny, nx))\nu[int(.5 / dy): int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2\n\nfor n in range(nt + 1): ##loop across number of time steps\n    un = u.copy()\n    u[1:, 1:] = (un[1:, 1:] - (c * dt / dx * (un[1:, 1:] - un[1:, 0:-1])) -\n                              (c * dt / dy * (un[1:, 1:] - un[0:-1, 1:])))\n    u[0, :] = 1\n    u[-1, :] = 1\n    u[:, 0] = 1\n    u[:, -1] = 1')


# As you can see, the speed increase is substantial.  The same calculation goes from 3.07 seconds to 7.38 milliseconds.  3 seconds isn't a huge amount of time to wait, but these speed gains will increase exponentially with the size and complexity of the problem being evaluated.  

# In[7]:


from IPython.core.display import HTML
def css_styling():
    styles = open("../styles/custom.css", "r").read()
    return HTML(styles)
css_styling()

