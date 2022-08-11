# @Time : 2022/8/11 11:19 
# @Author : 张恩硕
# @File : draw.py 
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

# First create some toy data:
'''生成从0到2*np.pi之间的400个数'''
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

# Create just a figure and only one subplot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')
# class matplotlib.axes.Axes
# https://matplotlib.org/stable/api/axes_api.html#plotting
# axes可以调用的方法集合


# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax1.set_xlabel('Horizontal Coordinate')
ax2.scatter(x, y)
ax2.set_xlabel('Horizontal Coordinate')

# Create four polar axes and access them through the returned array
fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
# 只设置了主对角线的两个子图
axs[0, 0].plot(x, y)
axs[1, 1].scatter(x, y)

# Share a X axis with each column of subplots
plt.subplots(2, 2, sharex='col')

# Share a Y axis with each row of subplots
plt.subplots(2, 2, sharey='row')

# Share both X and Y axes with all subplots
plt.subplots(2, 2, sharex='all', sharey='all')

# Note that this is the same as
plt.subplots(2, 2, sharex=True, sharey=True)

# Create figure number 10 with a single subplot
# and clears it if it already exists.
fig, ax = plt.subplots(num=10, clear=False)
