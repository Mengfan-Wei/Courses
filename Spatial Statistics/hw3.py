import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False


def plotZ():
    data = np.loadtxt('./XZ.txt', skiprows=2)
    x = data[:, 0]
    z = data[:, 1]

    plt.figure(figsize=(10, 10), dpi=100)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    ax1.set_title('a.XYplot')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.text(40, 35, "魏梦凡 12132708", fontdict={'size': 14, 'color': 'g'})
    ax1.axhline(y=34, xmin=0.21, xmax=0.61)
    ax1.plot(x, z, "b*")
    ax1.plot(x, z)

    ax2.set_title('b.Histogram')
    ax2.set_xlabel('Z')
    ax2.set_ylabel('Frequency')
    ax2.text(20, 23, "魏梦凡 12132708", fontdict={'size': 14, 'color': 'g'})
    ax2.axhline(y=22, xmin=0.47, xmax=0.88)
    ax2.hist(z)

    ax3.set_title('c.Normal probability plot')
    ax3.set_xlabel('Z')
    ax3.text(-2, 30, "魏梦凡 12132708", fontdict={'size': 14, 'color': 'g'})
    ax3.axhline(y=29, xmin=0.1, xmax=0.5)
    sm.qqplot(z, line='r', ax=ax3)

    ax4.set_title('d.Box & Whisker plot')
    ax4.set_ylabel('z')
    ax4.boxplot(z)
    ax4.text(1.02, 30, "魏梦凡 12132708", fontdict={'size': 14, 'color': 'g'})
    ax4.axhline(y=29, xmin=0.52, xmax=0.92)
    plt.savefig(fname="./problem1.svg", dpi=100)


def plotVU():
    data = np.loadtxt('./XYVU.txt')
    V = data[:, 2]
    U = data[:, 3]
    plt.figure(figsize=(10, 5), dpi=100)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)

    ax1.set_title('Scatter plot')
    ax1.set_xlabel('V')
    ax1.set_ylabel('U')
    ax1.scatter(V, U, marker='o', color='', edgecolor='b')
    ax1.plot(V, np.poly1d(np.polyfit(V, U, 1))(V), 'r')

    ax2.set_title('q-q plot')
    sm.qqplot_2samples(U, V, ax=ax2, line='r')
    plt.savefig(fname="./problem2.svg", dpi=100)


if __name__ == '__main__':
    plotZ()
    plotVU()
