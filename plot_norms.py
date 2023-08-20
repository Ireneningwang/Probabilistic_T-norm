import torch
import os
import matplotlib.pyplot as plt

from importlib import reload
import tyche.tnorms.tnorms_binary
reload(tyche.tnorms.tnorms_binary)
from tyche.tnorms.tnorms_binary import t_norm_and_interpolation_ratio
from tyche.tnorms.tnorms_binary import t_norm_or_interpolation_ratio
from tyche.tnorms.tnorms_binary import t_norm_and_conditional_ratio
from tyche.tnorms.tnorms_binary import t_norm_or_conditional_ratio
from tyche.tnorms.tnorms_binary import t_norm_and_pearsons_r
from tyche.tnorms.tnorms_binary import t_norm_or_pearsons_r


# define single plot function which can plot either 'or' or 'and' function for three paramarised t-norms
def plot_tnorm(pxgrid, pygrid, tnormgrid, title=None, zlabel='T(x,y)', levels2d=20, levels3d=50):
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.contour3D(pxgrid, pygrid, tnormgrid, levels3d)
    if title:
        ax.set_title(title)
    ax.set_xlabel('p(x)')
    ax.set_ylabel('p(y)')
    ax.set_zlabel(zlabel)
    ax = fig.add_subplot(1, 2, 2)    
    ax.set_title(title)
    contour_set = ax.contourf(pxgrid, pygrid, tnormgrid, levels=levels2d)
    ax.set_xlabel('p(x)')
    ax.set_ylabel('p(y)')
    return fig


def generate_images(pxs, pys, gamma, dir):
    pxgrid, pygrid = torch.meshgrid(pxs, pys, indexing='ij')
    rho = 2*gamma - 1 # rough equivalence between 

    # ---- Interpolation: AND ----
    title1 = fr"$T(p_x, p_y; \gamma) \approx p(X=1,Y=1)$ for $\gamma =$ {gamma:.2f}" 
    pandgrid_inter_and = t_norm_and_interpolation_ratio(pxgrid, pygrid, gamma)
    plot_tnorm(pxgrid, pygrid, pandgrid_inter_and, zlabel='T(p_x,p_y)', title= title1+' (interpolation)', levels3d=50, levels2d=20)
    filename = dir +  '\Interpolation_ratio_and' + '.png'
    plt.savefig(str(filename))
    # ---- Interpolation: OR ----
    title0 = fr"$T(p_x, p_y; \gamma) \approx p(X=0,Y=0)$ for $\gamma =$ {gamma:.2f}"
    pandgrid_inter_or = t_norm_or_interpolation_ratio(pxgrid, pygrid, gamma)
    plot_tnorm(pxgrid, pygrid, pandgrid_inter_or, zlabel='T(p_x,p_y)', title= title0+' (interpolation)', levels3d=50, levels2d=20)
    filename = dir +  '\Interpolation_ratio_or' + '.png'
    plt.savefig(str(filename))


    # ---- Pearsons r: AND ----
    title1 = fr"$T(x,y) \approx p(X=1,Y=1)$ for $\rho$ {rho:.2f}"
    pandgrid_pear_and = t_norm_and_pearsons_r(pxgrid, pygrid, rho)
    plot_tnorm(pxgrid, pygrid, pandgrid_pear_and, zlabel='T(p_x,p_y)', title=title1+' (pearsons r)', levels3d=50, levels2d=20)
    filename = dir + '\Pearsons_r_and' + '.png'
    plt.savefig(str(filename))
    # ---- Pearsons r: OR ----
    title0 = fr"$T(x,y) \approx p(X=0,Y=0)$ for $\rho =$ {rho:.2f}"
    pandgrid_pear_or = t_norm_or_pearsons_r(pxgrid, pygrid, rho)
    plot_tnorm(pxgrid, pygrid, pandgrid_pear_or, zlabel='T(p_x,p_y)', title= title0+' (pearsons r)', levels3d=50, levels2d=20)
    filename = dir +  '\Pearsons_r_or' + '.png'
    plt.savefig(str(filename))


    # ---- Conditional ratio: AND ----
    title1 = fr"$T(x,y) \approx p(x=1,y=1)$ for gamma {gamma:.2f}"
    pandgrid_cond_and = t_norm_and_conditional_ratio(pxgrid, pygrid, gamma)
    plot_tnorm(pxgrid, pygrid, pandgrid_cond_and, zlabel='T(p_x,p_y)',title=title1+' (conditional ratio)', levels3d=50, levels2d=20)
    filename = dir +  '\Conditional_ratio_and' + '.png'
    plt.savefig(str(filename))
    # ---- Conditional ratio: OR ----
    title0 = fr"$T(x,y) \approx p(x=0,y=0)$ for gamma {gamma:.2f}"
    pandgrid_cond_or = t_norm_or_conditional_ratio(pxgrid, pygrid, gamma)
    plot_tnorm(pxgrid, pygrid, pandgrid_cond_or, zlabel='T(p_x,p_y)',title=title0+' (conditional ratio)', levels3d=50, levels2d=20)
    filename = dir + '\Conditional_ratio_or' + '.png'
    plt.savefig(str(filename))


if __name__ == '__main__':
    # testing the single plot functions
    pxs = torch.linspace(0,1, 51)
    pys = torch.linspace(0,1, 51)
    gamma = 0.35
    dir = os.getcwd()+'\Images' +'\original_' + str(gamma)
    if not os.path.exists(dir):
        os.mkdir(dir)
    generate_images(pxs, pys, gamma, dir)