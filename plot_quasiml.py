import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import os
from importlib import reload

import tyche.tnorms.tnorms_binary
reload(tyche.tnorms.tnorms_binary)
from tyche.tnorms.tnorms_binary import t_norm_and_interpolation_ratio
from tyche.tnorms.tnorms_binary import t_norm_or_interpolation_ratio
from tyche.tnorms.tnorms_binary import t_norm_and_conditional_ratio
from tyche.tnorms.tnorms_binary import t_norm_or_conditional_ratio
from tyche.tnorms.tnorms_binary import t_norm_and_pearsons_r
from tyche.tnorms.tnorms_binary import t_norm_or_pearsons_r
from plot_norms import plot_tnorm


def quasiml_binary_piecewise_log(xs, k=250, mu=0.5):
    c = np.log(mu*k+1)
    a = 1/(c + np.log(k*(1-mu)+1))
    xlows = xs[xs <= mu]
    xhighs = xs[xs > mu]
    ylows = a*(c -np.log(k*(mu-xlows)+1))
    yhighs = a*(c + np.log(k*(xhighs-mu)+1))
    ys = np.empty(xs.shape)
    ys[xs <= mu] = ylows
    ys[xs > mu] = yhighs
    return torch.tensor(ys)


def quasiml_beta_cdf(xs, k=25, mu=0.5):
    b = k*(1-mu)
    a = k*mu
    ys = scipy.stats.beta.cdf(xs, a, b)
    return torch.tensor(ys)


def quasi_inputs(pxs, pys, k, mu, dependency, qml_method, dir):
    # quasi-ml applied to inputs
    kx =  k
    mux = mu
    ky =  k
    muy = mu

    title = fr"$T(x,y) \approx p(x=1,y=1)$ for dependency {dependency:.2f}"
    pxgrid, pygrid = torch.meshgrid(pxs, pys, indexing='ij')

    if qml_method  == 'piecewise_log':
        ptildexs = quasiml_binary_piecewise_log(pxgrid, k=kx, mu=mux)
        ptildeys = quasiml_binary_piecewise_log(pygrid, k=ky, mu=muy)
    else:
        ptildexs = quasiml_beta_cdf(pxgrid, k=kx, mu=mux)
        ptildeys = quasiml_beta_cdf(pygrid, k=ky, mu=muy)

    # ---- Interpolation ratio: AND ----
    pandgrid_inter_and = t_norm_and_interpolation_ratio(ptildexs, ptildeys, dependency)
    plot_tnorm(pxgrid, pygrid, pandgrid_inter_and, zlabel='p(x=1,y=1)', title=title+' (interpolation)', levels3d=50, levels2d=20)
    filename = dir+ '/Interpolation_ratio_and' + '.png'
    plt.savefig(str(filename))
    # ---- Interpolation ratio: OR ----
    pandgrid_inter_or = t_norm_or_interpolation_ratio(ptildexs, ptildeys, dependency)
    plot_tnorm(pxgrid, pygrid, pandgrid_inter_or, zlabel='p(x=0,y=0)', title=title+' (interpolation)', levels3d=50, levels2d=20)
    filename = dir + '/Interpolation_ratio_or' + '.png'
    plt.savefig(str(filename))

    # ---- Pearsons r: AND ----
    pandgrid_pear_and = t_norm_and_pearsons_r(ptildexs, ptildeys, dependency)
    plot_tnorm(pxgrid, pygrid, pandgrid_pear_and, zlabel='p(x=1,y=1)', title=title+' (pearsons r)', levels3d=50, levels2d=20)
    filename = dir + '/Pearsons_r_and' + '.png'
    plt.savefig(str(filename))
    # ---- Pearsons r: OR ----
    pandgrid_pear_or = t_norm_or_pearsons_r(ptildexs, ptildeys, dependency)
    plot_tnorm(pxgrid, pygrid, pandgrid_pear_or, zlabel='p(x=0,y=0)', title=title+' (pearsons r)', levels3d=50, levels2d=20)
    filename = dir + '/Pearsons_r_or' + '.png'
    plt.savefig(str(filename))

    # ---- Conditional ratio: AND ----
    pandgrid_cond_and = t_norm_and_conditional_ratio(ptildexs, ptildeys, dependency)
    plot_tnorm(pxgrid, pygrid, pandgrid_cond_and, zlabel='p(x=1,y=1)', title=title+' (conditional ratio)', levels3d=50, levels2d=20)
    filename = dir + '/Conditional_ratio_and' + '.png'
    plt.savefig(str(filename))
    # ---- Conditional ratio: OR ----
    pandgrid_cond_or = t_norm_or_conditional_ratio(ptildexs, ptildeys, dependency)
    plot_tnorm(pxgrid, pygrid, pandgrid_cond_or, zlabel='p(x=0,y=0)', title=title+' (conditional ratio)', levels3d=50, levels2d=20)
    filename = dir + '/Conditional_ratio_or' + '.png'
    plt.savefig(str(filename))



def quasi_outputs(pxs, pys, k, mu, dependency, qml_method, dir):
    # quasi-ml applied to outputs
    title1 = fr"$T(x,y) \approx p(x=1,y=1)$ for dependency {dependency:.2f}"
    title0 = fr"$T(x,y) \approx p(x=0,y=0)$ for dependency {dependency:.2f}"
    pxgrid, pygrid = torch.meshgrid(pxs, pys, indexing='ij')


    pandgrid_inter_and = t_norm_and_interpolation_ratio(pxgrid, pygrid, dependency)
    pandgrid_pear_and = t_norm_and_pearsons_r(pxgrid, pygrid, dependency)
    pandgrid_cond_and = t_norm_and_conditional_ratio(pxgrid, pygrid, dependency)

    pandgrid_inter_or = t_norm_or_interpolation_ratio(pxgrid, pygrid, dependency)
    pandgrid_pear_or = t_norm_or_pearsons_r(pxgrid, pygrid, dependency)
    pandgrid_cond_or = t_norm_or_conditional_ratio(pxgrid, pygrid, dependency)

    if qml_method  == 'piecewise_log':
        pandgrid_inter_tilde_and = quasiml_binary_piecewise_log(pandgrid_inter_and, k=k, mu=mu)
        pandgrid_pear_tilde_and = quasiml_binary_piecewise_log(pandgrid_pear_and, k=k, mu=mu)
        pandgrid_cond_tilde_and = quasiml_binary_piecewise_log(pandgrid_cond_and, k=k, mu=mu)

        pandgrid_inter_tilde_or = quasiml_binary_piecewise_log(pandgrid_inter_or, k=k, mu=mu)
        pandgrid_pear_tilde_or = quasiml_binary_piecewise_log(pandgrid_pear_or, k=k, mu=mu)
        pandgrid_cond_tilde_or = quasiml_binary_piecewise_log(pandgrid_cond_or, k=k, mu=mu)
    else:
        pandgrid_inter_tilde_and = quasiml_beta_cdf(pandgrid_inter_and, k=k, mu=mu)
        pandgrid_pear_tilde_and = quasiml_beta_cdf(pandgrid_pear_and, k=k, mu=mu)
        pandgrid_cond_tilde_and = quasiml_beta_cdf(pandgrid_cond_and, k=k, mu=mu)

        pandgrid_inter_tilde_or = quasiml_beta_cdf(pandgrid_inter_or, k=k, mu=mu)
        pandgrid_pear_tilde_or = quasiml_beta_cdf(pandgrid_pear_or, k=k, mu=mu)
        pandgrid_cond_tilde_or = quasiml_beta_cdf(pandgrid_cond_or, k=k, mu=mu)

    plot_tnorm(pxgrid, pygrid, pandgrid_inter_tilde_and, zlabel='p(x=1,y=1)',title=title1+' (interpolation)', levels3d=50, levels2d=20)
    filename = dir + '/Interpolation_ratio_and' + '.png'
    plt.savefig(str(filename))

    plot_tnorm(pxgrid, pygrid, pandgrid_pear_tilde_and, zlabel='p(x=1,y=1)',title=title1+' (pearsons r)', levels3d=50, levels2d=20)
    filename = dir + '/Pearsons_r_and' + '.png'
    plt.savefig(str(filename))

    plot_tnorm(pxgrid, pygrid, pandgrid_cond_tilde_and, zlabel='p(x=1,y=1)',title=title1+' (conditional ratio)', levels3d=50, levels2d=20)
    filename = dir + '/Conditional_ratio_and' + '.png'
    plt.savefig(str(filename))

    plot_tnorm(pxgrid, pygrid, pandgrid_inter_tilde_or, zlabel='p(x=0,y=0)',title=title0+' (interpolation)', levels3d=50, levels2d=20)
    filename = dir + '/Interpolation_ratio_or' + '.png'
    plt.savefig(str(filename))

    plot_tnorm(pxgrid, pygrid, pandgrid_pear_tilde_or, zlabel='p(x=0,y=0)',title=title0+' (pearsons r)', levels3d=50, levels2d=20)
    filename = dir + '/Pearsons_r_or' + '.png'
    plt.savefig(str(filename))
    plot_tnorm(pxgrid, pygrid, pandgrid_cond_tilde_or, zlabel='p(x=0,y=0)',title=title0+' (conditional ratio)', levels3d=50, levels2d=20)
    filename = dir + '/Conditional_ratio_or' + '.png'
    plt.savefig(str(filename))



if __name__ == '__main__':
    pxs = torch.linspace(0,1, 51)
    pys = torch.linspace(0,1, 51)
    k =  50
    mu = 0.7
    dependency = 0.35   # formally gamma
    for qml_method in ['piecewise_log','beta_cdf']: 
        dir = 'Images/' + qml_method + '_' + str(k) + '_' + str(mu) + '_' + str(dependency)
        if not os.path.exists(dir):
            os.mkdir(dir)
        quasi_inputs(pxs, pys, k, mu, dependency, qml_method, dir)
        quasi_outputs(pxs, pys, k, mu, dependency, qml_method, dir)