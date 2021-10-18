import jax.numpy as np
from jax.scipy.linalg import expm

from functools import partial
from jax import jit, grad, lax, vmap
from jax.nn import relu
from jax.ops import index, index_update

grad_relu = vmap(grad(relu))

@jit
def project_on_so_algebra(grad_eval, x):
    """Evaluates \sum_i (Tr(grad_f(x) * x * t_i) * t_i where {t_i} is a set
    of generators of the lie group, for the specific case of SO(n)"""
    return (grad_eval@x).T - grad_eval@x


@jit
def project_on_sl_algebra(grad_eval, x):
    """Evaluates \sum_i (Tr(grad_f(x) * x * t_i) * t_i where {t_i} is a set
    of generators of the lie group, for the specific case of SL(n)"""
    gx = grad_eval @ x
    diag_gx = np.diag(gx)
    out = (gx - np.diag(diag_gx)).T
    diag_out = np.zeros(out.shape[0])
    for i in range(out.shape[0] - 1):
        curr = np.zeros_like(diag_out)
        curr = index_update(
            curr, index[i:i+2], 
            np.array([diag_gx[i] - diag_gx[i+1], -diag_gx[i] + diag_gx[i+1]]))
        diag_out += curr
    return out + np.diag(diag_out)


def dissipative_lie_rattle(f, grad_f, x0, mu, stepsize, thr, maxiter=1000):
    """Algorithm 2 in Franca, Barp, Girolami, Jordan (2021)"""
    eps = 10 * thr
    curr_x = x0
    curr_y = np.zeros(x0.shape)
    grad_eval = grad_f(curr_x).T
    chi = np.cosh(-np.log(mu))
    for _ in range(maxiter):
        curr_y = mu * (curr_y - stepsize * project_on_sl_algebra(grad_eval, curr_x))
        curr_x = np.matmul(curr_x, expm(chi * curr_y))

        f_eval = f(curr_x)
        grad_eval = grad_f(curr_x).T
        # print("grad_norm: ", np.linalg.norm(grad_eval))
        if np.linalg.norm(grad_eval) < eps:
            print("breaking at {0}, f_eval: {1}".format(_, f_eval))
            return curr_x

        curr_y = mu * curr_y - stepsize * project_on_sl_algebra(grad_eval, curr_x)

    return curr_x


@partial(jit, static_argnums=(4,))
def lie_rattle_step(i, curr_x, curr_y, grad_eval, grad_fn, mu, stepsize, chi):
    curr_y = mu * (curr_y - stepsize * project_on_sl_algebra(grad_eval, curr_x))
    curr_x = np.matmul(curr_x, expm(chi * curr_y))
    grad_eval = grad_fn(curr_x).T
    curr_y = mu * curr_y - stepsize * project_on_sl_algebra(grad_eval, curr_x)
    return i+1, curr_x, curr_y, grad_eval


@partial(jit, static_argnums=(0, 1))
def dissipative_lie_rattle_fast(
        f, grad_f, x0, mu, stepsize, thr, maxiter=1000):

    chi = np.cosh(-np.log(mu))

    def loop_carry_fn(args):
        return lie_rattle_step(
            args[0], args[1], args[2], args[3], grad_f, mu, stepsize, chi)

    def loop_cond(args):
        c1 = args[0] < maxiter
        c2 = np.linalg.norm(args[3]) > thr
        c_out = (c1 + c2) == 2
        return c2
 
    curr_x = x0
    curr_y = np.zeros(x0.shape)
    grad_eval = grad_f(curr_x).T
    
    n_iter, x, y, g_eval = lax.while_loop(
        loop_cond,
        loop_carry_fn,
        (0, curr_x, curr_y, grad_eval))
    return x, n_iter


@partial(jit, static_argnums=(0, 1, 2))
def dissipative_lie_rattle_penalized(
        grad_loss, constraints, grad_constraints, x0, mu, stepsize, thr, 
        rho, lambdas, maxiter=1000):
    """ 
    Specific version of the dissipative RATTLE on SL(n) for a penalized loss function
    """
    chi = np.cosh(-np.log(mu))
    grad_f = lambda x: grad_loss(x) + grad_penalty(
        constraints, grad_constraints, x, rho, lambdas)

    def loop_carry_fn(args):
        return lie_rattle_step(
            args[0], args[1], args[2], args[3], grad_f, mu, stepsize, chi)

    def loop_cond(args):
        c1 = args[0] < 100
        c2 = np.linalg.norm(args[3]) > thr
        c_out = (c1 + c2) == 2
        return c1
 
    curr_x = x0
    curr_y = np.zeros(x0.shape)
    grad_eval = grad_f(curr_x).T
    
    n_iter, x, y, g_eval = lax.while_loop(
        loop_cond,
        loop_carry_fn,
        (0, curr_x, curr_y, grad_eval))
    return x, n_iter


def max0(x):
    return x * (x > 0)


@partial(jit, static_argnums=(0, 1))
def grad_penalty(constraints, grad_constraints, x, rho, lambdas):
    """
    Returns the gradient of
        rho/2 * SUM(relu( (lambda_j / rho * c_j(x))^2 ))
    """
    eval_c = constraints(x)
    g1 =  grad_relu((lambdas / rho * eval_c)**2)
    g2 =  2 * lambdas / rho * eval_c
    g3 = grad_constraints(x)
    out = 0.5 * rho  * np.sum(
        g3 * 
        g1[:, np.newaxis, np.newaxis] * 
        g2[:, np.newaxis, np.newaxis], axis=0)
    return out


@partial(jit, static_argnums=(0,))
def penalty(constraints, x, rho, lambdas):
    return 0.5 * rho * np.sum(relu(lambdas / rho + constraints(x))**2)


def ralm(loss_fn, grad_loss, constraints, grad_constraints, x0, mu, stepsize,
         init_thr, target_thr, init_lambdas, min_lambda, max_lambda, init_rho, 
         dmin, maxiter=1000):
    """
    loss_fn: the unconstrained objective function
    grad_loss: the gradient of the objective function
    constraints: a function g(x). The constraint is g(x) <= 0
    x0: the initial point
    init_lambdas: the initial values of the Lagrange multipliers
    init_rho: the initial value of the Lagrange multiplier
    """

    lambdas = init_lambdas
    rho = init_rho
    
    curr_x = x0
    prev_x = x0
    eps = init_thr
    print("Init Loss: ", loss_fn(curr_x))
    out = [curr_x]

    for i in range(maxiter):
        curr_x, _ = dissipative_lie_rattle_penalized(
            grad_loss, constraints, grad_constraints, curr_x, mu, 
            stepsize, eps, rho, lambdas, maxiter=100)
        out.append(curr_x)

        if np.isnan(curr_x).any():
            break

        print("Loss: {0}, step: {1}, eps: {2}".format(
            loss_fn(curr_x), np.linalg.norm(curr_x - prev_x), eps))

        if np.linalg.norm(curr_x - prev_x) < dmin and eps < target_thr:
            break

        lambdas = np.clip(
            lambdas + rho * (constraints(curr_x)),
            min_lambda, max_lambda)
        print("max(lambdas) : ", np.max(lambdas))
        rho = rho * 0.9
        eps = np.max(np.array([target_thr, eps * 0.9]))
        prev_x = curr_x

    return out



def dissipative_euclid_rattle(f, grad_f, x0, mu, stepsize, thr, maxiter=1000):
    """Algorithm 2 in Franca, Barp, Girolami, Jordan (2021)"""
    eps = 10 * thr
    curr_x = x0
    curr_y = np.zeros(x0.shape)
    grad_eval = grad_f(curr_x)
    chi = np.cosh(-np.log(mu))
    for _ in range(maxiter):
        curr_y = mu * curr_y - stepsize * grad_eval
        curr_x = chi * curr_y
        grad_eval = grad_f(curr_x)
        print("grad_norm: ", np.linalg.norm(grad_eval))
        if np.linalg.norm(grad_eval) < eps:
            return curr_x

        curr_y = mu * curr_y - stepsize * grad_eval

    return curr_x


def gd(f, grad_f, x0, stepsize, thr, maxiter=1000):
    eps = 10 * thr
    curr_x = x0
    curr_y = np.zeros(x0.shape)
    grad_eval = grad_f(curr_x)
    for _ in range(maxiter):
        curr_x = curr_x - stepsize * grad_eval
        grad_eval = grad_f(curr_x)
        print("grad_norm: ", np.linalg.norm(grad_eval))
        if np.linalg.norm(grad_eval) < eps:
            return curr_x

    return curr_x




    

    
    











