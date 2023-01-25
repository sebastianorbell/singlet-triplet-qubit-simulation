
import jax.numpy as np

import qutip as qt
import jax.random as random

key = random.PRNGKey(758493)  # Random seed is explicit in JAX
random.uniform(key, shape=(1000,))
# QUBIT PARAMETERS

def gen_drift_hamiltonian(
    omega,
    alpha,
        N
):
    a = qt.destroy(N)

    H = omega * a.dag() * a + alpha / 2 * a.dag() * a.dag() * a * a - omega / 2 * qt.qeye(N)

    return np.array(H)

def gen_control_hamiltonion(
        I,
        Q,
        non_linearity,
        w_d,
        phase,
        t,
        N
):
    a = np.array(qt.tensor(qt.destroy(N)))
    a_d = np.array(qt.tensor(qt.destroy(N)).dag())
    Hs = non_linearity((a+a_d)*(I(t)*np.sin(w_d*t+phase)+Q(t)*np.cos(w_d*t+phase)))
    return Hs

def generate_all_noise_parameters():
    g = 2 * np.pi * (.0009 * random.uniform(key, [1]) + .0001)
    e = 2 * np.pi * (random.uniform(key, [1]) * .015 + 4.5 - .015)
    d = 2 * np.pi * .367 * np.exp(-random.uniform(key, [1]) * np.log(4e1))
    TLS_decT = [250 + 2e3 * random.uniform(key, [1]), 500 + 4.5e3 * random.uniform(key, [1])]

    overrotation = 1e-2 * random.normal(key, [1])
    phase = .5 / 180 * np.pi * random.normal(key, [1])
    domega = 2 * np.pi * 2e-6 * random.normal(key, [1])

    # slow_T2 =  np.abs(5e2 + 99.5e2*random.uniform(key, [1]) )
    slow_T2 = np.abs(8e3 * (1 + .33 * random.normal(key, [1])))

    Mark_decT = [np.abs(35e3 * (1 + .33 * random.normal(key, [1]))), np.abs(35e3 * (1 + .33 * random.normal(key, [1])))]

    alpha = -.2 * 2 * np.pi * (1 + .15 * random.normal(key, [1]))
    return e, d, g, TLS_decT, overrotation, phase, domega, slow_T2, Mark_decT, alpha

def gen_c_ops(N, decT, TLS_decTs, Markov=False, TLS=0):
    nd = np.array(qt.destroy(N))
    nd_dag = np.array(qt.destroy(N).dag())
    eye = np.array(qt.qeye(N))
    if Markov == False and TLS == 0:
        print('No Markovian decoherence has been initialized!')
        return []
    elif TLS == 0:
        c_ops = []
        a = np.copy(nd)
        if decT[0]:
            c_ops.append(1 / np.sqrt(decT[0]) * a)
        if decT[1]:
            c_ops.append(1 / np.sqrt(decT[1]) * nd_dag * a)
    else:
        print('Initializing LOCAL Lindblad jump operators!')
        c_ops = []

        if Markov == True:
            a = nd + np.array([eye for x in range(TLS)])

            if decT[0]:
                c_ops.append(1 / np.sqrt(decT[0]) * a)
            if decT[1]:
                c_ops.append(1 / np.sqrt(decT[1]) * nd_dag * a)

        for i in range(TLS):
            op = nd + np.array([eye for x in range(TLS)])
            # op[1 + i] = nd
            b = np.array(qt.tensor(op))
            b_dag = np.array(qt.tensor(op).dag())
            if TLS_decTs[i][0]:
                c_ops.append(1 / np.sqrt(TLS_decTs[i][0]) * b)
            if TLS_decTs[i][1]:
                c_ops.append(1 / np.sqrt(TLS_decTs[i][1]) * b_dag * b)

    return c_ops

def add_TLSs(gs, Es, dels, omega, alpha,  N):
    TLS = len(gs)


    a = qt.tensor([qt.destroy(N)] + [qt.qeye(2) for x in range(TLS)])
    H = omega * a.dag() * a + ( alpha / 2 )* a.dag() * a.dag() * a * a

    for i in range(TLS):
        op = [qt.qeye(N)] + [qt.qeye(2) for x in range(TLS)]
        op[1 + i] = qt.destroy(2)
        b = qt.tensor(op)
        H += Es[i].__float__() / 2 * (b.dag() * b - b * b.dag()) + dels[i].__float__() * (b + b.dag()) + gs[i].__float__() * (
                    a.dag() * b + b.dag() * a)
    # c_ops = gen_c_ops()
    return H, b

def gaussian(x, amp, *args, x_max = 20, mu=10., sig=3.):
    return amp*(np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))-np.exp(-np.power(x_max - mu, 2.) / (2 * np.power(sig, 2.))))

def d_gaussian(x, amp, *args, mu=10., sig=3.):
    return -amp*(mu-x)/np.power(sig, 2.) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def return_initial_state(index, N):
    initial_state = np.zeros(N).astype(complex)
    initial_state = initial_state.at[index].set(1.0+0.0j)
    return initial_state

if __name__=='__main__':
    e, d, g, TLS_decT, overrotation, phase, domega, slow_T2, Mark_decT, alpha = generate_all_noise_parameters()
    # c_ops = gen_c_ops(3, Mark_decT, TLS_decT, Markov=True, TLS=3)

    h,b = add_TLSs(g, e, TLS_decT, domega.__float__(), alpha.__float__(), 3)