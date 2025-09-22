import numpy as np
import torch
import scipy.sparse as sp

####################################################
# This tool is to generate meta-path based adjacency
# matrix given original links.
####################################################


def DBLP():
    pa = np.genfromtxt("./data/DBLP/pa.txt")
    pc = np.genfromtxt("./data/DBLP/pc.txt")
    pt = np.genfromtxt("./data/DBLP/pt.txt")

    A = 4057
    P = 14328
    C = 20
    T = 7723

    pa = sp.coo_matrix((np.ones(pa.shape[0]), (pa[:, 0], pa[:, 1])), shape=(P, A)).toarray()
    pc = sp.coo_matrix((np.ones(pc.shape[0]), (pc[:, 0], pc[:, 1])), shape=(P, C)).toarray()
    pt = sp.coo_matrix((np.ones(pt.shape[0]), (pt[:, 0], pt[:, 1])), shape=(P, T)).toarray()

    apa = np.matmul(pa.T, pa)

    apa = sp.coo_matrix(apa)
    sp.save_npz("./data/DBLP/apa.npz", apa)

    apc = np.matmul(pa.T, pc)
    apcpa = np.matmul(apc, apc.T)
    apcpa = sp.coo_matrix(apcpa)
    sp.save_npz("./data/DBLP/apcpa.npz", apcpa)

    apt = np.matmul(pa.T, pt)
    aptpa = np.matmul(apt, apt.T)
    aptpa = sp.coo_matrix(aptpa)

    sp.save_npz("./data/DBLP/aptpa.npz", aptpa)

    # hop neighbors
    ap = pa.T
    apc = np.matmul(ap, pc)
    apt = np.matmul(ap, pt)

    ap = sp.coo_matrix(ap)
    apc = sp.coo_matrix(apc)
    apt = sp.coo_matrix(apt)

    sp.save_npz("./data/DBLP/ap.npz", ap)
    sp.save_npz("./data/DBLP/apc.npz", apc)
    sp.save_npz("./data/DBLP/apt.npz", apt)


def ACM():
    pa = np.genfromtxt("./data/ACM/pa.txt")
    ps = np.genfromtxt("./data/ACM/ps.txt")

    A = 7167
    P = 4019
    S = 60

    pa = sp.coo_matrix((np.ones(pa.shape[0]), (pa[:, 0], pa[:, 1])), shape=(P, A)).toarray()
    ps = sp.coo_matrix((np.ones(ps.shape[0]), (ps[:, 0], ps[:, 1])), shape=(P, S)).toarray()

    # meta-path neighbors
    pap = np.matmul(pa, pa.T)
    pap = sp.coo_matrix(pap)
    sp.save_npz("./data/ACM/pap.npz", pap)

    psp = np.matmul(ps, ps.T)
    psp = sp.coo_matrix(psp)
    sp.save_npz("./data/ACM/psp.npz", psp)

    # hop neighbors
    sp.save_npz("./data/ACM/pa.npz", pa)
    sp.save_npz("./data/ACM/ps.npz", ps)


def AMiner():
    pa = np.genfromtxt("./data/AMiner/pa.txt")
    pr = np.genfromtxt("./data/AMiner/pr.txt")

    P = 6564
    A = 13329
    R = 35890

    pa = sp.coo_matrix((np.ones(pa.shape[0]), (pa[:, 0], pa[:, 1])), shape=(P, A))
    pr = sp.coo_matrix((np.ones(pr.shape[0]), (pr[:, 0], pr[:, 1])), shape=(P, R))

    # meta-path neighbors
    # apa = np.matmul(pa.T, pa)
    # apa = sp.coo_matrix(apa)
    # sp.save_npz("./data/ACM/apa.npz", apa)

    # aps = np.matmul(pa.T, ps)
    # apspa = np.matmul(aps, aps.T)
    # apspa = sp.coo_matrix(apspa)
    # sp.save_npz("./data/ACM/apspa.npz", apspa)

    # hop neighbors
    sp.save_npz("./data/AMiner/pa.npz", pa)
    sp.save_npz("./data/AMiner/pr.npz", pr)


def IMDB():
    ma = torch.load('./data/IMDB/ma.pth').numpy()
    md = torch.load('./data/IMDB/md.pth').numpy()

    M = 4278
    A = 5257
    D = 2081

    ma = sp.coo_matrix((np.ones(ma.shape[1]), (ma[0, :], ma[1, :])), shape=(M, A))
    md = sp.coo_matrix((np.ones(md.shape[1]), (md[0, :], md[1, :])), shape=(M, D))

    # meta-path neighbors    (.toarray())
    # mam = np.matmul(ma, ma.T)
    # mam = sp.coo_matrix(mam)
    # sp.save_npz("./data/IMDB/mam.npz", mam)

    # mdm = np.matmul(md, md.T)
    # mdm = sp.coo_matrix(mdm)
    # sp.save_npz("./data/IMDB/mdm.npz", mdm)

    # hop neighbors
    sp.save_npz("./data/IMDB/ma.npz", ma)
    sp.save_npz("./data/IMDB/md.npz", md)

def FreeBase():
    md = np.genfromtxt("./data/FreeBase/md.txt",dtype=np.int32)
    ma = np.genfromtxt("./data/FreeBase/ma.txt",dtype=np.int32)
    mw = np.genfromtxt("./data/FreeBase/mw.txt",dtype=np.int32)

    M = 3492
    D = 2502
    A = 33401
    W = 4459

    md = sp.coo_matrix((np.ones(md.shape[0]), (md[:, 0], md[:, 1])), shape=(M, D))
    ma = sp.coo_matrix((np.ones(ma.shape[0]), (ma[:, 0], ma[:, 1])), shape=(M, A))
    mw = sp.coo_matrix((np.ones(mw.shape[0]), (mw[:, 0], mw[:, 1])), shape=(M, W))

    # meta-path neighbors    (.toarray())
    # mdm = np.matmul(md, md.T)
    # mdm = sp.coo_matrix(mdm)
    # sp.save_npz("./data/FreeBase/mdm.npz", mdm)

    # mam = np.matmul(ma, ma.T)
    # mam = sp.coo_matrix(mam)
    # sp.save_npz("./data/FreeBase/mam.npz", mam)

    # mwm = np.matmul(mw, mw.T)
    # mwm = sp.coo_matrix(mwm)
    # sp.save_npz("./data/FreeBase/mwm.npz", mwm)

    # hop neighbors
    # sp.save_npz("./data/FreeBase/md.npz", md)
    # sp.save_npz("./data/FreeBase/ma.npz", ma)
    # sp.save_npz("./data/FreeBase/mw.npz", mw)

def IMDB_():
    ma = np.genfromtxt("./data/IMDB_/ma.txt")
    md = np.genfromtxt("./data/IMDB_/md.txt")
    mk = np.genfromtxt("./data/IMDB_/mk.txt")

    M = 4275
    A = 5432
    D = 2083
    K = 7313

    ma = sp.coo_matrix((np.ones(ma.shape[0]), (ma[:, 0], ma[:, 1])), shape=(M, A)).toarray()
    md = sp.coo_matrix((np.ones(md.shape[0]), (md[:, 0], md[:, 1])), shape=(M, D)).toarray()
    mk = sp.coo_matrix((np.ones(mk.shape[0]), (mk[:, 0], mk[:, 1])), shape=(M, K)).toarray()

    # meta-path neighbors    (.toarray())
    mam = np.matmul(ma, ma.T)
    mam = sp.coo_matrix(mam)
    sp.save_npz("./data/IMDB_/mam.npz", mam)

    mdm = np.matmul(md, md.T)
    mdm = sp.coo_matrix(mdm)
    sp.save_npz("./data/IMDB_/mdm.npz", mdm)

    mkm = np.matmul(mk, mk.T)
    mkm = sp.coo_matrix(mkm)
    sp.save_npz("./data/IMDB_/mkm.npz", mkm)
    # hop neighbors
    # sp.save_npz("./data/IMDB_/ma.npz", ma)
    # sp.save_npz("./data/IMDB_/md.npz", md)
    # sp.save_npz("./data/IMDB_/mk.npz", mk)

# DBLP()
# ACM()
# AMiner()
IMDB()
# FreeBase()
# IMDB_()