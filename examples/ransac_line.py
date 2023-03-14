import numpy as np

def ransac_line(x, y, npoints=2, maxiter=20, t=1e-6, d=4):

    iterations = 0
    n = len(x)
    best_fit = np.inf
    best_model = None

    while True:
        inliers = list(np.random.randint(0, len(x), npoints))

        m, c, r2, *_ = sp.stats.linregress(x[inliers], y[inliers])
        new_inliers = []
        for k in range(len(x)):
            if k in inliers:
                continue
            if np.linalg.norm(m * x[k] + c - y[k]) < t:
                new_inliers.append(k)

        inliers.extend(new_inliers)
        if len(inliers) >= d:
            # decent model
            m, c, r2, *_ = sp.stats.linregress(x[inliers], y[inliers])
            err = np.linalg.norm(m * x[inliers] + c - y[inliers])
            if err < best_fit:
                best_fit = err
                best_model = (m, c)
                best_inliers = inliers

        iterations += 1
        if iterations > maxiter:
            break
    
    return best_model, best_inliers
