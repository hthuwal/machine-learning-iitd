
def analytical_solution(x, y):
    theta = np.matrix(x.T @ x)
    theta = theta.I @ x.T @ y
    theta = np.squeeze(np.asarray(theta))
    return theta