import numpy as np

def autoencoding(problem, traindataOutput, traindataInput, transferdata):
    curr_len = traindataOutput.shape[1]
    tmp_len = traindataInput.shape[1]
    if curr_len < tmp_len:
        traindataOutput = np.append(traindataOutput, np.zeros((traindataOutput.shape[0], tmp_len-curr_len)), axis=1)
    elif curr_len > tmp_len:
        traindataInput = np.append(traindataInput, np.zeros((traindataInput.shape[0], curr_len-tmp_len)), axis=1)

    xx = np.transpose(traindataOutput)
    noise = np.transpose(traindataInput)

    [d, n] = xx.shape
    xxb = np.append(xx, np.ones((1, n)), axis=0)
    noise_xb = np.append(noise, np.ones((1, n)), axis=0)
    Q = np.dot(noise_xb, np.transpose(noise_xb))
    P = np.dot(xxb, np.transpose(noise_xb))

    l = 1E-5
    reg = l*np.eye(d+1)
    reg[-1, -1] = 0
    QQ = Q+reg

    W = np.dot(P, np.linalg.inv(QQ))
    W = np.delete(W, -1, axis=1)
    W = np.delete(W, -1, axis=0)

    # inj_solution = np.zeros((W.shape))
    if curr_len <= tmp_len:
        tmp_solution = (np.dot(W, transferdata.T)).T
        inj_solution = tmp_solution[:, 0:curr_len]
    elif curr_len > tmp_len:
        transferdata[:, tmp_len:] = 0
        inj_solution = (np.dot(W, transferdata.T)).T

    # 处理边际
    for i in range(inj_solution.shape[0]):
        for j in range(inj_solution.shape[1]):
            if problem.varTypes[j] == 0: # 0表示对应的变量是连续的；1表示是离散的
                if inj_solution[i][j] < problem.lb[j]:
                    if problem.lbin[j] == 0: # 0表示不包含该变量的下边界，1表示包含
                        inj_solution[i][j] = problem.lb[j] + 1E-6
                    else:
                        inj_solution[i][j] = problem.lb[j]
                if inj_solution[i][j] > problem.ub[j]:
                    if problem.ubin[j] == 0: # 0表示不包含该变量的上边界，1表示包含
                        inj_solution[i][j] = problem.ub[j] - 1E-6
                    else:
                        inj_solution[i][j] = problem.ub[j]
            else:
                inj_solution[i][j] = np.int(inj_solution[i][j])
                if inj_solution[i][j] < problem.lb[j]:
                    if problem.lbin[j] == 0: # 0表示不包含该变量的下边界，1表示包含
                        inj_solution[i][j] = problem.lb[j] + 1
                    else:
                        inj_solution[i][j] = problem.lb[j]
                if inj_solution[i][j] > problem.ub[j]:
                    if problem.ubin[j] == 0: # 0表示不包含该变量的上边界，1表示包含
                        inj_solution[i][j] = problem.ub[j] - 1
                    else:
                        inj_solution[i][j] = problem.ub[j]

    return inj_solution
