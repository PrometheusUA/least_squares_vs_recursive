import numpy as np
from numpy.linalg import pinv
import os.path
import matplotlib.pyplot as plt

N = 100                         # count of samples to generate
BIGNUMBER = 300                 # number to initialize diagonal of P
PTODRAW = 3                     # p of process ARMA(p;q) to draw
QTODRAW = 2                     # q of process ARMA(p;q) to draw
A0, A1, A2, A3 = 0, 0.33, 0.15, 0.11
B1, B2, B3 = 0.5, 0, 0.5        # default regression params

def leastSquares(X: np.array, Y: np.array) -> np.array:
    XtX = X.T@X
    inv = pinv(XtX)
    thirdStep = inv @ X.T
    params = thirdStep @ Y
    return params


def recursiveLeastSquares(X: np.array, Y: np.array, start: int, count: int = N) -> np.array:
    params = np.zeros((p + q + 2, 1), dtype=float)
    P = np.eye(p + q + 2, dtype=float) * BIGNUMBER

    for k in range(count - start - 1):
        xk = np.expand_dims(X[k], axis=1)
        yk = Y[k]

        gammak = P @ xk / (1 + xk.T @ P @ xk)
        params += (gammak * (yk - params.T @ xk))
        P -= gammak @ xk.T @ P
    return params


def arma(p: int, q: int, a: np.array, b: np.array, count: int = N) -> tuple[np.array, np.array]:
    v = np.random.normal(0, 1, (count, 1))
    Y = np.full((count, 1), a[0])
    aReversed = np.flip(a)  # np.array(list(reversed(list(a))))
    bReversed = np.flip(b)  # np.array(list(reversed(list(b))))

    for i in range(max(p, q)):
        Y[i] = v[i]
    for i in range(max(p, q), count):
        Y[i] += aReversed[:-1].T@Y[i - p:i] + bReversed.T@v[i - q:i+1]

    return Y, v


def armaGivenV(p: int, q: int, a: np.array, b: np.array, v: np.array) -> tuple[np.array, np.array]:
    count = v.shape[0]
    Y = np.full((count, 1), a[0])
    aReversed = np.flip(a)
    bReversed = np.flip(b)

    for i in range(max(p, q)):
        Y[i] = v[i]
    for i in range(max(p, q), count):
        Y[i] = aReversed[:-1, :].T @ Y[i - p:i, :] + bReversed.T @ v[i - q:i+1]
    return Y


class Coefs:
    @staticmethod
    def sumSquares(Y: np.array, Ymodeled: np.array) -> float:
        return np.sum((Y - Ymodeled)**2)

    @staticmethod
    def r2(Y: np.array, Ymodeled: np.array) -> float:
        return np.var(Ymodeled)/np.var(Y)

    @staticmethod
    def akaike(Y: np.array, S: float, p: int, q: int) -> float:
        return len(Y)*np.log(S) + 2*(p+q+1)

def plotCoefResults(collection, title, coefname):
    x = [i for i in range(1, 10)]

    lsRes = collection[:, :, 0].reshape(-1)
    rlsRes = collection[:, :, 1].reshape(-1)

    ax = plt.subplot(111)
    ax.set_title(title)
    ax.bar([i - 0.1 for i in x], lsRes, width=0.15, align='center', label=f"{coefname} отримане за МНК")
    ax.bar([i + 0.1 for i in x], rlsRes, width=0.15, align='center', label=f"{coefname} отримане за РМНК")

    ax.set_xticks(x)
    ax.set_xticklabels(["АРКС(1,1)", "АРКС(1,2)", "АРКС(1,3)", "АРКС(2,1)", "АРКС(2,2)", "АРКС(2,3)", "АРКС(3,1)",
                        "АРКС(3,2)", "АРКС(3,3)"])

    ax.legend()
    plt.show()

if __name__ == '__main__':
    print("Hello, dear user!\nWe are happy to see you in our least squares app!\n"
          "Do you want to input data from the console, or from the text file, or use default data? [c/t/d]: ",
        end="")
    inputType = input().strip().lower()
    while inputType != 'c' and inputType != 't' and inputType != 'd':
        print(
            "Dear user, we don`t understand you...\nPlease, type 'c' if you want to input data from console, type 't' "
            "if you want to input it from text file or type 'd' if you want to use default data: ",
            end="")
        inputType = input().strip().lower()

    if inputType == 'c':
        print("Please, input p - autoregression order (1, 2 or 3 only): ", end="")
        p = input().strip().lower()
        while p != '1' and p != '2' and p != '3':
            print("Invalid input, please reinput autoregression order p (only 1, 2 and 3 are valid): ", end="")
            p = input().strip().lower()
        p = int(p)

        print("Please, input q - moving average order (0, 1, 2 or 3 only): ", end="")
        q = input().strip().lower()
        while q != '0' and q != '1' and q != '3' and q != '2':
            print("Invalid input, please reinput moving average order q (only 1, 2 and 3 are valid): ", end="")
            q = input().strip().lower()
        q = int(q)

        print("Now, please input coefficients")
        a = []
        b = [1]
        for i in range(p+1):
            print(f'a{i} = ', end="")
            a.append(float(input()))
        for i in range(1, q+1):
            print(f'b{i} = ', end="")
            b.append(float(input()))
        a = np.array(a)
        b = np.array(b)

        Y, v = arma(p, q, a, b, N)

    elif inputType == 'd':
        p = 3
        q = 3
        a = np.array([A0,  A1,   A2,  A3])
        b = np.array([1,   B1,   B2,  B3])
        Y, v = arma(p, q, a, b)

    else:
        print("Please, enter the v-s (random epsilons) file path:")
        vPath = input()
        if not os.path.exists(vPath) and not os.path.isdir(vPath):
            raise FileNotFoundError()
        vFile = open(vPath, "r")

        print("Please, enter the y-s (values) file path:")
        yPath = input()
        if not os.path.exists(yPath) and not os.path.isdir(yPath):
            raise FileNotFoundError()
        yFile = open(yPath, "r")

        vPrev = []
        YPrev = []
        for vk in vFile:
            vPrev.append(float(vk))
        v = np.array(vPrev)
        for yk in yFile:
            YPrev.append(float(yk))
        Y = np.expand_dims(np.array(YPrev), axis=1)
        v = np.expand_dims(np.array(vPrev[-len(YPrev):]), axis=1)

        N = len(YPrev)

        vFile.close()
        yFile.close()

    print("Do you want me to draw plots? [y/n]: ", end="")
    plots = input().lower().strip()

    collectData = plots == 'y' or plots == 'yes'

    if collectData:
        collectionS = np.zeros((3, 3, 2), dtype=float)
        collectionRSq = np.zeros((3, 3, 2), dtype=float)
        collectionIKA = np.zeros((3, 3, 2), dtype=float)

    print("\nResults:\n")

    # MAIN LOOP OVER DIFFERENT ARMA(p;q) variants
    for p in range(1, 4):
        for q in range(1, 4):
            start = max(p, q)

            # Standart least squares method
            X = np.ones((N - start, p + q + 2), dtype=float)

            YCropped = np.array(Y[start:N])

            for j in range(1, p + 1):
                X[:, j] = Y[start - j:N - j, 0]
            for j in range(q + 1):
                X[:, p + j + 1] = v[start - j:N - j, 0]

            thetaLS = leastSquares(X, YCropped)

            print(f"ARMA(p={p}, q={q}), Least squares method: ")
            for k in range(p + 1):
                print(f'a{k} = {round(thetaLS[k, 0], 4)}')
            for k in range(q + 1):
                print(f'b{k} = {round(thetaLS[p + 1 + k, 0], 4)}')

            YModel = armaGivenV(p, q, thetaLS[:p+1], thetaLS[p+1:], v[:N]) #np.zeros(N - start)

            S = Coefs.sumSquares(YCropped, YModel[start:])
            Rsq = Coefs.r2(YCropped, YModel[start:])
            IKA = Coefs.akaike(YCropped, S, p, q)
            if collectData:
                collectionS[p-1, q-1, 0] = S
                collectionRSq[p-1, q-1, 0] = Rsq
                collectionIKA[p-1, q-1, 0] = IKA
            print(f'S = {round(S, 6)}')
            print(f'R^2 = {round(Rsq, 6)}')
            print(f'IKA = {round(IKA, 4)}\n')

            # Recursive least squares method
            thetaRLS = recursiveLeastSquares(X, YCropped, start, N)

            print(f"ARMA(p={p}, q={q}), Recursive least squares method: ")
            for k in range(p + 1):
                print(f'a{k} = {round(thetaRLS[k, 0], 4)}')
            for k in range(q + 1):
                print(f'b{k} = {round(thetaRLS[p + 1 + k, 0], 4)}')

            YModel = armaGivenV(p, q, thetaRLS[:p+1], thetaRLS[p+1:], v[:N])

            S = Coefs.sumSquares(YCropped, YModel[start:])
            Rsq = Coefs.r2(YCropped, YModel[start:])
            IKA = Coefs.akaike(YCropped, S, p, q)
            if collectData:
                collectionS[p-1, q-1, 1] = S
                collectionRSq[p-1, q-1, 1] = Rsq
                collectionIKA[p-1, q-1, 1] = IKA
            print(f'S = {round(S, 6)}')
            print(f'R^2 = {round(Rsq, 6)}')
            print(f'IKA = {round(IKA, 4)}\n')

    if collectData:
        p = PTODRAW
        q = QTODRAW
        start = max(p, q)

        thetaLSStat = np.zeros((N - start + 1, p + q + 2))
        thetaRLSStat = np.zeros((N - start + 1, p + q + 2))
        for k in range(start+1, N+1):
            XCropped = np.ones((k - start, p + q + 2), dtype=float)

            YCropped = np.array(Y[start:k])

            for j in range(1, p + 1):
                XCropped[:, j] = Y[start - j:k - j, 0]
            for j in range(q + 1):
                XCropped[:, p + j + 1] = v[start - j:k - j, 0]

            thetaLS = leastSquares(XCropped, YCropped)
            thetaLSStat[k - start] = thetaLS.T

            thetaRLS = recursiveLeastSquares(XCropped, YCropped, start, k)
            thetaRLSStat[k - start] = thetaRLS.T

        for k in range(p + 1):
            fig, axs = plt.subplots(1)
            fig.suptitle('Графіки перехідного процесу оцінювання')
            axs.plot(thetaLSStat[:, k], label='Значення за МНК')
            axs.plot(thetaRLSStat[:, k], label='Значення за РМНК')
            try:
                axs.plot(np.ones_like(thetaRLSStat[:, k])*a[k], label='Реальне значення')
            except NameError:
                pass
            except IndexError:
                pass
            axs.set_title(f'a{k}')
            axs.legend()
            plt.show()

        for k in range(q + 1):
            fig, axs = plt.subplots(1)
            fig.suptitle('Графіки перехідного процесу оцінювання')
            axs.plot(thetaLSStat[:, p + 1 + k], label='Значення за МНК')
            axs.plot(thetaRLSStat[:, p + 1 + k], label='Значення за РМНК')
            try:
                axs.plot(np.ones_like(thetaRLSStat[:, p + 1 + k]) * b[k], label='Реальне значення')
            except NameError:
                pass
            axs.set_title(f'b{k}')
            axs.legend()
            plt.show()

        plotCoefResults(collectionS, 'Зміна суми кваратів похибок рівняння', 'S')
        plotCoefResults(collectionRSq, 'Зміна коефецієнту детермінації', 'R^2')
        plotCoefResults(collectionIKA, 'Зміна коефецієнту Акайке', 'IKA')