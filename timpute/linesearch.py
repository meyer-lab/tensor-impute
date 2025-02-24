from .impute_helper import calcR2X


class Nesterov:
    def __init__(self, gamma=1.1, gamma_bar=1.03, eta=1.5, beta_i=0.05, beta_i_bar=1.0):
        self.gamma = gamma
        self.gamma_bar = gamma_bar
        self.eta = eta
        self.beta_i = beta_i
        self.beta_i_bar = beta_i_bar
        self.factors_old = None

    def perform(self, factors, tOrig):
        if self.factors_old is None:
            self.factors_old = factors
            self.prev_R2X = calcR2X((None, factors), tOrig)

            return self.factors_old, self.prev_R2X, 1.0

        jump = self.beta_i + 1.0

        factors_ls = [
            self.factors_old[ii] + (factors[ii] - self.factors_old[ii]) * jump
            for ii in range(len(factors))
        ]

        R2X_ls = calcR2X((None, factors_ls), tOrig)

        if R2X_ls > self.prev_R2X:
            self.factors_old = factors_ls
            self.prev_R2X = R2X_ls

            self.beta_i = min(self.beta_i_bar, self.gamma * self.beta_i)
            self.beta_i_bar = max(1.0, self.gamma_bar * self.beta_i_bar)
        else:
            self.factors_old = factors
            self.prev_R2X = calcR2X((None, factors), tOrig)

            self.beta_i_bar = self.beta_i
            self.beta_i = self.beta_i / self.eta

        return self.factors_old, self.prev_R2X, jump
