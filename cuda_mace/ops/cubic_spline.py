import torch


class CubicSpline(torch.nn.Module):

    def __init__(
        self, r_knots: torch.Tensor, R_out: torch.Tensor, h: float, rmax: float
    ):
        super().__init__()

        self.register_buffer("r_knots", r_knots)
        self.register_buffer("R_out", R_out)
        self.register_buffer(
            "h",
            torch.tensor(h, device="cuda"),
        )
        self.register_buffer(
            "rmax",
            torch.tensor(rmax, device="cuda"),
        )

        coefficients = self.cubic_spline_coefficients(
            self.r_knots, self.R_out, self.h.item()
        ).cuda()

        self.register_buffer(
            "coefficients",
            coefficients,
        )

        self.cuda_obj = torch.classes.cubic_spline.CubicSpline()

    def forward(self, r_trial: torch.Tensor):  # [nedges]

        out = self.cuda_obj.forward(
            r_trial, self.r_knots, self.coefficients, self.h.item(), self.rmax.item()
        )

        return out  # outputs [nedges, R_out.shape[-1]]

    def cubic_spline_coefficients(self, x, y, h):
        nchannels = y.shape[-1]

        n = x.size(0) - 1  # Number of splines

        # Compute the alpha vector for the tridiagonal system
        alpha = torch.zeros(n, nchannels, dtype=x.dtype, device=x.device)
        for i in range(1, n):
            alpha[i, :] = (3 / h * (y[i + 1, :] - y[i, :])) - (
                3 / h * (y[i, :] - y[i - 1, :])
            )

        # Set up the tridiagonal system
        l = torch.ones(n + 1, nchannels, dtype=x.dtype, device=x.device)
        mu = torch.zeros(n, nchannels, dtype=x.dtype, device=x.device)
        z = torch.zeros(n + 1, nchannels, dtype=x.dtype, device=x.device)

        # Forward pass to solve the tridiagonal system for c
        for i in range(1, n):
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h * mu[i - 1]
            mu[i] = h / l[i]
            z[i] = (alpha[i] - h * z[i - 1]) / l[i]

        # Boundary conditions for natural spline
        l[n] = 1
        z[n] = 0
        c = torch.zeros(n + 1, nchannels, dtype=x.dtype, device=x.device)

        # Back substitution to solve for c
        for j in range(n - 1, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]

        # Calculate b and d coefficients
        b = torch.zeros(n, nchannels, dtype=x.dtype, device=x.device)
        d = torch.zeros(n, nchannels, dtype=x.dtype, device=x.device)
        a = y[:-1]  # Coefficient a_i corresponds to y_i

        for i in range(n):
            b[i] = (y[i + 1] - y[i]) / h - h * (c[i + 1] + 2 * c[i]) / 3
            d[i] = (c[i + 1] - c[i]) / (3 * h)

        coefficients = torch.stack((a, b, c[:-1], d), dim=1)

        return coefficients
