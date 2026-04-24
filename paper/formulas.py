import os

import matplotlib.pyplot as plt

# ---- Enable real LaTeX ----
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)

formulas = [
    r"R\left(x_0^{(i)}\right)",
    r"x_t^{(i,\, j=1)}",
    r"x_t^{(i,\, j=2)}",
    r"x_t^{(i,\, j=M)}",
    r"\hat{x}_t^{(i,\, j=1)}",
    r"\hat{x}_t^{(i,\, j=2)}",
    r"\hat{x}_t^{(i,\, j=M)}",
    r"R\left(\hat{x}_t^{(i,\, j=1)}\right)",
    r"R\left(\hat{x}_t^{(i,\, j=2)}\right)",
    r"R\left(\hat{x}_t^{(i,\, j=M)}\right)",
]

output_dir = "latex_images"
os.makedirs(output_dir, exist_ok=True)

for i, formula in enumerate(formulas):
    fig = plt.figure()
    fig.patch.set_alpha(0)

    plt.text(0.5, 0.5, f"${formula}$", fontsize=20, ha="center", va="center")

    plt.axis("off")

    plt.savefig(
        os.path.join(output_dir, f"formula_{i + 1}.png"),
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=300,
        transparent=True,
    )

    plt.close(fig)

print("Done!")
