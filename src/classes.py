import pandas as pd
import matplotlib.pyplot as plt

class Cell_Growth_Rates:
    def __init__(self, filepath):
            self.filepath = filepath
            self.data = pd.read_csv(self.filepath)

    def export(self, filepath=None):
        fig, ax = plt.subplots(2, 2, figsize=(10, 8), dpi=200)

            # Temperature vs. Growth Rate
        ax[0, 0].scatter(self.data["Temperature [C]"],self.data["Specific Growth Rate [d^-1]"],color="tab:blue", marker="o", s=16, alpha=0.7,edgecolors="black", linewidth=0.5)
        ax[0, 0].set_xlabel("Temperature [C]")
        ax[0, 0].set_ylabel("Growth Rate [d^-1]")
        ax[0, 0].set_title("Growth Rate vs Temperature")

        # pH vs. Growth Rate
        ax[0, 1].scatter(self.data["pH"],self.data["Specific Growth Rate [d^-1]"],color="tab:red", marker="^", s=16, alpha=0.7,edgecolors="black", linewidth=0.5)
        ax[0, 1].set_xlabel("pH")
        ax[0, 1].set_ylabel("Growth Rate [d^-1]")
        ax[0, 1].set_title("Growth Rate vs pH")

        # Glucose vs. Growth Rate
        ax[1, 0].scatter(self.data["Glucose [mM]"],self.data["Specific Growth Rate [d^-1]"],color="tab:pink", marker="D", s=16, alpha=0.7,edgecolors="black", linewidth=0.5)
        ax[1, 0].set_xlabel("Glucose [mM]")
        ax[1, 0].set_ylabel("Growth Rate [d^-1]")
        ax[1, 0].set_title("Growth Rate vs Glucose [mM]")

        # Lactate vs. Growth Rate
        ax[1, 1].scatter(self.data["Lactate [mM]"],self.data["Specific Growth Rate [d^-1]"],color="tab:orange", marker="s", s=16, alpha=0.7,edgecolors="black", linewidth=0.5)
        ax[1, 1].set_xlabel("Lactate [mM]")
        ax[1, 1].set_ylabel("Growth Rate [d^-1]")
        ax[1, 1].set_title("Growth Rate vs Lactate [mM]")

        plt.tight_layout()

        if filepath is not None:
            plt.savefig(filepath, dpi=200, bbox_inches="tight")

            plt.show()

    ### PLOT AND SAVE RESULTS ###
    # plot actual vs predicted (linear Regression)
    def plot_linear_regression(self, y_train, y_test, y_train_pred, y_test_pred, filepath, title="Linear Regression: Actual vs Predicted Growth Rate"):

        plt.figure(figsize=(8, 6), dpi=200)

        plt.scatter(y_train, y_train_pred, color="green", label="Training Data", alpha=0.7, marker="o")
        plt.scatter(y_test, y_test_pred, color="red", label="Test Data", alpha=0.7, marker="^")

        min_value = min(y_train.min(), y_test.min(), y_train_pred.min(), y_test_pred.min())
        max_value = max(y_train.max(), y_test.max(), y_train_pred.max(), y_test_pred.max())

        plt.plot([min_value, max_value], [min_value, max_value], "b--", label="Ideal Predictions", alpha=0.7)

        plt.xlabel("Actual Growth Rate")
        plt.ylabel("Predicted Growth Rate")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(filepath, dpi=200)
        plt.show()


# plot actual vs predicted (polynomial Regression)
    def plot_polynomial_regression(self, y_train, y_test, y_train_pred_poly, y_test_pred_poly, filepath,
                                   title="Polynomial Regression: Actual vs Predicted Growth Rate"):
        plt.figure(figsize=(8, 6), dpi=200)

        plt.scatter(y_train, y_train_pred_poly, color="green", label="Training Data", alpha=0.7, marker="o")
        plt.scatter(y_test, y_test_pred_poly, color="red", label="Test Data", alpha=0.7, marker="^")

        min_value = min(y_train.min(), y_test.min(), y_train_pred_poly.min(), y_test_pred_poly.min())
        max_value = max(y_train.max(), y_test.max(), y_train_pred_poly.max(), y_test_pred_poly.max())

        plt.plot([min_value, max_value], [min_value, max_value], "b--", label="Ideal Predictions", alpha=0.7)

        plt.xlabel("Actual Growth Rate")
        plt.ylabel("Predicted Growth Rate")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(filepath, dpi=200)
        plt.show()

    def plot_polynomial_deg_comparison(self, y_train, y_test, predictions, filepath):
        fig, axes = plt.subplots(2,3, figsize=(15,9), dpi=200)
        axes = axes.flatten()

        all_values = [y_train, y_test]
        for y_train_pred, y_test_pred in predictions.values():
            all_values.extend([y_train_pred, y_test_pred])

        for ax, (degree, preds) in zip(axes, sorted(predictions.items())):
            y_train_pred, y_test_pred = preds

            ax.scatter(y_train, y_train_pred, color="green",
                       label="Training Data", alpha=0.6, s=25, marker="o")
            ax.scatter(y_test, y_test_pred, color="red",
                       label="Test Data", alpha=0.8, s=35, marker="^")

            min_value = min(y_train.min(), y_test.min(), y_train_pred.min(), y_test_pred.min())
            max_value = max(y_train.max(), y_test.max(), y_train_pred.max(), y_test_pred.max())

            padding = (max_value - min_value) * 0.05
            lower = min_value - padding
            upper = max_value + padding

            ax.plot([lower, upper], [lower, upper], "b--", label="Ideal Predictions", alpha=0.8)

            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.set_title(f"Polynomial Degree {degree}")
            ax.set_xlabel("Actual Growth Rate")
            ax.set_ylabel("Predicted Growth Rate")
            ax.grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98))

        fig.suptitle("Polynomial Regression: Degrees 2-7", y=1.02)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)



# plot actual vs predicted (KNN Regression)
    def plot_knn_regression(self, y_train, y_test, y_train_pred_knn, y_test_pred_knn, filepath,
                                   title="KNN Regression: Actual vs Predicted Growth Rate"):
        plt.figure(figsize=(8, 6), dpi=200)

        plt.scatter(y_train, y_train_pred_knn, color="green", label="Training Data", alpha=0.7, marker="o")
        plt.scatter(y_test, y_test_pred_knn, color="red", label="Test Data", alpha=0.7, marker="^")

        min_value = min(y_train.min(), y_test.min(), y_train_pred_knn.min(), y_test_pred_knn.min())
        max_value = max(y_train.max(), y_test.max(), y_train_pred_knn.max(), y_test_pred_knn.max())

        plt.plot([min_value, max_value], [min_value, max_value], "b--", label="Ideal Predictions", alpha=0.7)

        plt.xlabel("Actual Growth Rate")
        plt.ylabel("Predicted Growth Rate")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(filepath, dpi=200)
        plt.show()

    def save_metrics(self, results_df, filepath):
        metrics = pd.DataFrame(results_df)
        metrics.to_csv(filepath, index=False)